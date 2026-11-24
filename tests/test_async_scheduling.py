"""Async scheduling (step n+1 launched before step n's tokens reach the CPU) must
not change what the engine generates, only when the CPU learns about it."""

from __future__ import annotations

from collections import Counter

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.executor.fake import FakeExecutor


def next_token(seq) -> int:  # type: ignore[no-untyped-def]
    # Depends on the whole context, and hits EOS (0) or 1 now and then.
    return (seq.token_ids[-1] * 31 + len(seq.token_ids) * 7 + seq.num_prompt_tokens) % 40


def make_engine(
    async_scheduling: bool,
    *,
    num_kv_blocks: int = 512,
    block_size: int = 4,
    max_model_len: int = 256,
    **config: object,
) -> tuple[LLMEngine, FakeExecutor]:
    executor = FakeExecutor(
        num_kv_blocks=num_kv_blocks,
        block_size=block_size,
        max_model_len=max_model_len,
        vocab_size=40,
        eos_token_ids=[0],
        next_token=next_token,
        model_kv_cache=True,
    )
    cfg = EngineConfig(model="fake", async_scheduling=async_scheduling, **config)  # type: ignore[arg-type]
    return LLMEngine(cfg, executor=executor), executor


def run(engine: LLMEngine, prompts: list[list[int]], params: list[SamplingParams]):  # type: ignore[no-untyped-def]
    results = engine.generate(prompts, params)
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    assert not engine.scheduler.running and not engine.scheduler.waiting
    assert not engine.has_unfinished_requests()
    return [(r.output_token_ids, r.finish_reason) for r in results]


workloads = st.fixed_dictionaries(
    {
        "prompts": st.lists(
            st.lists(st.integers(2, 39), min_size=1, max_size=40), min_size=1, max_size=12
        ),
        "max_tokens": st.lists(st.integers(1, 30), min_size=12, max_size=12),
        "ignore_eos": st.lists(st.booleans(), min_size=12, max_size=12),
        "stop": st.lists(st.sampled_from([(), (1,), (5, 7)]), min_size=12, max_size=12),
        "budget": st.sampled_from([8, 16, 64, 512]),
        "num_kv_blocks": st.sampled_from([24, 40, 512]),
        "max_num_seqs": st.sampled_from([1, 3, 64]),
    }
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(w=workloads)
def test_async_matches_sync(w: dict) -> None:  # type: ignore[type-arg]
    """Chunked prefill, preemption, EOS, stop tokens and length limits, any mix."""
    params = [
        SamplingParams(
            max_tokens=w["max_tokens"][i],
            temperature=0,
            ignore_eos=w["ignore_eos"][i],
            stop_token_ids=list(w["stop"][i]),
        )
        for i in range(len(w["prompts"]))
    ]
    config = dict(
        num_kv_blocks=w["num_kv_blocks"],
        max_model_len=64,
        max_num_batched_tokens=w["budget"],
        max_num_seqs=w["max_num_seqs"],
    )
    sync_out = run(make_engine(False, **config)[0], w["prompts"], params)
    async_out = run(make_engine(True, **config)[0], w["prompts"], params)
    assert async_out == sync_out


def steps_per_request(ex: FakeExecutor) -> Counter[str]:
    return Counter(rid for step in ex.history for rid, _, _ in step)


def test_a_stop_token_costs_exactly_one_discarded_step() -> None:
    def script(seq) -> int:  # type: ignore[no-untyped-def]
        return [11, 12, 0, 13, 14][seq.num_output_tokens]

    counts = {}
    for mode in (False, True):
        engine, ex = make_engine(mode)
        ex._next_token = script
        out = engine.generate([[5, 5]], SamplingParams(max_tokens=10, temperature=0))[0]
        assert out.output_token_ids == [11, 12, 0] and out.finish_reason == "stop"
        counts[mode] = steps_per_request(ex)["req-0"]
    assert counts == {False: 3, True: 4}  # async computed one token past EOS, then dropped it


def test_length_limits_cost_no_extra_steps() -> None:
    for mode in (False, True):
        engine, ex = make_engine(mode, max_model_len=80)
        engine.generate(
            [[5, 5, 5], [6] * 60],
            [
                SamplingParams(max_tokens=7, ignore_eos=True),
                SamplingParams(max_tokens=100, ignore_eos=True),  # stops at max_model_len
            ],
        )
        assert steps_per_request(ex) == {"req-0": 7, "req-1": 80 - 60}


def test_steps_overlap_with_the_previous_steps_results() -> None:
    """Step n+1 is launched before step n's tokens are applied, so it is
    scheduled with step n's token still pending."""
    engine, ex = make_engine(True)
    engine.add_request([5, 6, 7], SamplingParams(max_tokens=5, ignore_eos=True), "a")
    assert engine.step() == []  # prefill launched, nothing back yet
    seq = engine._requests["a"]
    assert seq.num_pending_tokens == 1 and seq.num_output_tokens == 0
    out = engine.step()  # launches the first decode, then collects the prefill's token
    assert [o.token_id for o in out] == [next_token(_view([5, 6, 7]))]
    assert seq.num_pending_tokens == 1  # the decode's token is in flight
    assert ex.history[-1] == [("a", 1, True)]


def test_abort_while_a_step_is_in_flight() -> None:
    prompts = [[3 + i] * (4 + i) for i in range(4)]
    params = SamplingParams(max_tokens=20, ignore_eos=True)
    reference = make_engine(False)[0].generate(prompts, params)

    engine, _ = make_engine(True)
    for i, p in enumerate(prompts):
        engine.add_request(p, params, f"r{i}")
    outputs: dict[str, list[int]] = {f"r{i}": [] for i in range(4)}
    for step in range(200):
        for o in engine.step():
            outputs[o.request_id].append(o.token_id)
        if step == 5:
            assert engine._inflight is not None
            assert engine.abort_request("r1")
        if not engine.has_unfinished_requests():
            break
    for i in (0, 2, 3):  # the others are unaffected
        assert outputs[f"r{i}"] == reference[i].output_token_ids
    assert outputs["r1"] == reference[1].output_token_ids[: len(outputs["r1"])]
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks


def test_aborting_everything_collects_the_step_in_flight() -> None:
    engine, _ = make_engine(True)
    engine.add_request([5] * 6, SamplingParams(max_tokens=50, ignore_eos=True), "a")
    engine.step()
    engine.step()
    assert engine._inflight is not None
    engine.abort_request("a")
    assert engine._inflight is None and not engine.has_unfinished_requests()


def test_preempted_mid_flight_requests_finish_correctly() -> None:
    prompts = [[10 + i] * (20 + 3 * i) for i in range(6)]
    params = SamplingParams(max_tokens=40, temperature=0)
    reference = make_engine(False)[0].generate(prompts, params)
    engine, _ = make_engine(True, num_kv_blocks=40, max_model_len=128)
    results = engine.generate(prompts, params)
    assert engine.scheduler.num_preemptions > 0
    assert [r.output_token_ids for r in results] == [r.output_token_ids for r in reference]


class _view:  # what next_token sees for a bare prompt
    def __init__(self, tokens: list[int]) -> None:
        self.token_ids = tokens
        self.num_prompt_tokens = len(tokens)
