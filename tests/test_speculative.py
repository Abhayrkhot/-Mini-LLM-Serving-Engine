"""Speculative decoding with n-gram drafts must not change what is generated."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.executor.fake import FakeExecutor


def echo_token(seq) -> int:  # type: ignore[no-untyped-def]
    """Repeats what came three tokens earlier, with a twist now and then: drafts are
    often right, but not always, and EOS (0) or 1 shows up occasionally."""
    t = seq.token_ids
    if len(t) % 7 == 0:
        return (t[-1] * 13 + len(t)) % 40
    return t[-3] if len(t) >= 3 else (t[-1] + 1) % 40


def make_engine(
    spec: int, *, next_token=echo_token, **config: object
) -> tuple[LLMEngine, FakeExecutor]:  # type: ignore[no-untyped-def]
    executor = FakeExecutor(
        num_kv_blocks=int(config.pop("num_kv_blocks", 512)),  # type: ignore[call-overload]
        block_size=4,
        max_model_len=int(config.pop("max_model_len", 128)),  # type: ignore[call-overload]
        vocab_size=40,
        eos_token_ids=[0],
        next_token=next_token,
        model_kv_cache=True,
    )
    cfg = EngineConfig(model="fake", speculative_tokens=spec, **config)  # type: ignore[arg-type]
    return LLMEngine(cfg, executor=executor), executor


def run(engine: LLMEngine, prompts: list[list[int]], params: list[SamplingParams]):  # type: ignore[no-untyped-def]
    results = engine.generate(prompts, params)
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    return [(r.output_token_ids, r.finish_reason) for r in results]


workloads = st.fixed_dictionaries(
    {
        "prompts": st.lists(
            st.lists(st.integers(2, 39), min_size=1, max_size=30), min_size=1, max_size=8
        ),
        "max_tokens": st.lists(st.integers(1, 40), min_size=8, max_size=8),
        "ignore_eos": st.lists(st.booleans(), min_size=8, max_size=8),
        "draft": st.sampled_from([1, 2, 4, 8]),
        "budget": st.sampled_from([6, 16, 512]),
        "num_kv_blocks": st.sampled_from([24, 512]),
    }
)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(w=workloads)
def test_speculation_matches_plain_decoding(w: dict) -> None:  # type: ignore[type-arg]
    """Draft lengths, token budgets, preemption, EOS and length limits, any mix."""
    params = [
        SamplingParams(max_tokens=w["max_tokens"][i], temperature=0, ignore_eos=w["ignore_eos"][i])
        for i in range(len(w["prompts"]))
    ]
    config = dict(
        num_kv_blocks=w["num_kv_blocks"], max_model_len=64, max_num_batched_tokens=w["budget"]
    )
    plain = run(make_engine(0, **config)[0], w["prompts"], params)
    spec = run(make_engine(w["draft"], **config)[0], w["prompts"], params)
    assert spec == plain


def test_accepted_drafts_save_steps() -> None:
    """A model that repeats its prompt: nearly every draft is right."""

    def copy_prompt(seq) -> int:  # type: ignore[no-untyped-def]
        return seq.token_ids[seq.num_output_tokens % seq.num_prompt_tokens]

    prompt = [5, 9, 11, 3, 17, 23, 8]
    params = SamplingParams(max_tokens=60, temperature=0, ignore_eos=True)
    outputs, steps = {}, {}
    for k in (0, 4):
        engine, ex = make_engine(k, next_token=copy_prompt)
        outputs[k] = engine.generate([prompt], params)[0].output_token_ids
        steps[k] = len(ex.history)
        if k:
            sched = engine.scheduler
            assert sched.spec_accepted / sched.spec_drafted > 0.9
    assert outputs[4] == outputs[0]
    assert steps[4] < steps[0] / 3  # up to 5 tokens per step instead of 1


def test_stop_token_inside_accepted_drafts() -> None:
    def cycle(seq) -> int:  # type: ignore[no-untyped-def]
        return [7, 8, 9, 0][seq.num_output_tokens % 4]  # EOS (0) every fourth token

    prompt = [7, 8, 9, 0, 7, 8, 9]  # drafts will propose "0 7 8 9" straight away
    for k in (0, 6):
        engine, _ = make_engine(k, next_token=cycle)
        out = engine.generate([prompt], SamplingParams(max_tokens=20, temperature=0))[0]
        assert out.output_token_ids == [7, 8, 9, 0] and out.finish_reason == "stop"


def test_sampled_requests_do_not_draft() -> None:
    engine, ex = make_engine(4)
    engine.generate([[1, 2, 3, 1, 2, 3]], SamplingParams(max_tokens=10, temperature=0.7))
    assert engine.scheduler.spec_drafted == 0
    assert all(n == 1 for step in ex.history[1:] for _, n, _ in step)


def test_speculation_turns_async_scheduling_off() -> None:
    assert make_engine(4)[0].async_scheduling is False
    assert make_engine(0)[0].async_scheduling is True
