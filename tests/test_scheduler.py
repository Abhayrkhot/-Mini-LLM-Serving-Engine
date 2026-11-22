"""Scheduling behaviour, exercised through LLMEngine with the deterministic fake executor."""

from __future__ import annotations

import pytest

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.executor.fake import FakeExecutor


def make_engine(
    *,
    num_kv_blocks: int = 512,
    block_size: int = 4,
    max_model_len: int = 256,
    eos: list[int] | None = None,
    **config,
) -> tuple[LLMEngine, FakeExecutor]:
    executor = FakeExecutor(
        num_kv_blocks=num_kv_blocks,
        block_size=block_size,
        max_model_len=max_model_len,
        vocab_size=1000,
        eos_token_ids=eos,
    )
    cfg = EngineConfig(model="fake", **config)
    return LLMEngine(cfg, executor=executor), executor


def run_all(engine: LLMEngine) -> None:
    for _ in range(10_000):
        if not engine.has_unfinished_requests():
            return
        engine.step()
    raise AssertionError("engine did not finish")


def params(n: int = 8, **kw) -> SamplingParams:
    return SamplingParams(max_tokens=n, temperature=0, ignore_eos=kw.pop("ignore_eos", True), **kw)


def test_new_requests_join_a_running_batch():
    engine, ex = make_engine()
    engine.add_request([5] * 6, params(20), "a")
    for _ in range(3):
        engine.step()
    engine.add_request([7] * 6, params(20), "b")
    engine.step()
    step = {rid: n for rid, n, _ in ex.history[-1]}
    assert step == {"a": 1, "b": 6}  # a decodes while b's whole prompt is prefilled


def test_token_budget_is_never_exceeded():
    engine, ex = make_engine(max_num_batched_tokens=16)
    for i in range(10):
        engine.add_request([3 + i] * (5 + 7 * i), params(12), f"r{i}")
    run_all(engine)
    assert all(sum(n for _, n, _ in step) <= 16 for step in ex.history)


def test_long_prompt_is_chunked_and_does_not_stall_decodes():
    engine, ex = make_engine(max_num_batched_tokens=32)
    engine.add_request([4] * 5, params(30), "decoder")
    engine.step()  # prefill
    engine.add_request([9] * 100, params(4), "long")
    chunks = []
    for _ in range(4):
        engine.step()
        step = {rid: (n, sample) for rid, n, sample in ex.history[-1]}
        assert step["decoder"] == (1, True)  # the decoder gets a token every step
        chunks.append(step["long"])
    # 100 prompt tokens in chunks of the budget left after the decode; only the
    # last chunk produces a token.
    assert chunks == [(31, False), (31, False), (31, False), (7, True)]


def test_without_chunked_prefill_prompts_wait_for_a_full_budget():
    engine, ex = make_engine(
        max_model_len=64, max_num_batched_tokens=64, enable_chunked_prefill=False
    )
    engine.add_request([4] * 10, params(30), "a")
    engine.step()
    engine.add_request([9] * 10, params(30), "b")
    engine.step()
    engine.add_request([8] * 62, params(1), "c")
    engine.step()
    # Two decodes + 62 prompt tokens = 64: fits the budget exactly.
    assert {r: n for r, n, _ in ex.history[-1]} == {"a": 1, "b": 1, "c": 62}
    engine.add_request([7] * 63, params(1), "d")
    engine.step()
    # Two decodes + 63 = 65 > 64: d must wait rather than be split.
    assert {r: n for r, n, _ in ex.history[-1]} == {"a": 1, "b": 1}
    with pytest.raises(ValueError):
        make_engine(max_model_len=64, max_num_batched_tokens=32, enable_chunked_prefill=False)


def test_max_num_seqs_caps_the_running_batch():
    engine, ex = make_engine(max_num_seqs=3)
    for i in range(8):
        engine.add_request([3 + i] * 4, params(5), f"r{i}")
    run_all(engine)
    assert max(len(step) for step in ex.history) == 3


def test_preemption_recomputes_and_preserves_outputs():
    prompts = [[10 + i] * (20 + 3 * i) for i in range(6)]
    roomy, _ = make_engine(num_kv_blocks=512)
    tight, _ = make_engine(num_kv_blocks=40, max_model_len=128)  # 160 tokens for ~6 x 60
    reference = roomy.generate(prompts, params(40))
    results = tight.generate(prompts, params(40))
    assert tight.scheduler.num_preemptions > 0
    assert sum(r.num_preemptions for r in results) == tight.scheduler.num_preemptions
    assert [r.output_token_ids for r in results] == [r.output_token_ids for r in reference]
    assert tight.block_manager.num_free_blocks == tight.block_manager.num_blocks


def test_preempted_requests_resume_before_new_ones():
    engine, _ = make_engine(num_kv_blocks=20, max_model_len=64)
    for i in range(4):
        engine.add_request([3 + i] * 15, params(30), f"r{i}")
    while engine.scheduler.num_preemptions == 0:
        engine.step()
    preempted = engine.scheduler.waiting[0].request_id
    engine.add_request([99] * 4, params(2), "late")
    assert [s.request_id for s in engine.scheduler.waiting][-1] == "late"
    assert engine.scheduler.waiting[0].request_id == preempted


def test_stop_conditions():
    engine, _ = make_engine(eos=[0])
    # Force specific tokens so EOS / stop tokens actually occur.
    engine.executor._next_token = lambda seq: [11, 12, 0, 13][seq.num_output_tokens % 4]  # type: ignore[attr-defined]
    r_eos = engine.generate([[5, 5]], SamplingParams(max_tokens=10, temperature=0))[0]
    assert r_eos.output_token_ids == [11, 12, 0] and r_eos.finish_reason == "stop"
    r_stop = engine.generate([[5, 5]], SamplingParams(max_tokens=10, stop_token_ids=[12]))[0]
    assert r_stop.output_token_ids == [11, 12] and r_stop.finish_reason == "stop"
    r_ign = engine.generate([[5, 5]], SamplingParams(max_tokens=6, ignore_eos=True))[0]
    assert len(r_ign.output_token_ids) == 6 and r_ign.finish_reason == "length"


def test_max_model_len_ends_generation():
    engine, _ = make_engine(max_model_len=32)
    res = engine.generate([[5] * 30], params(100))[0]
    assert len(res.output_token_ids) == 2 and res.finish_reason == "length"


def test_abort_frees_blocks_for_waiting_and_running():
    engine, _ = make_engine(max_num_seqs=1)
    engine.add_request([5] * 9, params(50), "running")
    engine.add_request([6] * 9, params(50), "waiting")
    engine.step()
    assert engine.abort_request("waiting")
    assert engine.abort_request("running")
    assert not engine.abort_request("running")
    assert not engine.has_unfinished_requests()
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_shared_system_prompt_hits_the_prefix_cache():
    engine, _ = make_engine()
    system = list(range(10, 42))  # 32 tokens = 8 blocks
    engine.generate([[*system, 1, 2]], params(4))
    results = engine.generate([[*system, 3 + i] for i in range(5)], params(4))
    assert all(r.num_cached_tokens == 32 for r in results)
    bm = engine.block_manager
    assert bm.prefix_hit_tokens == 5 * 32


def test_invalid_requests_are_rejected():
    engine, _ = make_engine(max_model_len=16)
    with pytest.raises(ValueError):
        engine.add_request([5] * 16, params())  # no room for a single output token
    with pytest.raises(ValueError):
        engine.add_request([], params())
    with pytest.raises(ValueError):
        engine.add_request([5, 5000], params())  # out of vocabulary
    engine.add_request([5], params(), "dup")
    with pytest.raises(ValueError):
        engine.add_request([5], params(), "dup")
