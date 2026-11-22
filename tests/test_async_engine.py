from __future__ import annotations

import asyncio

import pytest

from minillm import AsyncLLMEngine, EngineConfig, LLMEngine, SamplingParams
from minillm.async_engine import EngineDeadError
from minillm.executor.fake import FakeExecutor
from minillm.scheduler import ScheduledSeq

PARAMS = SamplingParams(max_tokens=20, temperature=0, ignore_eos=True)


def make(executor: FakeExecutor | None = None) -> AsyncLLMEngine:
    executor = executor or FakeExecutor(num_kv_blocks=256, block_size=4, max_model_len=128)
    return AsyncLLMEngine(LLMEngine(EngineConfig(model="fake"), executor=executor))


async def collect(engine: AsyncLLMEngine, rid: str, prompt: list[int]) -> list[int]:
    return [out.token_id async for out in engine.generate(rid, prompt, PARAMS)]


async def test_concurrent_streams_match_offline_generation():
    prompts = [[5 + i] * (3 + i) for i in range(12)]
    offline = LLMEngine(
        EngineConfig(model="fake"),
        executor=FakeExecutor(num_kv_blocks=256, block_size=4, max_model_len=128),
    ).generate(prompts, PARAMS)
    engine = make()
    await engine.start()
    try:
        streamed = await asyncio.gather(
            *(collect(engine, f"r{i}", p) for i, p in enumerate(prompts))
        )
    finally:
        await engine.stop()
    assert streamed == [r.output_token_ids for r in offline]


async def test_closing_a_stream_aborts_the_request():
    engine = make(
        FakeExecutor(num_kv_blocks=256, block_size=4, max_model_len=128, cost_model=lambda b: 0.005)
    )
    await engine.start()
    try:
        stream = engine.generate("r", [5, 6, 7], SamplingParams(max_tokens=100, ignore_eos=True))
        got = [await stream.__anext__() for _ in range(3)]
        await stream.aclose()
        assert len(got) == 3
        for _ in range(200):
            if not engine.engine.has_unfinished_requests():
                break
            await asyncio.sleep(0.01)
        assert not engine.engine.has_unfinished_requests()
        bm = engine.engine.block_manager
        assert bm.num_free_blocks == bm.num_blocks
    finally:
        await engine.stop()


async def test_invalid_request_raises_in_the_caller():
    engine = make()
    await engine.start()
    try:
        with pytest.raises(ValueError, match="max_model_len"):
            await collect(engine, "r", [5] * 200)
        assert engine.is_healthy
    finally:
        await engine.stop()


class CrashingExecutor(FakeExecutor):
    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        if len(self.history) >= 3:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return super().execute(scheduled)


async def test_engine_crash_fails_in_flight_and_new_requests():
    engine = make(CrashingExecutor(num_kv_blocks=256, block_size=4, max_model_len=128))
    await engine.start()
    try:
        with pytest.raises(EngineDeadError):
            await collect(engine, "a", [5, 5])
        assert not engine.is_healthy
        with pytest.raises(EngineDeadError):
            await collect(engine, "b", [6, 6])
    finally:
        await engine.stop()
