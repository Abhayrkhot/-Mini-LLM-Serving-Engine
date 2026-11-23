"""The engine running in its own process (EngineCoreClient), over the fake executor."""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from minillm import EngineCoreClient, EngineSpec, SamplingParams
from minillm.async_engine import EngineDeadError
from minillm.executor.fake import build_fake_engine
from minillm.metrics import EngineMetrics

PARAMS = SamplingParams(max_tokens=20, temperature=0, ignore_eos=True)
SMALL = dict(num_kv_blocks=256, block_size=4, max_model_len=128)


def spec(**kwargs: object) -> EngineSpec:
    return EngineSpec(build_fake_engine, dict(kwargs))


async def collect(
    client: EngineCoreClient, rid: str, prompt: list[int], params: SamplingParams = PARAMS
) -> list[int]:
    return [out.token_id async for out in client.generate(rid, prompt, params)]


def sample(metrics: EngineMetrics, name: str, **labels: str) -> float:
    return metrics.registry.get_sample_value(name, labels or None) or 0.0


async def eventually(check: object, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():  # type: ignore[operator]
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


async def test_streams_match_offline_generation() -> None:
    prompts = [[5 + i] * (3 + i) for i in range(12)]
    offline = build_fake_engine(executor=SMALL).generate(prompts, PARAMS)
    client = EngineCoreClient(spec(executor=SMALL))
    await client.start()
    try:
        assert (client.max_model_len, client.vocab_size) == (128, 259)
        assert client.pid != os.getpid()
        streamed = await asyncio.gather(
            *(collect(client, f"r{i}", p) for i, p in enumerate(prompts))
        )
    finally:
        await client.stop()
    assert streamed == [r.output_token_ids for r in offline]


async def test_metrics_cross_the_process_boundary_and_aborts_free_the_request() -> None:
    metrics = EngineMetrics()
    client = EngineCoreClient(spec(executor=SMALL, step_time_s=0.005), metrics=metrics)
    await client.start()
    try:
        await collect(client, "done", [5, 6, 7], SamplingParams(max_tokens=4, ignore_eos=True))
        stream = client.generate("cut", [5, 6, 7], SamplingParams(max_tokens=100, ignore_eos=True))
        for _ in range(3):
            await stream.__anext__()
        await stream.aclose()  # the client went away mid-generation
        await eventually(
            lambda: sample(metrics, "minillm_requests_finished_total", finish_reason="abort") == 1
        )
        assert sample(metrics, "minillm_requests_finished_total", finish_reason="length") == 1
        assert sample(metrics, "minillm_time_to_first_token_seconds_count") == 2
        assert sample(metrics, "minillm_generation_tokens_total") >= 7
        await eventually(lambda: sample(metrics, "minillm_num_requests_running") == 0)
        assert client.num_active_requests == 0
    finally:
        await client.stop()


async def test_invalid_request_is_rejected_and_engine_stays_up() -> None:
    client = EngineCoreClient(spec(executor=SMALL))
    await client.start()
    try:
        with pytest.raises(ValueError, match="max_model_len"):
            await collect(client, "too-long", [5] * 200)
        assert client.is_healthy
        assert len(await collect(client, "fine", [5, 6])) == 20
    finally:
        await client.stop()


async def test_engine_crash_fails_in_flight_and_new_requests() -> None:
    client = EngineCoreClient(spec(executor=dict(SMALL, crash_after_steps=3)))
    await client.start()
    try:
        with pytest.raises(EngineDeadError, match="illegal memory access"):
            await collect(client, "a", [5, 5])
        assert not client.is_healthy
        with pytest.raises(EngineDeadError):
            await collect(client, "b", [6, 6])
    finally:
        await client.stop()


async def test_killed_engine_process_is_detected() -> None:
    client = EngineCoreClient(spec(executor=SMALL, step_time_s=0.01))
    await client.start()
    try:
        stream = client.generate("r", [5, 6], SamplingParams(max_tokens=100, ignore_eos=True))
        await stream.__anext__()
        assert client.pid is not None
        os.kill(client.pid, signal.SIGKILL)  # e.g. the kernel's OOM killer
        with pytest.raises(EngineDeadError):
            async for _ in stream:
                pass
        assert not client.is_healthy
    finally:
        await client.stop()


def test_startup_failure_is_reported() -> None:
    client = EngineCoreClient(spec(config={"max_num_seqs": 0}))  # EngineConfig rejects this
    with pytest.raises(RuntimeError, match="max_num_seqs"):
        client.launch()
