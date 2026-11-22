"""The OpenAI-compatible server, run for real (uvicorn on a local port) over the fake executor."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn

from minillm import AsyncLLMEngine, EngineConfig, LLMEngine
from minillm.entrypoints.openai_server import build_app
from minillm.executor.fake import FakeExecutor
from minillm.metrics import EngineMetrics
from minillm.sequence import Sequence
from tests.conftest import CharTokenizer

TEXT = "hello world, this is minillm speaking. "


def text_token(seq: Sequence) -> int:
    """The fake model 'says' TEXT on repeat, so outputs are readable and predictable."""
    return ord(TEXT[seq.num_output_tokens % len(TEXT)]) + 3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class LiveServer:
    def __init__(self, step_time: float = 0.0, max_pending: int = 64) -> None:
        executor = FakeExecutor(
            num_kv_blocks=1024,
            block_size=16,
            max_model_len=512,
            vocab_size=259,
            eos_token_ids=[1],
            next_token=text_token,
            cost_model=(lambda batch: step_time) if step_time else None,
        )
        self.metrics = EngineMetrics()
        self.llm = LLMEngine(
            EngineConfig(model="fake"),
            executor=executor,
            tokenizer=CharTokenizer(),
            metrics=self.metrics,
        )
        self.engine = AsyncLLMEngine(self.llm)
        app = build_app(
            self.engine,
            CharTokenizer(),
            model_name="fake-model",
            max_pending_requests=max_pending,
            metrics=self.metrics,
        )
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> LiveServer:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(10)

    def metric(self, name: str, **labels: str) -> float:
        value = self.metrics.registry.get_sample_value(name, labels or None)
        return 0.0 if value is None else value

    def wait_idle(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.llm.num_unfinished_requests or self.engine.num_active_requests:
            if time.monotonic() > deadline:
                raise AssertionError("requests still in flight")
            time.sleep(0.02)


@pytest.fixture
def server() -> Iterator[LiveServer]:
    with LiveServer() as s:
        yield s


@pytest.fixture
def slow_server() -> Iterator[LiveServer]:
    with LiveServer(step_time=0.01, max_pending=1) as s:
        yield s


def sse_events(response: httpx.Response) -> list[Any]:
    events: list[Any] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_completion(server: LiveServer) -> None:
    r = httpx.post(
        f"{server.url}/v1/completions",
        json={"model": "fake-model", "prompt": "Hi", "max_tokens": 11, "temperature": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == 3  # BOS + "H" + "i"
    assert body["usage"]["completion_tokens"] == 11
    assert r.headers["x-request-id"] == body["id"]


def test_streaming_completion(server: LiveServer) -> None:
    payload = {
        "prompt": "Hi",
        "max_tokens": 11,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with httpx.stream("POST", f"{server.url}/v1/completions", json=payload) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
    assert events[-1] == "[DONE]"
    usage_event = events[-2]
    assert usage_event["choices"] == [] and usage_event["usage"]["completion_tokens"] == 11
    chunks = events[:-2]
    assert "".join(e["choices"][0]["text"] for e in chunks) == "hello world"
    assert len(chunks) > 1  # actually streamed, not one blob
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(e["choices"][0]["finish_reason"] is None for e in chunks[:-1])


def test_chat_completion(server: LiveServer) -> None:
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    r = httpx.post(f"{server.url}/v1/chat/completions", json={"messages": msgs, "max_tokens": 5})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}

    payload = {"messages": msgs, "max_tokens": 5, "stream": True}
    with httpx.stream("POST", f"{server.url}/v1/chat/completions", json=payload) as r:
        events = sse_events(r)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[-1] == "[DONE]"
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:-1])
    assert content == "hello"
    assert events[-2]["choices"][0]["finish_reason"] == "length"


def test_chat_default_max_tokens_fills_the_context(server: LiveServer) -> None:
    r = httpx.post(
        f"{server.url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
        timeout=30,
    )
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == 512


@pytest.mark.parametrize("stream", [False, True])
def test_stop_strings_are_excluded_from_output(server: LiveServer, stream: bool) -> None:
    payload = {"prompt": "Hi", "max_tokens": 30, "stop": ["xyz", ", th"], "stream": stream}
    if stream:
        with httpx.stream("POST", f"{server.url}/v1/completions", json=payload) as r:
            events = [e for e in sse_events(r) if e != "[DONE]"]
        text = "".join(e["choices"][0]["text"] for e in events)
        reason = events[-1]["choices"][0]["finish_reason"]
    else:
        choice = httpx.post(f"{server.url}/v1/completions", json=payload).json()["choices"][0]
        text, reason = choice["text"], choice["finish_reason"]
    assert text == "hello world"
    assert reason == "stop"
    server.wait_idle()  # the engine-side request was aborted once the stop string matched


def test_token_id_prompt(server: LiveServer) -> None:
    r = httpx.post(f"{server.url}/v1/completions", json={"prompt": [2, 75, 108], "max_tokens": 5})
    assert r.json()["choices"][0]["text"] == "hello"


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"prompt": "x" * 500, "max_tokens": 100}, "maximum context length is 512"),
        ({"prompt": [], "max_tokens": 4}, "empty"),
        ({"prompt": [9999], "max_tokens": 4}, "token ids must be in"),
        ({"prompt": "hi", "n": 2}, "n > 1"),
        ({"prompt": "hi", "logprobs": 5}, "logprobs"),
        ({"prompt": "hi", "temperature": -1}, "temperature"),
        ({"max_tokens": 4}, "prompt"),
    ],
)
def test_bad_requests_get_400(server: LiveServer, payload: dict[str, Any], fragment: str) -> None:
    r = httpx.post(f"{server.url}/v1/completions", json=payload)
    assert r.status_code == 400
    assert fragment in r.json()["error"]["message"]


def test_health_models_and_metrics(server: LiveServer) -> None:
    assert httpx.get(f"{server.url}/health").json() == {"status": "ok"}
    models = httpx.get(f"{server.url}/v1/models").json()
    assert models["data"][0]["id"] == "fake-model"
    assert models["data"][0]["max_model_len"] == 512
    httpx.post(f"{server.url}/v1/completions", json={"prompt": "Hi", "max_tokens": 4})
    text = httpx.get(f"{server.url}/metrics").text
    for name in (
        "minillm_time_to_first_token_seconds_count 1.0",
        "minillm_e2e_request_latency_seconds_count 1.0",
        "minillm_generation_tokens_total 4.0",
        'minillm_requests_finished_total{finish_reason="length"} 1.0',
        "minillm_inter_token_latency_seconds_count 3.0",
    ):
        assert name in text, name


async def test_many_concurrent_requests(server: LiveServer) -> None:
    async with httpx.AsyncClient(base_url=server.url, timeout=30) as client:

        async def one(i: int) -> str:
            r = await client.post("/v1/completions", json={"prompt": f"req {i}", "max_tokens": 11})
            text: str = r.json()["choices"][0]["text"]
            return text

        texts = await asyncio.gather(*(one(i) for i in range(32)))
    assert texts == ["hello world"] * 32
    assert server.llm.block_manager.num_free_blocks == server.llm.block_manager.num_blocks


def test_client_disconnect_aborts_streaming_request(slow_server: LiveServer) -> None:
    payload = {"prompt": "Hi", "max_tokens": 400, "stream": True}
    with httpx.stream("POST", f"{slow_server.url}/v1/completions", json=payload) as r:
        lines = r.iter_lines()
        received = [next(lines) for _ in range(6)]
    assert any(line.startswith("data: ") for line in received)
    slow_server.wait_idle(timeout=3)  # 400 tokens at 10 ms/step would take 4 s
    assert slow_server.metric("minillm_requests_finished_total", finish_reason="abort") == 1
    bm = slow_server.llm.block_manager
    assert bm.num_free_blocks == bm.num_blocks


def test_client_disconnect_aborts_non_streaming_request(slow_server: LiveServer) -> None:
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(
            f"{slow_server.url}/v1/completions",
            json={"prompt": "Hi", "max_tokens": 400},
            timeout=0.3,
        )
    slow_server.wait_idle(timeout=3)
    assert slow_server.metric("minillm_requests_finished_total", finish_reason="abort") == 1


def test_full_queue_returns_429(slow_server: LiveServer) -> None:
    payload = {"prompt": "Hi", "max_tokens": 200, "stream": True}
    with httpx.stream("POST", f"{slow_server.url}/v1/completions", json=payload) as r:
        # Keep a reference: a garbage-collected line iterator makes httpx drop
        # the connection, which the server (correctly) treats as a disconnect.
        lines = r.iter_lines()
        next(lines)  # the first request is now in flight
        rejected = httpx.post(f"{slow_server.url}/v1/completions", json={"prompt": "x"})
        assert slow_server.engine.num_active_requests == 1
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"
    assert slow_server.metric("minillm_requests_rejected_total", reason="queue_full") == 1
