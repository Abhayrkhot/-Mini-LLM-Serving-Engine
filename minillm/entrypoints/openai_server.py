"""OpenAI-compatible HTTP server: /v1/completions, /v1/chat/completions, /v1/models, /health, /metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from minillm.async_engine import AsyncLLMEngine, EngineClient, EngineDeadError
from minillm.config import EngineConfig
from minillm.engine import LLMEngine
from minillm.engine_core import EngineCoreClient, EngineSpec, build_engine
from minillm.entrypoints.protocol import ChatCompletionRequest, CompletionRequest, _SamplingFields
from minillm.metrics import EngineMetrics
from minillm.models.loader import resolve_model_path
from minillm.sampling import SamplingParams
from minillm.tokenizer import IncrementalDetokenizer, StopStringMatcher, Tokenizer, load_tokenizer

logger = logging.getLogger("minillm.server")


@dataclass
class _Chunk:
    text: str
    finish_reason: str | None
    usage: dict[str, Any] | None


class _ClientDisconnected(Exception):
    pass


def _error(
    status: int, message: str, err_type: str = "invalid_request_error", **headers: str
) -> JSONResponse:
    body = {"error": {"message": message, "type": err_type, "code": status}}
    return JSONResponse(body, status_code=status, headers=headers or None)


def build_app(
    engine: EngineClient,
    tokenizer: Tokenizer,
    *,
    model_name: str,
    max_pending_requests: int = 1024,
    metrics: EngineMetrics | None = None,
) -> FastAPI:
    max_model_len = engine.max_model_len  # an engine process must be launched first
    vocab_size = engine.vocab_size

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="minillm", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(400, str(exc.errors()))

    # ------------------------------------------------------------------ plain endpoints

    @app.get("/health")
    async def health() -> Response:
        if engine.is_healthy:
            return JSONResponse({"status": "ok"})
        return _error(503, "engine is not running", "server_error")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "minillm",
                    "max_model_len": max_model_len,
                }
            ],
        }

    if metrics is not None:

        @app.get("/metrics")
        async def prometheus() -> Response:
            return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ helpers

    def reject(
        status: int, message: str, reason: str, err_type: str, **headers: str
    ) -> JSONResponse:
        if metrics is not None:
            metrics.requests_rejected.labels(reason).inc()
        return _error(status, message, err_type, **headers)

    def admit(prompt_ids: list[int], max_tokens: int) -> JSONResponse | None:
        if not engine.is_healthy:
            return reject(503, "engine is not running", "engine_dead", "server_error")
        if engine.num_active_requests >= max_pending_requests:
            return reject(
                429, "server is at capacity, retry later", "queue_full", "rate_limit_error",
                **{"Retry-After": "1"},
            )  # fmt: skip
        if not prompt_ids:
            return reject(400, "prompt is empty", "invalid", "invalid_request_error")
        if any(not 0 <= t < vocab_size for t in prompt_ids):
            return reject(
                400, f"token ids must be in [0, {vocab_size})", "invalid", "invalid_request_error"
            )
        if len(prompt_ids) + max_tokens > max_model_len:
            return reject(
                400,
                f"This model's maximum context length is {max_model_len} tokens. However, you "
                f"requested {len(prompt_ids) + max_tokens} tokens ({len(prompt_ids)} in the "
                f"prompt, {max_tokens} in the completion).",
                "too_long",
                "invalid_request_error",
            )
        return None

    def sampling_params(body: _SamplingFields, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop_token_ids=list(body.stop_token_ids or []),
            ignore_eos=body.ignore_eos,
        )

    async def text_stream(
        request_id: str, prompt_ids: list[int], params: SamplingParams, stops: list[str]
    ) -> AsyncGenerator[_Chunk, None]:
        detok = IncrementalDetokenizer(tokenizer)
        matcher = StopStringMatcher(stops)
        start = time.monotonic()
        async with aclosing(engine.generate(request_id, prompt_ids, params)) as tokens:
            async for out in tokens:
                delta = detok.add(out.token_id)
                if out.finished:
                    delta += detok.flush()
                text, stopped = matcher.push(delta, final=out.finished)
                if not (stopped or out.finished):
                    if text:
                        yield _Chunk(text, None, None)
                    continue
                usage = {
                    "prompt_tokens": out.num_prompt_tokens,
                    "completion_tokens": out.num_output_tokens,
                    "total_tokens": out.num_prompt_tokens + out.num_output_tokens,
                    "prompt_tokens_details": {"cached_tokens": out.num_cached_tokens},
                }
                reason = "stop" if stopped else out.finish_reason
                logger.info(
                    "%s finished: reason=%s prompt=%d cached=%d completion=%d e2e=%.3fs",
                    request_id, reason, out.num_prompt_tokens, out.num_cached_tokens,
                    out.num_output_tokens, time.monotonic() - start,
                )  # fmt: skip
                yield _Chunk(text, reason, usage)
                return

    async def collect(stream: AsyncGenerator[_Chunk, None], request: Request) -> tuple[str, _Chunk]:
        """Drain a stream for a non-streaming response, aborting if the client disconnects."""

        async def drain() -> tuple[str, _Chunk]:
            parts: list[str] = []
            last: _Chunk | None = None
            async with aclosing(stream):
                async for chunk in stream:
                    parts.append(chunk.text)
                    last = chunk
            assert last is not None
            return "".join(parts), last

        task = asyncio.ensure_future(drain())
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=1.0)
                if done:
                    return task.result()
                if await request.is_disconnected():
                    raise _ClientDisconnected
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    async def sse_stream(
        stream: AsyncGenerator[_Chunk, None],
        make_chunk: Any,
        base: dict[str, Any],
        include_usage: bool,
        preamble: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        usage = None
        try:
            if preamble is not None:
                yield sse(preamble)
            async with aclosing(stream):
                async for chunk in stream:
                    yield sse(make_chunk(chunk))
                    usage = chunk.usage or usage
            if include_usage and usage is not None:
                yield sse({**base, "choices": [], "usage": usage})
        except EngineDeadError as exc:
            yield sse({"error": {"message": str(exc), "type": "server_error", "code": 503}})
        yield "data: [DONE]\n\n"

    async def respond(
        request: Request,
        body: _SamplingFields,
        prompt_ids: list[int],
        max_tokens: int,
        kind: str,
    ) -> Response:
        if (problem := admit(prompt_ids, max_tokens)) is not None:
            return problem
        prefix = "chatcmpl" if kind == "chat" else "cmpl"
        request_id = f"{prefix}-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = text_stream(
            request_id, prompt_ids, sampling_params(body, max_tokens), body.stop_strings()
        )
        obj = "chat.completion" if kind == "chat" else "text_completion"
        base = {"id": request_id, "created": created, "model": model_name}

        if body.stream:
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            if kind == "chat":
                chunk_base = {**base, "object": "chat.completion.chunk"}

                def make_chunk(c: _Chunk) -> dict[str, Any]:
                    delta = {"content": c.text} if c.text else {}
                    choice = {"index": 0, "delta": delta, "finish_reason": c.finish_reason}
                    return {**chunk_base, "choices": [choice]}

                preamble = {
                    **chunk_base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                gen = sse_stream(stream, make_chunk, chunk_base, include_usage, preamble)
            else:
                chunk_base = {**base, "object": obj}

                def make_chunk(c: _Chunk) -> dict[str, Any]:
                    choice = {
                        "index": 0,
                        "text": c.text,
                        "logprobs": None,
                        "finish_reason": c.finish_reason,
                    }
                    return {**chunk_base, "choices": [choice]}

                gen = sse_stream(stream, make_chunk, chunk_base, include_usage)
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )

        try:
            text, last = await collect(stream, request)
        except _ClientDisconnected:
            return Response(status_code=499)
        except EngineDeadError as exc:
            return _error(503, str(exc), "server_error")
        if kind == "chat":
            choice: dict[str, Any] = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": last.finish_reason,
            }
        else:
            choice = {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": last.finish_reason,
            }
        return JSONResponse(
            {**base, "object": obj, "choices": [choice], "usage": last.usage},
            headers={"X-Request-Id": request_id},
        )

    # ------------------------------------------------------------------ API endpoints

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request) -> Response:
        if (msg := body.unsupported()) is not None:
            return _error(400, msg)
        if isinstance(body.prompt, str):
            prompt_ids = await asyncio.to_thread(tokenizer.encode, body.prompt)
        else:
            prompt_ids = list(body.prompt)
        return await respond(request, body, prompt_ids, body.max_tokens, "completion")

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
        if (msg := body.unsupported()) is not None:
            return _error(400, msg)
        messages = [{"role": m.role, "content": m.text()} for m in body.messages]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            return _error(400, f"could not apply chat template: {exc}")
        prompt_ids = await asyncio.to_thread(tokenizer.encode, prompt, add_special_tokens=False)
        max_tokens = body.max_completion_tokens or body.max_tokens
        if max_tokens is None:
            max_tokens = max(1, max_model_len - len(prompt_ids))
        return await respond(request, body, prompt_ids, max_tokens, "chat")

    return app


# ---------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="minillm-serve", description=__doc__)
    p.add_argument("--model", required=True, help="local path or Hugging Face Hub id")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument(
        "--attention-backend", default="auto", choices=["auto", "torch", "flash_attn", "flashinfer"]
    )
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    p.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--kv-cache-memory-gb", type=float, default=1.0)
    p.add_argument("--num-kv-blocks", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true", help="disable decode CUDA graphs")
    p.add_argument(
        "--no-async-scheduling",
        dest="async_scheduling",
        action="store_false",
        help="wait for each step's tokens before scheduling the next",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pending-requests", type=int, default=1024)
    p.add_argument(
        "--no-engine-process",
        dest="engine_process",
        action="store_false",
        help="run the engine on a thread in the API process instead of in its own process",
    )
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = EngineConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_gb=args.kv_cache_memory_gb,
        num_kv_blocks=args.num_kv_blocks,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        seed=args.seed,
    )
    metrics = EngineMetrics()
    engine: EngineClient
    if args.engine_process:
        # The model loads in the engine process; this one only needs the tokenizer.
        client = EngineCoreClient(
            EngineSpec(build_engine, {"config": config}), metrics=metrics, log_level=args.log_level
        )
        client.launch()
        engine = client
        tokenizer = load_tokenizer(config.tokenizer or resolve_model_path(config.model))
    else:
        llm = LLMEngine(config, metrics=metrics)
        engine = AsyncLLMEngine(llm)
        tokenizer = llm.tokenizer
    if tokenizer is None:
        raise SystemExit(f"no tokenizer found for {config.tokenizer or config.model}")
    app = build_app(
        engine,
        tokenizer,
        model_name=args.served_model_name or args.model,
        max_pending_requests=args.max_pending_requests,
        metrics=metrics,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
