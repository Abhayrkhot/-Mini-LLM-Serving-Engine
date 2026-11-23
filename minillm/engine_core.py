"""Run the engine in its own process.

The API server does a lot of per-token Python work: detokenizing, matching
stop strings, formatting and writing server-sent events. With the engine on a
thread in the same process, that work and the scheduler take turns holding the
GIL, so the GPU idles while the event loop formats JSON. Here the engine runs
in a child process (the "engine core", as in vLLM) and the two sides exchange
small messages over pipes:

    API process  --(add | abort | stop)-->  engine core
    API process  <--(ready | out | reject | error)--  engine core

One ``out`` message per engine step carries every token that step produced
(as tuples, cheap to pickle) plus an ``EngineStats`` snapshot, which the API
process applies to its Prometheus registry. Arrival times are taken in the API
process with ``time.monotonic()``, a system-wide clock, so time-to-first-token
includes the trip to the engine.

The child is started with ``spawn`` (CUDA cannot survive ``fork``) and exits
when its pipe closes, so it never outlives the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

from minillm.async_engine import EngineClient, EngineDeadError
from minillm.config import EngineConfig
from minillm.engine import LLMEngine, TokenOutput
from minillm.metrics import EngineMetrics
from minillm.sampling import SamplingParams

logger = logging.getLogger(__name__)

_IDLE_POLL_S = 0.5


@dataclass(frozen=True)
class EngineSpec:
    """How the child process builds its engine. Must be picklable: a module-level
    function and its keyword arguments."""

    factory: Callable[..., LLMEngine]
    kwargs: dict[str, Any] = field(default_factory=dict)


def build_engine(config: EngineConfig) -> LLMEngine:
    return LLMEngine(config)


def _pack(out: TokenOutput) -> tuple[Any, ...]:
    return (
        out.request_id,
        out.token_id,
        out.finished,
        out.finish_reason,
        out.num_prompt_tokens,
        out.num_output_tokens,
        out.num_cached_tokens,
    )


def run_engine_core(
    spec: EngineSpec, requests: Connection, outputs: Connection, log_level: str
) -> None:
    """Entry point of the engine process."""
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s [core]: %(message)s"
    )
    try:
        engine = spec.factory(**spec.kwargs)
    except BaseException as exc:
        logger.exception("Engine failed to start")
        outputs.send(("error", f"engine failed to start: {exc!r}"))
        return
    outputs.send(
        ("ready", {"max_model_len": engine.max_model_len, "vocab_size": engine.executor.vocab_size})
    )
    try:
        while True:
            if not engine.has_unfinished_requests() and not requests.poll(_IDLE_POLL_S):
                continue
            while requests.poll():
                msg = requests.recv()
                kind = msg[0]
                if kind == "add":
                    _, request_id, prompt_ids, params, arrival = msg
                    try:
                        engine.add_request(prompt_ids, params, request_id, arrival)
                    except ValueError as exc:
                        outputs.send(("reject", request_id, str(exc)))
                elif kind == "abort":
                    engine.abort_request(msg[1])
                elif kind == "stop":
                    return
            produced = engine.step() if engine.has_unfinished_requests() else []
            stats = engine.take_stats()
            if produced or not stats.is_empty:
                outputs.send(("out", [_pack(o) for o in produced], stats))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return  # the API process went away
    except BaseException as exc:
        logger.exception("Engine core crashed")
        with contextlib.suppress(OSError):
            outputs.send(("error", f"engine core crashed: {exc!r}"))


class EngineCoreClient(EngineClient):
    """The API-process side: owns the child process and both pipe ends."""

    def __init__(
        self,
        spec: EngineSpec,
        metrics: EngineMetrics | None = None,
        log_level: str = "INFO",
        startup_timeout_s: float = 1800.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.metrics = metrics
        self.startup_timeout_s = startup_timeout_s
        ctx = mp.get_context("spawn")
        child_requests, self._requests = ctx.Pipe(duplex=False)
        self._outputs, child_outputs = ctx.Pipe(duplex=False)
        self._child_ends = (child_requests, child_outputs)
        self._proc = ctx.Process(
            target=run_engine_core,
            args=(spec, child_requests, child_outputs, log_level),
            name="minillm-engine-core",
            daemon=True,
        )
        self._info: dict[str, int] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._stopping = False

    # ------------------------------------------------------------------ lifecycle

    def launch(self) -> None:
        """Start the engine process and wait until its model is loaded. Blocking."""
        if self._info:
            return
        self._proc.start()
        for end in self._child_ends:  # the child holds its own copies
            end.close()
        deadline = time.monotonic() + self.startup_timeout_s
        while not self._outputs.poll(1.0):
            if not self._proc.is_alive():
                raise RuntimeError(
                    f"engine process exited during startup (code {self._proc.exitcode})"
                )
            if time.monotonic() > deadline:
                self._proc.kill()
                raise TimeoutError("engine process did not become ready")
        msg = self._outputs.recv()
        if msg[0] != "ready":
            self._proc.join(10)
            raise RuntimeError(msg[1] if len(msg) > 1 else "engine process failed to start")
        self._info = msg[1]

    async def start(self) -> None:
        if self._loop is not None:
            return
        if not self._info:
            await asyncio.to_thread(self.launch)
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(self._outputs.fileno(), self._on_readable)
        self._watchdog = asyncio.create_task(self._watch())

    async def stop(self, timeout: float = 30.0) -> None:
        self._stopping = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self._loop is not None:
            self._loop.remove_reader(self._outputs.fileno())
        with contextlib.suppress(OSError):
            self._requests.send(("stop",))
        if self._proc.is_alive():
            await asyncio.to_thread(self._proc.join, timeout)
            if self._proc.is_alive():
                self._proc.kill()

    @property
    def max_model_len(self) -> int:
        return self._info["max_model_len"]

    @property
    def vocab_size(self) -> int:
        return self._info["vocab_size"]

    @property
    def started(self) -> bool:
        return self._loop is not None

    @property
    def is_healthy(self) -> bool:
        return self.started and self.error is None and self._proc.is_alive()

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    # ------------------------------------------------------------------ requests

    def _send(self, msg: tuple[Any, ...]) -> None:
        try:
            self._requests.send(msg)
        except OSError as exc:
            self._die(EngineDeadError(f"engine process unreachable: {exc!r}"))

    def _submit(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams, arrival: float
    ) -> None:
        self._send(("add", request_id, prompt_token_ids, params, arrival))

    def abort(self, request_id: str) -> None:
        if self.error is None:
            self._send(("abort", request_id))

    # ------------------------------------------------------------------ event-loop side

    def _on_readable(self) -> None:
        try:
            while self._outputs.poll():
                self._handle(self._outputs.recv())
        except (EOFError, OSError):
            self._die(EngineDeadError("engine process exited"))

    def _handle(self, msg: tuple[Any, ...]) -> None:
        kind = msg[0]
        if kind == "out":
            _, packed, stats = msg
            self._dispatch([TokenOutput(*t) for t in packed])
            if self.metrics is not None:
                self.metrics.record(stats)
        elif kind == "reject":
            self._deliver(msg[1], ValueError(msg[2]))
        elif kind == "error":
            self._die(EngineDeadError(msg[1]))

    def _die(self, exc: BaseException) -> None:
        if self.error is not None:
            return
        logger.error("%s", exc)
        if self._loop is not None:
            with contextlib.suppress(ValueError, OSError):
                self._loop.remove_reader(self._outputs.fileno())
        self._fail_all(exc)

    async def _watch(self) -> None:
        """Notice an engine process that died without saying so (e.g. OOM-killed)."""
        while not self._stopping:
            await asyncio.sleep(1.0)
            if not self._proc.is_alive() and self.error is None and not self._stopping:
                self._die(
                    EngineDeadError(
                        f"engine process exited unexpectedly (code {self._proc.exitcode})"
                    )
                )
