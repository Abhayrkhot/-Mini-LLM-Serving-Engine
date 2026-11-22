"""Runs LLMEngine on a dedicated thread and streams tokens to asyncio callers.

The engine loop owns all scheduler and model state; the event loop only sends
commands (add, abort) through a queue and receives copied ``TokenOutput``
values. Each engine step's outputs are handed to the event loop in a single
``call_soon_threadsafe`` callback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

from minillm.engine import LLMEngine, TokenOutput
from minillm.sampling import SamplingParams

logger = logging.getLogger(__name__)


class EngineDeadError(RuntimeError):
    """The engine thread crashed; the server must be restarted."""


@dataclass(frozen=True)
class _Add:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float


@dataclass(frozen=True)
class _Abort:
    request_id: str


_Command = _Add | _Abort | None


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._streams: dict[str, asyncio.Queue[TokenOutput | BaseException]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.error: BaseException | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="minillm-engine", daemon=True)
        self._thread.start()

    async def stop(self, timeout: float = 30.0) -> None:
        if self._thread is None:
            return
        self._stopping.set()
        self._commands.put(None)
        await asyncio.to_thread(self._thread.join, timeout)
        self._thread = None

    @property
    def is_healthy(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self.error is None

    @property
    def num_active_requests(self) -> int:
        return len(self._streams)

    # ------------------------------------------------------------------ requests

    async def generate(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[TokenOutput, None]:
        """Stream tokens for one request. Closing the iterator early aborts the request."""
        if self.error is not None:
            raise EngineDeadError("engine is not running") from self.error
        if self._thread is None:
            raise RuntimeError("AsyncLLMEngine.start() has not been called")
        if request_id in self._streams:
            raise ValueError(f"duplicate request id {request_id!r}")
        stream: asyncio.Queue[TokenOutput | BaseException] = asyncio.Queue()
        self._streams[request_id] = stream
        self._commands.put(_Add(request_id, prompt_token_ids, params, time.monotonic()))
        finished = False
        try:
            while True:
                item = await stream.get()
                if isinstance(item, BaseException):
                    finished = True  # the engine already dropped it
                    raise item
                if item.finished:
                    finished = True
                yield item
                if finished:
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished:
                self._commands.put(_Abort(request_id))

    def abort(self, request_id: str) -> None:
        self._commands.put(_Abort(request_id))

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        engine = self.engine
        try:
            while not self._stopping.is_set():
                self._drain_commands(block=not engine.has_unfinished_requests())
                if self._stopping.is_set():
                    break
                if engine.has_unfinished_requests():
                    outputs = engine.step()
                    if outputs:
                        self._post(self._dispatch, outputs)
        except BaseException as exc:
            logger.exception("Engine loop crashed")
            self.error = exc
            self._post(self._fail_all, exc)

    def _drain_commands(self, block: bool) -> None:
        try:
            cmd = self._commands.get(timeout=0.5) if block else self._commands.get_nowait()
        except queue.Empty:
            return
        while True:
            self._apply(cmd)
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return

    def _apply(self, cmd: _Command) -> None:
        if isinstance(cmd, _Add):
            try:
                self.engine.add_request(
                    cmd.prompt_token_ids, cmd.params, cmd.request_id, cmd.arrival_time
                )
            except ValueError as exc:
                self._post(self._deliver, cmd.request_id, exc)
        elif isinstance(cmd, _Abort):
            self.engine.abort_request(cmd.request_id)

    def _post(self, fn: Callable[..., None], *args: object) -> None:
        assert self._loop is not None
        with contextlib.suppress(RuntimeError):  # event loop already closed during shutdown
            self._loop.call_soon_threadsafe(fn, *args)

    # ------------------------------------------------------------------ event-loop side

    def _dispatch(self, outputs: list[TokenOutput]) -> None:
        for out in outputs:
            stream = self._streams.get(out.request_id)
            if stream is not None:
                stream.put_nowait(out)

    def _deliver(self, request_id: str, item: BaseException) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            stream.put_nowait(item)

    def _fail_all(self, exc: BaseException) -> None:
        for stream in self._streams.values():
            stream.put_nowait(EngineDeadError(f"engine loop crashed: {exc!r}"))
