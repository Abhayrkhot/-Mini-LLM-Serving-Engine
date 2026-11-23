"""Asyncio front ends for LLMEngine.

``EngineClient`` holds what every front end shares: one asyncio queue per
request, streaming, and abort-on-close. Two implementations:

* ``AsyncLLMEngine`` (here): the engine runs on a thread in this process.
  Simple and fast to start; used by the tests. Under heavy streaming load the
  HTTP event loop and the scheduler compete for the GIL.
* ``EngineCoreClient`` (minillm/engine_core.py): the engine runs in its own
  process, as in vLLM. The server's default.
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
    """The engine crashed or exited; the server must be restarted."""


StreamItem = TokenOutput | BaseException


class EngineClient:
    """Per-request token streams over some engine. Subclasses deliver the tokens."""

    def __init__(self) -> None:
        self._streams: dict[str, asyncio.Queue[StreamItem]] = {}
        self.error: BaseException | None = None

    # Implemented by subclasses ------------------------------------------------

    @property
    def max_model_len(self) -> int:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def is_healthy(self) -> bool:
        raise NotImplementedError

    @property
    def started(self) -> bool:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self, timeout: float = 30.0) -> None:
        raise NotImplementedError

    def _submit(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams, arrival: float
    ) -> None:
        raise NotImplementedError

    def abort(self, request_id: str) -> None:
        raise NotImplementedError

    # Shared --------------------------------------------------------------------

    @property
    def num_active_requests(self) -> int:
        return len(self._streams)

    async def generate(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[TokenOutput, None]:
        """Stream tokens for one request. Closing the iterator early aborts the request."""
        if self.error is not None:
            raise EngineDeadError("engine is not running") from self.error
        if not self.started:
            raise RuntimeError(f"{type(self).__name__}.start() has not been called")
        if request_id in self._streams:
            raise ValueError(f"duplicate request id {request_id!r}")
        stream: asyncio.Queue[StreamItem] = asyncio.Queue()
        self._streams[request_id] = stream
        self._submit(request_id, prompt_token_ids, params, time.monotonic())
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
                self.abort(request_id)

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
        if self.error is None:
            self.error = exc
        for stream in self._streams.values():
            stream.put_nowait(EngineDeadError(f"engine failed: {exc!r}"))


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


class AsyncLLMEngine(EngineClient):
    """Runs LLMEngine on a dedicated thread in this process.

    The engine thread owns all scheduler and model state; the event loop only
    sends commands (add, abort) through a queue and receives copied
    ``TokenOutput`` values, one ``call_soon_threadsafe`` callback per step.
    """

    def __init__(self, engine: LLMEngine) -> None:
        super().__init__()
        self.engine = engine
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def max_model_len(self) -> int:
        return self.engine.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.engine.executor.vocab_size

    @property
    def started(self) -> bool:
        return self._thread is not None

    @property
    def is_healthy(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self.error is None

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

    def _submit(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams, arrival: float
    ) -> None:
        self._commands.put(_Add(request_id, prompt_token_ids, params, arrival))

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
