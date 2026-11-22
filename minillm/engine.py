"""The synchronous engine: owns the scheduler, block manager and executor."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass

from minillm.block_manager import BlockManager
from minillm.config import EngineConfig
from minillm.executor.base import Executor
from minillm.metrics import EngineMetrics
from minillm.sampling import SamplingParams
from minillm.scheduler import Scheduler, SchedulerOutput, TokenEvent
from minillm.sequence import Sequence
from minillm.tokenizer import Tokenizer, load_tokenizer

logger = logging.getLogger(__name__)

_STATS_INTERVAL_S = 10.0


@dataclass(frozen=True)
class TokenOutput:
    """One generated token, copied out of the engine so other threads can read it safely."""

    request_id: str
    token_id: int
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str | None
    finish_reason: str | None
    num_cached_tokens: int
    num_preemptions: int
    queue_time: float
    time_to_first_token: float
    e2e_latency: float


class LLMEngine:
    def __init__(
        self,
        config: EngineConfig,
        executor: Executor | None = None,
        tokenizer: Tokenizer | None = None,
        metrics: EngineMetrics | None = None,
    ) -> None:
        self.config = config
        if executor is None:
            from minillm.executor.model_runner import ModelRunner

            runner = ModelRunner(config)
            executor = runner
            if tokenizer is None:
                tokenizer = load_tokenizer(config.tokenizer or runner.model_dir)
        self.executor = executor
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.max_model_len = executor.max_model_len
        eos = set(executor.eos_token_ids)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            eos.add(tokenizer.eos_token_id)
        self.eos_token_ids = sorted(eos)

        self.block_manager = BlockManager(
            executor.num_kv_blocks, executor.block_size, config.enable_prefix_caching
        )
        if executor.num_kv_blocks * executor.block_size < self.max_model_len:
            raise ValueError("KV cache cannot hold a single max_model_len sequence")
        self.scheduler = Scheduler(
            self.block_manager,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=self.max_model_len,
            enable_chunked_prefill=config.enable_chunked_prefill,
            eos_token_ids=self.eos_token_ids,
        )
        self._requests: dict[str, Sequence] = {}
        self._ids = itertools.count()
        self._last_stats = time.monotonic()
        self._window_prompt = 0
        self._window_generated = 0
        self._prefix_seen = (0, 0)

    # ------------------------------------------------------------------ requests

    def add_request(
        self,
        prompt_token_ids: list[int],
        params: SamplingParams,
        request_id: str | None = None,
        arrival_time: float | None = None,
    ) -> str:
        request_id = request_id if request_id is not None else f"req-{next(self._ids)}"
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id!r}")
        bad = [t for t in prompt_token_ids if not 0 <= t < self.executor.vocab_size]
        if bad:
            raise ValueError(f"token ids out of range [0, {self.executor.vocab_size}): {bad[:5]}")
        seq = Sequence(
            request_id,
            prompt_token_ids,
            params,
            arrival_time if arrival_time is not None else time.monotonic(),
        )
        self.scheduler.add(seq)
        self._requests[request_id] = seq
        return request_id

    def abort_request(self, request_id: str) -> bool:
        seq = self._requests.pop(request_id, None)
        if seq is None:
            return False
        self.scheduler.abort(seq)
        self._record_finished(seq)
        return True

    def has_unfinished_requests(self) -> bool:
        return bool(self._requests)

    @property
    def num_unfinished_requests(self) -> int:
        return len(self._requests)

    # ------------------------------------------------------------------ stepping

    def step(self) -> list[TokenOutput]:
        output = self.scheduler.schedule()
        if output.is_empty:
            self._update_gauges()
            return []
        prefill_tokens = sum(
            s.num_new_tokens
            for s in output.scheduled
            if s.seq.num_computed_tokens < s.seq.num_prompt_tokens
        )
        start = time.perf_counter()
        sampled = self.executor.execute(output.scheduled)
        elapsed = time.perf_counter() - start
        events = self.scheduler.update(output, sampled)
        self._record_step(output, events, elapsed, prefill_tokens)

        results = []
        for ev in events:
            seq = ev.seq
            if seq.is_finished:
                self._requests.pop(seq.request_id, None)
                self._record_finished(seq)
            results.append(
                TokenOutput(
                    request_id=seq.request_id,
                    token_id=ev.token_id,
                    finished=seq.is_finished,
                    finish_reason=seq.finish_reason.value if seq.finish_reason else None,
                    num_prompt_tokens=seq.num_prompt_tokens,
                    num_output_tokens=seq.num_output_tokens,
                    num_cached_tokens=seq.num_cached_tokens,
                )
            )
        return results

    def generate(
        self,
        prompts: SequenceABC[str | list[int]],
        params: SamplingParams | list[SamplingParams],
    ) -> list[RequestResult]:
        """Offline batch generation: submit everything, step until done."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("need one SamplingParams per prompt")
        seqs = []
        for prompt, p in zip(prompts, params, strict=True):
            ids = self._encode(prompt)
            rid = self.add_request(ids, p)
            seqs.append(self._requests[rid])
        while self.has_unfinished_requests():
            self.step()
        return [self._result(seq) for seq in seqs]

    # ------------------------------------------------------------------ helpers

    def _encode(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("string prompts need a tokenizer")
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def _result(self, seq: Sequence) -> RequestResult:
        out = seq.output_token_ids
        text = self.tokenizer.decode(out, skip_special_tokens=True) if self.tokenizer else None
        first_token = seq.first_token_time or seq.finish_time or seq.arrival_time
        return RequestResult(
            request_id=seq.request_id,
            prompt_token_ids=seq.token_ids[: seq.num_prompt_tokens],
            output_token_ids=out,
            text=text,
            finish_reason=seq.finish_reason.value if seq.finish_reason else None,
            num_cached_tokens=seq.num_cached_tokens,
            num_preemptions=seq.num_preemptions,
            queue_time=(seq.first_scheduled_time or seq.arrival_time) - seq.arrival_time,
            time_to_first_token=first_token - seq.arrival_time,
            e2e_latency=(seq.finish_time or time.monotonic()) - seq.arrival_time,
        )

    def _record_step(
        self,
        output: SchedulerOutput,
        events: list[TokenEvent],
        elapsed: float,
        prefill_tokens: int,
    ) -> None:
        self._window_prompt += prefill_tokens
        self._window_generated += len(events)
        m = self.metrics
        if m is not None:
            m.step_duration.observe(elapsed)
            m.step_tokens.observe(output.num_batched_tokens)
            m.generation_tokens.inc(len(events))
            if output.preempted:
                m.preemptions.inc(len(output.preempted))
            bm = self.block_manager
            queries, hits = bm.prefix_query_tokens, bm.prefix_hit_tokens
            m.prefix_cache_queries.inc(queries - self._prefix_seen[0])
            m.prefix_cache_hits.inc(hits - self._prefix_seen[1])
            self._prefix_seen = (queries, hits)
            for ev in events:
                if ev.inter_token_latency is None:
                    seq = ev.seq
                    assert seq.first_token_time is not None
                    m.time_to_first_token.observe(seq.first_token_time - seq.arrival_time)
                else:
                    m.inter_token_latency.observe(ev.inter_token_latency)
        self._update_gauges()
        self._maybe_log_stats()

    def _record_finished(self, seq: Sequence) -> None:
        m = self.metrics
        if m is None or seq.finish_reason is None:
            return
        m.requests_finished.labels(seq.finish_reason.value).inc()
        m.prompt_tokens.inc(seq.num_prompt_tokens)
        if seq.finish_time is not None:
            m.e2e_latency.observe(seq.finish_time - seq.arrival_time)
        if seq.first_scheduled_time is not None:
            m.queue_time.observe(seq.first_scheduled_time - seq.arrival_time)

    def _update_gauges(self) -> None:
        if self.metrics is not None:
            self.metrics.num_running.set(len(self.scheduler.running))
            self.metrics.num_waiting.set(len(self.scheduler.waiting))
            self.metrics.kv_cache_usage.set(self.block_manager.usage)

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        window = now - self._last_stats
        if window < _STATS_INTERVAL_S:
            return
        bm = self.block_manager
        hit_rate = bm.prefix_hit_tokens / bm.prefix_query_tokens if bm.prefix_query_tokens else 0.0
        logger.info(
            "prompt %.1f tok/s, generation %.1f tok/s | running %d, waiting %d | "
            "KV usage %.1f%% | prefix hit rate %.1f%% | preemptions %d",
            self._window_prompt / window,
            self._window_generated / window,
            len(self.scheduler.running),
            len(self.scheduler.waiting),
            100 * bm.usage,
            100 * hit_rate,
            self.scheduler.num_preemptions,
        )
        self._last_stats = now
        self._window_prompt = 0
        self._window_generated = 0
