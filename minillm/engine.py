"""The synchronous engine: owns the scheduler, block manager and executor."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import NamedTuple

from minillm.block_manager import BlockManager
from minillm.config import EngineConfig
from minillm.executor.base import Executor, StepHandle
from minillm.metrics import EngineMetrics, EngineStats, FinishedRequest
from minillm.ngram import NgramProposer
from minillm.sampling import SamplingParams
from minillm.scheduler import Scheduler, SchedulerOutput, TokenEvent
from minillm.sequence import Sequence
from minillm.tokenizer import Tokenizer, load_tokenizer

logger = logging.getLogger(__name__)

_STATS_INTERVAL_S = 10.0
# Async scheduling does not plan past a step predicted to take longer than this
# (see LLMEngine.step).
_LONG_STEP_S = 0.020


class TokenOutput(NamedTuple):
    """One generated token, copied out of the engine so other threads can read it safely.

    A NamedTuple: immutable, cheap to build for every token of every step, and it
    crosses the process boundary as a plain tuple.
    """

    request_id: str
    token_id: int
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int


@dataclass
class _Step:
    """A launched step whose tokens have not been collected yet."""

    output: SchedulerOutput
    handle: StepHandle
    launched: float
    prefill_tokens: int


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
        proposer = None
        if config.speculative_tokens > 0:
            proposer = NgramProposer(config.speculative_tokens, config.speculative_ngram_max)
        self.scheduler = Scheduler(
            self.block_manager,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=self.max_model_len,
            enable_chunked_prefill=config.enable_chunked_prefill,
            eos_token_ids=self.eos_token_ids,
            proposer=proposer,
        )
        # A speculative step's token count is known only when it finishes, so the
        # next step cannot be planned ahead of it.
        self.async_scheduling = config.async_scheduling and proposer is None
        if proposer is not None and config.async_scheduling:
            logger.info("Speculative decoding is on: async scheduling is off")
        self._inflight: _Step | None = None
        self._last_ready = 0.0
        # Seconds per batched token of recent steps with prompt chunks (an EMA).
        self._prefill_s_per_token: float | None = None
        self._requests: dict[str, Sequence] = {}
        self._ids = itertools.count()
        self._stats = EngineStats()
        self._last_log = time.monotonic()
        self._window_prompt = 0
        self._window_generated = 0
        self._prefix_seen = (0, 0)
        self._spec_seen = (0, 0)

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
        if not self._requests and self._inflight is not None:
            # Nothing left to wait for; collect the step now rather than leave it.
            step, self._inflight = self._inflight, None
            self._complete(step)
        self._flush_metrics()
        return True

    def has_unfinished_requests(self) -> bool:
        """True while requests are outstanding or a launched step is uncollected."""
        return bool(self._requests) or self._inflight is not None

    @property
    def num_unfinished_requests(self) -> int:
        return len(self._requests)

    # ------------------------------------------------------------------ stepping

    def step(self) -> list[TokenOutput]:
        """Run one engine iteration; return the tokens that became available.

        Synchronous: schedule a step, run it, return its tokens. Async: schedule
        and launch step n+1, then wait for step n's tokens and return those, so
        the CPU's work on one step overlaps the GPU's work on the other.

        Except after a long step (a big batch of prompt chunks): planning the
        next step while it runs would make every request that arrives meanwhile
        wait for that next step as well as for its own. So a step predicted to
        take over 20 ms is collected and returned first; the caller takes in new
        requests, and the next call plans a step that includes them. The GPU idles
        for one step's CPU work, a few percent of such a step. On Qwen2.5-7B with
        128 clients streaming, this lowered mean time to first token from about
        700 to 560 ms at the same throughput.
        """
        inflight = self._inflight
        if inflight is not None and self._predicted_s(inflight) > _LONG_STEP_S:
            self._inflight = None
            results = self._complete(inflight)
            self._flush_metrics()
            return results
        output = self.scheduler.schedule()
        launched = None
        if not output.is_empty:
            prefill_tokens = sum(
                s.num_new_tokens for s in output.scheduled if s.start < s.seq.num_prompt_tokens
            )
            start = time.perf_counter()
            handle = self.executor.execute_async(output.scheduled)
            launched = _Step(output, handle, start, prefill_tokens)

        previous, self._inflight = self._inflight, None
        results = self._complete(previous) if previous is not None else []
        if launched is not None:
            if self.async_scheduling:
                self._inflight = launched
            else:
                results.extend(self._complete(launched))
        self._flush_metrics()
        return results

    def _predicted_s(self, step: _Step) -> float:
        rate = self._prefill_s_per_token
        if not step.prefill_tokens or rate is None:
            return 0.0
        return rate * step.output.num_batched_tokens

    def _complete(self, step: _Step) -> list[TokenOutput]:
        """Wait for a launched step's tokens and apply them."""
        sampled = step.handle.result()
        ready = time.perf_counter()
        # Time on the device: from when the step could start (its launch, or the
        # previous step finishing, whichever was later) until its tokens arrived.
        elapsed = ready - max(step.launched, self._last_ready)
        self._last_ready = ready
        if step.prefill_tokens:
            rate = elapsed / step.output.num_batched_tokens
            prev = self._prefill_s_per_token
            self._prefill_s_per_token = rate if prev is None else 0.8 * prev + 0.2 * rate
        events = self.scheduler.update(step.output, sampled)
        self._record_step(step.output, events, elapsed, step.prefill_tokens)

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

    def take_stats(self) -> EngineStats:
        """Cut the stats snapshot accumulated since the last call (and start a new one)."""
        stats = self._stats
        stats.num_running = len(self.scheduler.running)
        stats.num_waiting = len(self.scheduler.waiting)
        stats.kv_cache_usage = self.block_manager.usage
        self._stats = EngineStats()
        return stats

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
        st = self._stats
        st.step_durations.append(elapsed)
        st.step_tokens.append(output.num_batched_tokens)
        st.generation_tokens += len(events)
        st.preemptions += len(output.preempted)
        bm = self.block_manager
        queries, hits = bm.prefix_query_tokens, bm.prefix_hit_tokens
        st.prefix_cache_queries += queries - self._prefix_seen[0]
        st.prefix_cache_hits += hits - self._prefix_seen[1]
        self._prefix_seen = (queries, hits)
        sched = self.scheduler
        st.spec_draft_tokens += sched.spec_drafted - self._spec_seen[0]
        st.spec_accepted_tokens += sched.spec_accepted - self._spec_seen[1]
        self._spec_seen = (sched.spec_drafted, sched.spec_accepted)
        for ev in events:
            if ev.inter_token_latency is None:
                seq = ev.seq
                assert seq.first_token_time is not None
                st.time_to_first_token.append(seq.first_token_time - seq.arrival_time)
            else:
                st.inter_token_latency.append(ev.inter_token_latency)
        self._maybe_log_stats()

    def _record_finished(self, seq: Sequence) -> None:
        if seq.finish_reason is None:
            return
        self._stats.finished.append(
            FinishedRequest(
                finish_reason=seq.finish_reason.value,
                prompt_tokens=seq.num_prompt_tokens,
                e2e_latency=None if seq.finish_time is None else seq.finish_time - seq.arrival_time,
                queue_time=None
                if seq.first_scheduled_time is None
                else seq.first_scheduled_time - seq.arrival_time,
            )
        )

    def _flush_metrics(self) -> None:
        """In-process metrics: apply the snapshot now. (An engine process instead
        leaves it for take_stats() and ships it to the API process.)"""
        if self.metrics is not None:
            self.metrics.record(self.take_stats())

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        window = now - self._last_log
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
        sched = self.scheduler
        if sched.spec_drafted:
            logger.info(
                "speculation: %.1f%% of %d drafted tokens accepted",
                100 * sched.spec_accepted / sched.spec_drafted,
                sched.spec_drafted,
            )
        self._last_log = now
        self._window_prompt = 0
        self._window_generated = 0
