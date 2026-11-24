"""Continuous-batching scheduler with chunked prefill and preemption.

Every engine step the scheduler builds one batch under a token budget
(``max_num_batched_tokens``):

1. Running sequences first, oldest first. A decoding sequence needs one token;
   a sequence part-way through a long prompt gets the next chunk that fits.
   If the KV cache is out of blocks, the newest running sequence is preempted:
   its blocks are freed and it goes back to the front of the waiting queue to
   be recomputed later (its finished blocks usually survive in the prefix cache,
   so recompute is often cheap).
2. Then, if nothing was preempted this step, waiting sequences are admitted
   FIFO while budget, ``max_num_seqs`` and free blocks allow. A prompt longer
   than the remaining budget is split across steps (chunked prefill), so one
   long prompt cannot stall every decoding sequence behind it.

Async scheduling: the engine schedules step n+1 before step n's sampled tokens
reach the CPU. So ``schedule`` commits each step's work up front (it advances
``num_computed_tokens`` and counts the token about to be sampled as pending),
and ``update`` later fills in the sampled tokens and checks stop conditions.
Length limits are known in advance, so a sequence whose pending token ends it
is not scheduled again. EOS and stop tokens are not: such a sequence gets one
more step whose token is discarded.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from minillm.block_manager import BlockManager
from minillm.sequence import FinishReason, Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int
    # First position this step computes. Fixed at scheduling time: the
    # sequence's own counters move on before the step's results come back.
    start: int = field(init=False)
    # True when this step computes the sequence's last uncomputed token, so the
    # model must sample a new token for it.
    sample: bool = field(init=False)

    def __post_init__(self) -> None:
        self.start = self.seq.num_computed_tokens
        self.sample = self.start + self.num_new_tokens == self.seq.num_tokens


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq]
    preempted: list[Sequence]

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


@dataclass
class TokenEvent:
    seq: Sequence
    token_id: int
    inter_token_latency: float | None


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        enable_chunked_prefill: bool = True,
        eos_token_ids: Iterable[int] = (),
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if not enable_chunked_prefill and max_num_batched_tokens < max_model_len:
            raise ValueError(
                "without chunked prefill, max_num_batched_tokens "
                f"({max_num_batched_tokens}) must be >= max_model_len ({max_model_len})"
            )
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.eos_token_ids = frozenset(eos_token_ids)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    def add(self, seq: Sequence) -> None:
        if seq.num_tokens >= self.max_model_len:
            raise ValueError(
                f"prompt has {seq.num_tokens} tokens; max_model_len is {self.max_model_len}"
            )
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, seq: Sequence) -> None:
        if seq.is_finished:
            return
        if seq.status is SequenceStatus.RUNNING:
            self.running.remove(seq)
        else:
            self.waiting.remove(seq)
        self._finish(seq, FinishReason.ABORT, time.monotonic())

    def schedule(self) -> SchedulerOutput:
        bm = self.block_manager
        budget = self.max_num_batched_tokens
        scheduled: list[ScheduledSeq] = []
        preempted: list[Sequence] = []

        i = 0
        finishing = 0  # running, but their pending token ends them
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            if self._finishes_on_pending(seq):
                finishing += 1
                i += 1
                continue
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            while not bm.try_allocate(seq, n):
                victim = self.running.pop()
                self._preempt(victim)
                preempted.append(victim)
                if victim is seq:
                    break
            else:
                scheduled.append(self._commit(seq, n))
                budget -= n
                i += 1

        if not preempted:
            now = time.monotonic()
            while self.waiting and budget > 0 and len(self.running) - finishing < self.max_num_seqs:
                seq = self.waiting[0]
                if seq.num_pending_tokens:
                    # Preempted while its last step was in flight: recompute it only
                    # once that token arrives. (Cannot happen while admission is
                    # skipped in rounds that preempt; the runner relies on it.)
                    break
                prefix = bm.find_cached_prefix(seq.token_ids)
                remaining = seq.num_tokens - len(prefix) * bm.block_size
                n = min(remaining, budget)
                if n < remaining and not self.enable_chunked_prefill:
                    break
                if not bm.can_allocate(seq, n, prefix, admit=True):
                    break
                self.waiting.popleft()
                bm.allocate(seq, n, prefix)
                seq.status = SequenceStatus.RUNNING
                if seq.first_scheduled_time is None:
                    seq.first_scheduled_time = now
                self.running.append(seq)
                scheduled.append(self._commit(seq, n))
                budget -= n

        return SchedulerOutput(scheduled, preempted)

    @staticmethod
    def _commit(seq: Sequence, num_new_tokens: int) -> ScheduledSeq:
        """Record the step's work on the sequence before it runs."""
        item = ScheduledSeq(seq, num_new_tokens)
        seq.num_computed_tokens += num_new_tokens
        if item.sample:
            seq.num_pending_tokens += 1
        return item

    def _finishes_on_pending(self, seq: Sequence) -> bool:
        """True if the token still in flight will end ``seq`` on a length limit."""
        return seq.num_pending_tokens > 0 and (
            seq.num_output_tokens + seq.num_pending_tokens >= seq.params.max_tokens
            or seq.num_tokens >= self.max_model_len
        )

    def update(self, output: SchedulerOutput, sampled: list[int]) -> list[TokenEvent]:
        """Apply a finished step: append its sampled tokens and check stop conditions."""
        num_samples = sum(1 for s in output.scheduled if s.sample)
        if len(sampled) != num_samples:
            raise ValueError(f"expected {num_samples} sampled tokens, got {len(sampled)}")
        now = time.monotonic()
        tokens = iter(sampled)
        events: list[TokenEvent] = []
        any_finished = False
        bm = self.block_manager
        caching, bs = bm.enable_prefix_caching, bm.block_size
        running, finished = SequenceStatus.RUNNING, SequenceStatus.FINISHED
        for s in output.scheduled:
            seq = s.seq
            if not s.sample:
                if seq.status is running and caching:
                    bm.cache_full_blocks(seq)
                continue
            token = next(tokens)
            seq.num_pending_tokens -= 1
            if seq.status is finished:
                # Aborted while the step ran, or (async) stopped on the previous
                # step's token: this one is the discarded extra.
                continue
            gap = seq.append_token(token, now)
            # Register newly full blocks (cheap check first: most steps fill none).
            if (
                caching
                and seq.status is running  # not preempted meanwhile
                and min(seq.num_computed_tokens, len(seq.token_ids))
                >= (len(seq.block_hashes) + 1) * bs
            ):
                bm.cache_full_blocks(seq)
            events.append(TokenEvent(seq, token, gap))
            reason = self._stop_reason(seq)
            if reason is not None:
                if seq.status is SequenceStatus.WAITING:  # preempted while the step ran
                    self.waiting.remove(seq)
                else:
                    any_finished = True
                self._finish(seq, reason, now)
        if any_finished:
            self.running = [s for s in self.running if not s.is_finished]
        return events

    def _stop_reason(self, seq: Sequence) -> FinishReason | None:
        params = seq.params
        token = seq.last_token_id
        if not params.ignore_eos and token in self.eos_token_ids:
            return FinishReason.STOP
        if token in params.stop_token_ids:
            return FinishReason.STOP
        if seq.num_output_tokens >= params.max_tokens or len(seq.token_ids) >= self.max_model_len:
            return FinishReason.LENGTH
        return None

    def _preempt(self, seq: Sequence) -> None:
        self.block_manager.free(seq)
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence, reason: FinishReason, now: float) -> None:
        self.block_manager.free(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = now
