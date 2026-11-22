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
    # True when this step computes the sequence's last uncomputed token, so the
    # model must sample a new token for it. Fixed at scheduling time.
    sample: bool = field(init=False)

    def __post_init__(self) -> None:
        self.sample = self.seq.num_computed_tokens + self.num_new_tokens == self.seq.num_tokens


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
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            while not bm.can_allocate(seq, n):
                victim = self.running.pop()
                self._preempt(victim)
                preempted.append(victim)
                if victim is seq:
                    break
            else:
                bm.allocate(seq, n)
                scheduled.append(ScheduledSeq(seq, n))
                budget -= n
                i += 1

        if not preempted:
            now = time.monotonic()
            while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
                seq = self.waiting[0]
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
                scheduled.append(ScheduledSeq(seq, n))
                budget -= n

        return SchedulerOutput(scheduled, preempted)

    def update(self, output: SchedulerOutput, sampled: list[int]) -> list[TokenEvent]:
        """Apply a finished step: advance computed counts and append sampled tokens."""
        num_samples = sum(1 for s in output.scheduled if s.sample)
        if len(sampled) != num_samples:
            raise ValueError(f"expected {num_samples} sampled tokens, got {len(sampled)}")
        now = time.monotonic()
        tokens = iter(sampled)
        events: list[TokenEvent] = []
        any_finished = False
        for s in output.scheduled:
            seq = s.seq
            token = next(tokens) if s.sample else None
            if seq.is_finished:  # aborted while the step was running
                continue
            seq.num_computed_tokens += s.num_new_tokens
            self.block_manager.cache_full_blocks(seq)
            if token is None:
                continue
            gap = seq.append_token(token, now)
            events.append(TokenEvent(seq, token, gap))
            reason = self._stop_reason(seq)
            if reason is not None:
                self._finish(seq, reason, now)
                any_finished = True
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
        if seq.num_output_tokens >= params.max_tokens or seq.num_tokens >= self.max_model_len:
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
