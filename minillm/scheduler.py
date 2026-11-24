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
from minillm.ngram import NgramProposer
from minillm.sequence import FinishReason, Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int
    # Speculative decoding: guessed next tokens, appended to this step's inputs.
    # The model's token at each of the last 1 + len(drafts) positions comes back.
    drafts: list[int] = field(default_factory=list)
    # First position this step computes. Fixed at scheduling time: the
    # sequence's own counters move on before the step's results come back.
    start: int = field(init=False)
    # True when this step computes the sequence's last uncomputed token, so the
    # model must sample a new token for it.
    sample: bool = field(init=False)

    def __post_init__(self) -> None:
        self.start = self.seq.num_computed_tokens
        computed = self.start + self.num_new_tokens - len(self.drafts)
        self.sample = computed == self.seq.num_tokens

    @property
    def num_samples(self) -> int:
        return 1 + len(self.drafts) if self.sample else 0


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
        proposer: NgramProposer | None = None,
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
        self.proposer = proposer
        self.spec_drafted = 0  # cumulative draft tokens proposed, and accepted
        self.spec_accepted = 0
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
            if n == 1 and self.proposer is not None and seq.params.greedy:
                drafts = self._draft(seq, budget)
                if drafts and bm.try_allocate(seq, 1 + len(drafts)):
                    scheduled.append(self._commit(seq, 1 + len(drafts), drafts))
                    budget -= 1 + len(drafts)
                    i += 1
                    continue
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

    def _draft(self, seq: Sequence, budget: int) -> list[int]:
        """Draft tokens for a decoding sequence, never past its length limits."""
        assert self.proposer is not None
        limit = min(
            budget - 1,
            seq.params.max_tokens - seq.num_output_tokens - 1,
            self.max_model_len - seq.num_tokens - 1,
        )
        return self.proposer.propose(seq, limit) if limit > 0 else []

    @staticmethod
    def _commit(
        seq: Sequence, num_new_tokens: int, drafts: list[int] | None = None
    ) -> ScheduledSeq:
        """Record the step's work on the sequence before it runs."""
        item = ScheduledSeq(seq, num_new_tokens, drafts or [])
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
        num_samples = sum(s.num_samples for s in output.scheduled)
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
            if s.drafts:
                targets = [next(tokens) for _ in range(1 + len(s.drafts))]
                seq.num_pending_tokens -= 1
                if seq.status is finished:
                    continue
                if self._apply_drafts(s, targets, now, events):
                    any_finished = True
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

    def _apply_drafts(
        self, s: ScheduledSeq, targets: list[int], now: float, events: list[TokenEvent]
    ) -> bool:
        """Accept the drafts the model agrees with, plus the model's next token.

        ``targets[j]`` is the model's greedy token after the inputs up to draft
        ``j``. Drafts are accepted while they equal it; the first disagreement
        is replaced by the model's token, and if all agree its token after the
        last draft comes free. Positions holding rejected drafts stop counting
        as computed: their K/V is simply overwritten later. Returns True if the
        sequence finished.
        """
        seq, drafts = s.seq, s.drafts
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == targets[accepted]:
            accepted += 1
        self.spec_drafted += len(drafts)
        self.spec_accepted += accepted
        seq.num_computed_tokens -= len(drafts) - accepted
        for token in [*drafts[:accepted], targets[accepted]]:
            gap = seq.append_token(token, now)
            events.append(TokenEvent(seq, token, gap))
            reason = self._stop_reason(seq)
            if reason is not None:
                self._finish(seq, reason, now)
                return True
        if self.block_manager.enable_prefix_caching:
            self.block_manager.cache_full_blocks(seq)
        return False

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
        if self.proposer is not None:
            self.proposer.forget(seq)
        self.block_manager.free(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = now
