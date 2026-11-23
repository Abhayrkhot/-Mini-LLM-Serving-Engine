"""Request state tracked by the scheduler."""

from __future__ import annotations

import enum

from minillm.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, enum.Enum):
    STOP = "stop"  # EOS or a stop token
    LENGTH = "length"  # max_tokens or max_model_len
    ABORT = "abort"  # client went away or request was cancelled


class Sequence:
    """One generation request.

    The engine never distinguishes "prefill" from "decode". Each sequence has
    ``num_tokens`` tokens (prompt plus everything generated so far) of which the
    first ``num_computed_tokens`` have their K/V in the paged cache, or are being
    written by a step already sent to the GPU. A scheduling step computes some of
    the remaining tokens; when it catches up to ``num_tokens`` the model samples
    one new token. A fresh prompt, a chunk of a long prompt, a decode step and a
    recompute after preemption are all the same operation.

    With async scheduling the engine schedules step n+1 before step n's sampled
    tokens reach the CPU. Those tokens exist only on the GPU for a while:
    ``num_pending_tokens`` counts them (0 to 2), and ``num_tokens`` includes them,
    so the scheduler plans as if they had arrived. ``token_ids`` holds only the
    tokens the CPU has seen.
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        params: SamplingParams,
        arrival_time: float,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.request_id = request_id
        self.token_ids: list[int] = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.params = params
        self.status = SequenceStatus.WAITING
        self.finish_reason: FinishReason | None = None

        # Sampled by a step in flight, not yet seen by the CPU.
        self.num_pending_tokens = 0

        # Paged KV-cache state, owned by the BlockManager.
        self.num_computed_tokens = 0
        self.block_table: list[int] = []
        self.block_hashes: list[int] = []
        self.num_cached_tokens = 0  # tokens served from the prefix cache on admission

        self.num_preemptions = 0

        # Timestamps from time.monotonic().
        self.arrival_time = arrival_time
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        """Tokens the scheduler plans with: known tokens plus pending ones."""
        return len(self.token_ids) + self.num_pending_tokens

    @property
    def num_output_tokens(self) -> int:
        """Generated tokens the CPU has seen (pending ones excluded)."""
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int, now: float) -> float | None:
        """Record a sampled token. Returns the gap since the previous token, if any."""
        self.token_ids.append(token_id)
        gap = None if self.last_token_time is None else now - self.last_token_time
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        return gap

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.request_id!r}, status={self.status.value}, "
            f"tokens={self.num_tokens}, computed={self.num_computed_tokens}, "
            f"blocks={len(self.block_table)})"
        )
