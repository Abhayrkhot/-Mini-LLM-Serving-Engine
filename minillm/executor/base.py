"""Interface between the scheduler and whatever runs the model."""

from __future__ import annotations

from typing import Protocol

from minillm.scheduler import ScheduledSeq


class Executor(Protocol):
    """Runs one scheduled batch and returns the sampled tokens.

    ``execute`` returns exactly one token for every ``ScheduledSeq`` whose
    ``sample`` flag is set, in batch order.
    """

    block_size: int
    num_kv_blocks: int
    max_model_len: int
    vocab_size: int
    eos_token_ids: list[int]

    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]: ...
