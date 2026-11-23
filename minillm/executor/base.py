"""Interface between the scheduler and whatever runs the model."""

from __future__ import annotations

from typing import Protocol

from minillm.scheduler import ScheduledSeq


class StepHandle(Protocol):
    """A launched step. ``result`` blocks until its sampled tokens are on the CPU."""

    def result(self) -> list[int]: ...


class DoneStep:
    """A step whose tokens are already known."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    def result(self) -> list[int]:
        return self.tokens


class Executor(Protocol):
    """Runs scheduled batches and returns the sampled tokens.

    ``execute_async`` launches a step and returns without waiting for it; its
    handle yields exactly one token for every ``ScheduledSeq`` whose ``sample``
    flag is set, in batch order. At most one step is in flight when the next
    one is launched, and the next step may need tokens the in-flight step
    sampled: an input position past ``len(seq.token_ids)`` is the token the
    previous step sampled for ``seq``, which the executor must supply itself
    (on a GPU, straight from device memory).
    """

    block_size: int
    num_kv_blocks: int
    max_model_len: int
    vocab_size: int
    eos_token_ids: list[int]

    def execute_async(self, scheduled: list[ScheduledSeq]) -> StepHandle: ...
