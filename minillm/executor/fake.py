"""A deterministic stand-in for the GPU model runner.

Used by the test suite and CI so the scheduler, engine and server can be
exercised without a GPU or model weights. The next token depends only on the
sequence's own tokens, so outputs are identical however requests are batched,
chunked or preempted, which is exactly the property the real engine must have.
An optional cost model makes step latency depend on batch shape, which is what
the load tests need to show queueing and batching behaviour.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from minillm.scheduler import ScheduledSeq
from minillm.sequence import Sequence

CostModel = Callable[[list[ScheduledSeq]], float]


class LinearCost:
    """Step time = fixed launch cost + compute per new token + memory reads per context token.

    Roughly how a real step behaves: decode-heavy batches are dominated by the
    fixed cost and KV reads, prefill-heavy batches by per-token compute. A class,
    not a closure, so it pickles into an engine process.
    """

    def __init__(self, base_s: float, per_token_s: float, per_context_token_s: float) -> None:
        self.base_s = base_s
        self.per_token_s = per_token_s
        self.per_context_token_s = per_context_token_s

    def __call__(self, batch: list[ScheduledSeq]) -> float:
        new_tokens = sum(s.num_new_tokens for s in batch)
        context = sum(s.seq.num_computed_tokens + s.num_new_tokens for s in batch)
        return self.base_s + self.per_token_s * new_tokens + self.per_context_token_s * context


def linear_cost(
    base_s: float = 0.004, per_token_s: float = 2e-5, per_context_token_s: float = 1e-7
) -> CostModel:
    return LinearCost(base_s, per_token_s, per_context_token_s)


def default_next_token(seq: Sequence, vocab_size: int, reserved: int) -> int:
    # Skip the first `reserved` ids so special tokens (EOS etc.) never appear by accident.
    h = (seq.token_ids[-1] * 1_000_003 + len(seq.token_ids) * 7919) % (vocab_size - reserved)
    return reserved + h


class FakeExecutor:
    def __init__(
        self,
        *,
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        max_model_len: int = 2048,
        vocab_size: int = 32_000,
        eos_token_ids: list[int] | None = None,
        reserved_token_ids: int = 4,
        cost_model: CostModel | None = None,
        next_token: Callable[[Sequence], int] | None = None,
        text: str | None = None,
        crash_after_steps: int | None = None,
    ) -> None:
        self.num_kv_blocks = num_kv_blocks
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self.eos_token_ids = list(eos_token_ids or [])
        self.cost_model = cost_model
        if text is not None:  # "say" this text on repeat, for the byte tokenizer
            next_token = _TextTokens(text)
        self.crash_after_steps = crash_after_steps
        self._next_token = next_token or (
            lambda seq: default_next_token(seq, vocab_size, reserved_token_ids)
        )
        # One entry per step: (request_id, num_new_tokens, sample) for each batch item.
        self.history: list[list[tuple[str, int, bool]]] = []

    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        if self.crash_after_steps is not None and len(self.history) >= self.crash_after_steps:
            raise RuntimeError("CUDA error: an illegal memory access was encountered (simulated)")
        self.history.append([(s.seq.request_id, s.num_new_tokens, s.sample) for s in scheduled])
        if self.cost_model is not None:
            time.sleep(self.cost_model(scheduled))
        return [self._next_token(s.seq) for s in scheduled if s.sample]


class ByteTokenizer:
    """UTF-8 byte tokenizer to pair with FakeExecutor: id = byte + 3; 0/1/2 are pad/EOS/BOS."""

    EOS = 1
    eos_token_id: int | None = EOS
    bos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [b + 3 for b in text.encode()]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(t - 3 for t in token_ids if t >= 3).decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in conversation)
        return text + ("<assistant>" if add_generation_prompt else "")


class _TextTokens:
    """Next token = the next byte of ``text`` (ByteTokenizer ids). Picklable."""

    def __init__(self, text: str) -> None:
        self.ids = [b + 3 for b in text.encode()]

    def __call__(self, seq: Sequence) -> int:
        return self.ids[seq.num_output_tokens % len(self.ids)]


def build_fake_engine(
    config: dict[str, Any] | None = None,
    executor: dict[str, Any] | None = None,
    step_time_s: float = 0.0,
) -> Any:
    """An LLMEngine over FakeExecutor + ByteTokenizer, buildable inside a child process.

    Module-level so ``EngineSpec(build_fake_engine, {...})`` pickles. ``step_time_s``
    adds a fixed sleep per step.
    """
    from minillm.config import EngineConfig
    from minillm.engine import LLMEngine

    tokenizer = ByteTokenizer()
    kwargs: dict[str, Any] = dict(
        vocab_size=tokenizer.vocab_size, eos_token_ids=[ByteTokenizer.EOS]
    )
    kwargs.update(executor or {})
    if step_time_s:
        kwargs["cost_model"] = _FixedCost(step_time_s)
    return LLMEngine(
        EngineConfig(model="fake", **(config or {})),
        executor=FakeExecutor(**kwargs),
        tokenizer=tokenizer,
    )


class _FixedCost:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self, batch: list[ScheduledSeq]) -> float:
        return self.seconds
