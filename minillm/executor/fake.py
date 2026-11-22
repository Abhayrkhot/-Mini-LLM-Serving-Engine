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


def linear_cost(
    base_s: float = 0.004, per_token_s: float = 2e-5, per_context_token_s: float = 1e-7
) -> CostModel:
    """Step time = fixed launch cost + compute per new token + memory reads per context token.

    Roughly how a real step behaves: decode-heavy batches are dominated by the
    fixed cost and KV reads, prefill-heavy batches by per-token compute.
    """

    def cost(batch: list[ScheduledSeq]) -> float:
        new_tokens = sum(s.num_new_tokens for s in batch)
        context = sum(s.seq.num_computed_tokens + s.num_new_tokens for s in batch)
        return base_s + per_token_s * new_tokens + per_context_token_s * context

    return cost


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
    ) -> None:
        self.num_kv_blocks = num_kv_blocks
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self.eos_token_ids = list(eos_token_ids or [])
        self.cost_model = cost_model
        self._next_token = next_token or (
            lambda seq: default_next_token(seq, vocab_size, reserved_token_ids)
        )
        # One entry per step: (request_id, num_new_tokens, sample) for each batch item.
        self.history: list[list[tuple[str, int, bool]]] = []

    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        self.history.append([(s.seq.request_id, s.num_new_tokens, s.sample) for s in scheduled])
        if self.cost_model is not None:
            time.sleep(self.cost_model(scheduled))
        return [self._next_token(s.seq) for s in scheduled if s.sample]


class ByteTokenizer:
    """UTF-8 byte tokenizer to pair with FakeExecutor: id = byte + 3; 0/1/2 are pad/EOS/BOS."""

    eos_token_id = 1
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
