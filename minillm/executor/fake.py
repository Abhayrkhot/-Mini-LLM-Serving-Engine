"""A deterministic stand-in for the GPU model runner.

Used by the test suite and CI so the scheduler, engine and server can be
exercised without a GPU or model weights. The next token depends only on the
sequence's own tokens, so outputs are identical however requests are batched,
chunked or preempted, which is exactly the property the real engine must have.
An optional cost model makes step latency depend on batch shape, which is what
the load tests need to show queueing and batching behaviour.

Like the GPU runner, it supports async scheduling: a step's input may include
the token the previous step sampled, which the CPU has not seen yet. The fake
keeps those tokens to itself (its "device memory") and fails loudly if asked
for a token it cannot supply. Its cost model plays the GPU's role: steps queue
behind each other, and ``result`` waits until the step would have finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from minillm.executor.base import StepHandle
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
        context = sum(s.start + s.num_new_tokens for s in batch)
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
        model_kv_cache: bool = False,
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
        self._last_sampled: dict[Sequence, int] = {}  # by the latest step
        self._busy_until = 0.0
        # With model_kv_cache, the fake keeps the token "stored" in every physical
        # KV slot and predicts from those slots plus the step's inputs, as the
        # real model reads its context from the cache. Then any bookkeeping error
        # (a position counted as computed but never written, a stale reused
        # block, a bad prefix-cache hit, a draft not rolled back) changes outputs.
        self.model_kv_cache = model_kv_cache
        self._slots: dict[int, int] = {}

    def execute_async(self, scheduled: list[ScheduledSeq]) -> StepHandle:
        if self.crash_after_steps is not None and len(self.history) >= self.crash_after_steps:
            raise RuntimeError("CUDA error: an illegal memory access was encountered (simulated)")
        self.history.append([(s.seq.request_id, s.num_new_tokens, s.sample) for s in scheduled])
        tokens: list[int] = []
        sampled: dict[Sequence, int] = {}
        for s in scheduled:
            context = self._inputs(s)
            if self.model_kv_cache:
                context = self._through_cache(s, context)
            if s.sample:
                # The model's token after each input from the last real token on:
                # one position normally, one per draft plus one when speculating.
                for j in range(s.num_samples, 0, -1):
                    view = _SeqView(s.seq, context[: len(context) - j + 1])
                    token = self._next_token(view)  # type: ignore[arg-type]
                    tokens.append(token)
                sampled[s.seq] = token
        self._last_sampled = sampled
        ready = None
        if self.cost_model is not None:
            now = time.perf_counter()
            self._busy_until = max(now, self._busy_until) + self.cost_model(scheduled)
            ready = self._busy_until
        return _FakeStep(tokens, ready)

    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        return self.execute_async(scheduled).result()

    def _through_cache(self, item: ScheduledSeq, context: list[int]) -> list[int]:
        """The context as the cache holds it: earlier positions read from their slots,
        this step's inputs written to theirs."""
        seq, bs, start = item.seq, self.block_size, item.start
        slot = [seq.block_table[p // bs] * bs + p % bs for p in range(len(context))]
        try:
            cached = [self._slots[slot[p]] for p in range(start)]
        except KeyError as exc:
            raise RuntimeError(
                f"{seq.request_id}: a position before {start} was never written"
            ) from exc
        for p in range(start, len(context)):
            self._slots[slot[p]] = context[p]
        return [*cached, *context[start:]]

    def _inputs(self, item: ScheduledSeq) -> list[int]:
        """The sequence's tokens up to the end of this step's inputs."""
        seq = item.seq
        end = item.start + item.num_new_tokens
        known = seq.token_ids
        if item.drafts:
            return [*known, *item.drafts]
        if end <= len(known):
            return known[:end]
        if end == len(known) + 1 and seq in self._last_sampled:
            return [*known, self._last_sampled[seq]]
        raise RuntimeError(
            f"{seq.request_id}: input position {end - 1} is neither known to the CPU "
            "nor a token the previous step sampled"
        )


class _FakeStep:
    def __init__(self, tokens: list[int], ready_at: float | None) -> None:
        self.tokens = tokens
        self.ready_at = ready_at

    def result(self) -> list[int]:
        if self.ready_at is not None:
            time.sleep(max(0.0, self.ready_at - time.perf_counter()))
        return self.tokens


class _SeqView:
    """What a next-token function sees: the sequence with its inputs filled in."""

    def __init__(self, seq: Sequence, token_ids: list[int]) -> None:
        self.request_id = seq.request_id
        self.token_ids = token_ids
        self.num_prompt_tokens = seq.num_prompt_tokens

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]


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
