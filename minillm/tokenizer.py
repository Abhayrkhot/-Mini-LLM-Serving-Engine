"""Tokenizer protocol, loading, and streaming detokenization."""

from __future__ import annotations

from typing import Any, Protocol, cast


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> Any: ...


def load_tokenizer(path: str) -> Tokenizer | None:
    """Load a Hugging Face tokenizer, or return None if ``path`` has none (token-id-only use)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        return None
    try:
        tok = cast(Tokenizer, AutoTokenizer.from_pretrained(path))
    except (OSError, ValueError):
        return None
    return tok


class IncrementalDetokenizer:
    """Turns a token stream into text deltas.

    Decoding token by token breaks multi-byte characters and drops leading
    spaces for some tokenizers. Instead we decode a short window
    (``prefix_offset:``) and emit only the text past the previous window
    (``prefix_offset:read_offset``), holding back output while the tail is an
    incomplete UTF-8 sequence (decoded as U+FFFD).
    """

    def __init__(self, tokenizer: Tokenizer, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)

    def add(self, token_id: int) -> str:
        self.token_ids.append(token_id)
        prefix = self._decode(self.token_ids[self.prefix_offset : self.read_offset])
        full = self._decode(self.token_ids[self.prefix_offset :])
        if len(full) > len(prefix) and not full.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.token_ids)
            return full[len(prefix) :]
        return ""

    def flush(self) -> str:
        """Emit whatever is held back (call once the stream has finished)."""
        prefix = self._decode(self.token_ids[self.prefix_offset : self.read_offset])
        full = self._decode(self.token_ids[self.prefix_offset :])
        self.prefix_offset = self.read_offset = len(self.token_ids)
        return full[len(prefix) :] if len(full) > len(prefix) else ""


class StopStringMatcher:
    """Applies OpenAI ``stop`` strings to streamed text.

    The stop string itself is never emitted. While streaming, the last
    ``len(longest_stop) - 1`` characters are held back because they could be
    the start of a stop string that the next token completes.
    """

    def __init__(self, stops: list[str]) -> None:
        self.stops = [s for s in stops if s]
        self.holdback = max((len(s) for s in self.stops), default=1) - 1
        self.text = ""
        self.emitted = 0

    def push(self, delta: str, final: bool = False) -> tuple[str, bool]:
        """Add ``delta``; return (text safe to emit now, whether a stop string matched)."""
        search_from = max(0, len(self.text) - self.holdback)
        self.text += delta
        if self.stops:
            hits = [i for i in (self.text.find(s, search_from) for s in self.stops) if i >= 0]
            if hits:
                self.text = self.text[: min(hits)]
                out = self.text[self.emitted :]
                self.emitted = len(self.text)
                return out, True
        safe = len(self.text) if final else max(self.emitted, len(self.text) - self.holdback)
        out = self.text[self.emitted : safe]
        self.emitted = safe
        return out, False
