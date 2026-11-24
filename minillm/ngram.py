"""N-gram ("prompt lookup") drafts for speculative decoding.

Generated text often repeats spans of its own context: code being edited,
documents being summarised or quoted, structured output. So the next few
tokens can be guessed by finding the most recent earlier place where the
sequence's last ``n`` tokens occurred and proposing what followed them. The
model then checks every guess in one step; a right guess is a token for free.

Each sequence keeps an index from n-gram to the position right after its
most recent occurrence, extended as tokens arrive, so a lookup costs O(1)
instead of a scan of the whole context every step.
"""

from __future__ import annotations

from minillm.sequence import Sequence


class _Index:
    __slots__ = ("indexed", "where")

    def __init__(self, n_max: int) -> None:
        self.indexed = 0  # n-grams ending before this position are in the index
        self.where: list[dict[tuple[int, ...], int]] = [{} for _ in range(n_max + 1)]


class NgramProposer:
    def __init__(self, max_draft: int, n_max: int = 3, n_min: int = 1) -> None:
        if max_draft < 1 or not 1 <= n_min <= n_max:
            raise ValueError("need max_draft >= 1 and 1 <= n_min <= n_max")
        self.max_draft = max_draft
        self.n_max = n_max
        self.n_min = n_min
        self._index: dict[Sequence, _Index] = {}

    def propose(self, seq: Sequence, limit: int) -> list[int]:
        """Up to ``min(limit, max_draft)`` tokens likely to follow ``seq``'s tokens."""
        k = min(limit, self.max_draft)
        tokens = seq.token_ids
        length = len(tokens)
        if k <= 0 or length <= self.n_min:
            return []
        index = self._index.get(seq)
        if index is None:
            index = self._index[seq] = _Index(self.n_max)
        # Index every n-gram that ends at least one token before the end, so the
        # sequence's own suffix never matches itself.
        for end in range(index.indexed, length - 1):
            for n in range(self.n_min, self.n_max + 1):
                if end + 1 >= n:
                    index.where[n][tuple(tokens[end + 1 - n : end + 1])] = end + 1
        index.indexed = max(index.indexed, length - 1)
        for n in range(min(self.n_max, length - 1), self.n_min - 1, -1):
            start = index.where[n].get(tuple(tokens[length - n :]))
            if start is not None:
                return tokens[start : start + k]
        return []

    def forget(self, seq: Sequence) -> None:
        self._index.pop(seq, None)
