"""Per-request sampling parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

_GREEDY_EPS = 1e-5


@dataclass
class SamplingParams:
    """How to pick tokens for one request.

    ``temperature == 0`` means greedy decoding. ``top_k <= 0`` and ``top_p == 1``
    disable those filters.
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k == 0:
            self.top_k = -1
        if self.top_k < -1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {self.top_k}")

    @property
    def greedy(self) -> bool:
        return self.temperature < _GREEDY_EPS
