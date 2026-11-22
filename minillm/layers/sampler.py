"""Batched token sampling with per-request temperature, top-k and top-p."""

from __future__ import annotations

import torch

_GREEDY_EPS = 1e-5


@torch.inference_mode()
def sample(
    logits: torch.Tensor,  # [S, V], float32
    temperature: torch.Tensor,  # [S]
    top_p: torch.Tensor,  # [S]
    top_k: torch.Tensor,  # [S] int64, <= 0 disables
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperature < _GREEDY_EPS
    if bool(is_greedy.all()):
        return greedy_tokens

    vocab = logits.shape[-1]
    t = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    logits = logits / t[:, None]

    k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab))
    if bool((k < vocab).any()) or bool((top_p < 1.0).any()):
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        ranks = torch.arange(vocab, device=logits.device)
        drop = ranks[None, :] >= k[:, None]
        probs = sorted_logits.softmax(dim=-1)
        # Mass strictly before each token; the top token always survives.
        drop |= (probs.cumsum(dim=-1) - probs) > top_p[:, None]
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    # Exponential-race sampling: argmax(p / E) with E ~ Exp(1) draws from p
    # without the device-host sync torch.multinomial needs.
    noise = torch.empty_like(probs).exponential_(generator=generator).clamp_(min=1e-20)
    sampled = (probs / noise).argmax(dim=-1)
    return torch.where(is_greedy, greedy_tokens, sampled)
