"""FlashInfer's fused kernels for a decoder layer's small ops (CUDA only).

Written as plain PyTorch, RMSNorm is eight kernels (casts, square, mean, add,
rsqrt, two multiplies), rotary embeddings about ten per tensor, and SiLU-and-mul
two. At small batches those tiny kernels, not the matmuls, take most of a decode
step's GPU time. The fused path runs each as one kernel and folds the residual
add into the following norm, the layer structure vLLM and SGLang use:

    residual += x;  x = rmsnorm(residual)        # one fused_add_rmsnorm kernel

Numerics differ from the reference path in rounding only: the fused kernels
accumulate in float32 and round once, where the PyTorch ops round to bf16 after
the residual add and before the weight multiply.
"""

from __future__ import annotations

from typing import Any

import torch


class FusedOps:
    """Bound FlashInfer kernels; constructing one fails if FlashInfer is missing."""

    def __init__(self) -> None:
        import flashinfer

        self._rmsnorm: Any = flashinfer.norm.rmsnorm
        self._fused_add_rmsnorm: Any = flashinfer.norm.fused_add_rmsnorm
        self._rope: Any = flashinfer.rope.apply_rope_with_cos_sin_cache_inplace
        self._silu_and_mul: Any = flashinfer.activation.silu_and_mul

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        out: torch.Tensor = self._rmsnorm(x, weight, eps)
        return out

    def add_rmsnorm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """``residual += x``, then ``x = rmsnorm(residual)``, both in place; returns ``x``."""
        self._fused_add_rmsnorm(x, residual, weight, eps)
        return x

    def rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        """Rotate ``q`` [T, Hq * D] and ``k`` [T, Hk * D] in place (half-split convention).

        Strided views are fine, e.g. slices of the fused qkv projection's output.
        """
        self._rope(positions, q, k, head_dim, cos_sin_cache, True)

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """``silu(x[..., :d]) * x[..., d:]`` for x [..., 2d]."""
        out: torch.Tensor = self._silu_and_mul(x)
        return out
