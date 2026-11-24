"""GPTQ int32 packing for 4-bit codes (from Quantized-inference-lab, ``qlab/packing.py``).

* ``qweight`` is ``int32 [in_features / 8, out_features]``: each word holds eight
  consecutive *input* rows of one output column, row ``i`` of the pack at bit ``4 * i``.
  Callers pass codes as ``[in, out]``, the transpose of ``nn.Linear.weight``.
* ``qzeros`` is ``int32 [groups, out_features / 8]``, packed along the output dimension.

Values are unsigned 32-bit words stored in int32 (two's complement), as checkpoints hold them.
"""

from __future__ import annotations

import torch


def _to_int32(words: torch.Tensor) -> torch.Tensor:
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _from_int32(words: torch.Tensor) -> torch.Tensor:
    return words.to(torch.int64) & 0xFFFFFFFF


def pack_rows_int32(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack ``[K, N]`` codes along K into ``int32 [K * bits / 32, N]``."""
    per = 32 // bits
    k, n = codes.shape
    if k % per:
        raise ValueError(f"rows ({k}) must be a multiple of {per} for {bits}-bit packing")
    c = codes.to(torch.int64).reshape(k // per, per, n)
    shifts = (torch.arange(per, device=codes.device, dtype=torch.int64) * bits).view(1, per, 1)
    return _to_int32((c << shifts).sum(dim=1))


def unpack_rows_int32(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Inverse of :func:`pack_rows_int32`; returns uint8 ``[K, N]``."""
    per = 32 // bits
    w = _from_int32(packed).unsqueeze(1)
    shifts = (torch.arange(per, device=packed.device, dtype=torch.int64) * bits).view(1, per, 1)
    return ((w >> shifts) & (2**bits - 1)).reshape(-1, packed.shape[1]).to(torch.uint8)


def pack_cols_int32(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack ``[G, N]`` codes along N into ``int32 [G, N * bits / 32]``."""
    return pack_rows_int32(codes.t().contiguous(), bits).t().contiguous()


def unpack_cols_int32(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Inverse of :func:`pack_cols_int32`."""
    return unpack_rows_int32(packed.t().contiguous(), bits).t().contiguous()
