"""4-bit round-to-nearest quantization and the GPTQ packing (CPU; the kernel is in test_gpu)."""

from __future__ import annotations

import torch

from minillm.quant import dequantize, quantize_rtn
from minillm.quant.packing import (
    pack_cols_int32,
    pack_rows_int32,
    unpack_cols_int32,
    unpack_rows_int32,
)


def test_round_to_nearest_error_is_at_most_half_a_step() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 512) * torch.logspace(-2, 1, 512)  # columns of very different scales
    codes, scale, zero = quantize_rtn(w, group_size=128)
    assert codes.dtype == torch.uint8 and int(codes.max()) <= 15
    assert scale.shape == zero.shape == (64, 4)
    err = (dequantize(codes, scale, zero) - w).abs().reshape(64, 4, 128)
    assert bool((err <= scale.unsqueeze(-1) / 2 + 1e-6).all())


def test_zero_is_exact_and_constant_groups_survive() -> None:
    w = torch.zeros(8, 128)
    w[0] = 0.5  # all-positive group: zero point at code 0
    w[1, :64] = -1.0  # mixed with exact zeros
    codes, scale, zero = quantize_rtn(w, group_size=128)
    back = dequantize(codes, scale, zero)
    assert bool((back[w == 0] == 0).all())
    torch.testing.assert_close(back, w, atol=1e-6, rtol=0)


def test_gptq_packing_round_trips() -> None:
    gen = torch.Generator().manual_seed(1)
    codes = torch.randint(0, 16, (256, 96), generator=gen, dtype=torch.uint8)
    packed = pack_rows_int32(codes)
    assert packed.shape == (32, 96) and packed.dtype == torch.int32
    assert torch.equal(unpack_rows_int32(packed), codes)
    zeros = torch.randint(0, 16, (4, 96), generator=gen, dtype=torch.uint8)
    packed_z = pack_cols_int32(zeros)
    assert packed_z.shape == (4, 12)
    assert torch.equal(unpack_cols_int32(packed_z), zeros)
    # Row 8i + j of a column sits at bits 4j..4j+3 of word i.
    assert int(packed[0, 0]) & 0xF == int(codes[0, 0])
    assert (int(packed[0, 0]) >> 4) & 0xF == int(codes[1, 0])
