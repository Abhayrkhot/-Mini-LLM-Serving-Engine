"""Paged attention against plain dense causal attention."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from minillm.layers.attention import AttentionMetadata, torch_attention, write_kv_cache
from minillm.utils import cdiv


def dense_reference(q, k, v, scale):
    """q: [Tq, H, D] for the last Tq positions; k, v: [Tk, Hkv, D]."""
    tq, h, _ = q.shape
    tk, hkv, _ = k.shape
    k = k.repeat_interleave(h // hkv, dim=1).float()
    v = v.repeat_interleave(h // hkv, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    q_pos = torch.arange(tk - tq, tk, device=q.device)[:, None]
    mask = torch.arange(tk, device=q.device)[None, :] <= q_pos
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@dataclass
class PagedBatch:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    meta: AttentionMetadata
    reference: torch.Tensor
    scale: float


def random_paged_batch(
    seed: int,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    block_size: int = 4,
    max_len: int = 40,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
) -> PagedBatch:
    """A batch mixing decodes, full prompts and chunks on cached context, in shuffled blocks.

    The cache already holds each sequence's context; the step's new K/V still
    has to be written (via ``write_kv_cache``) before attention runs.
    """
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)

    k_lens = [rng.randint(1, max_len) for _ in range(rng.randint(1, 5))]
    q_lens = [rng.choice([1, n, rng.randint(1, n)]) for n in k_lens]
    num_blocks = sum(cdiv(n, block_size) for n in k_lens) + 4
    k_cache = randn(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = randn(num_blocks, block_size, num_kv_heads, head_dim)
    free = list(range(num_blocks))
    rng.shuffle(free)  # physical blocks deliberately out of order
    scale = head_dim**-0.5

    qs, ks, vs, slots, tables, refs = [], [], [], [], [], []
    for seq_len, q_len in zip(k_lens, q_lens, strict=True):
        table = [free.pop() for _ in range(cdiv(seq_len, block_size))]
        tables.append(table)
        k_full = randn(seq_len, num_kv_heads, head_dim)
        v_full = randn(seq_len, num_kv_heads, head_dim)
        q_new = randn(q_len, num_heads, head_dim)
        for pos in range(seq_len):
            b, off = divmod(pos, block_size)
            if pos < seq_len - q_len:
                k_cache[table[b], off] = k_full[pos]
                v_cache[table[b], off] = v_full[pos]
            else:
                slots.append(table[b] * block_size + off)
        qs.append(q_new)
        ks.append(k_full[seq_len - q_len :])
        vs.append(v_full[seq_len - q_len :])
        refs.append(dense_reference(q_new, k_full, v_full, scale))

    def i32(xs: list[int]) -> torch.Tensor:
        return torch.tensor(xs, dtype=torch.int32, device=device)

    width = max(len(t) for t in tables)
    meta = AttentionMetadata(
        backend=backend,
        num_seqs=len(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        query_lens=i32(q_lens),
        seq_lens_k=i32(k_lens),
        cu_seqlens_q=i32([0, *torch.tensor(q_lens).cumsum(0).tolist()]),
        cu_seqlens_k=i32([0, *torch.tensor(k_lens).cumsum(0).tolist()]),
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        block_tables=i32([x for t in tables for x in t + [0] * (width - len(t))]).view(
            len(tables), width
        ),
        decode_only=max(q_lens) == 1,
    )
    return PagedBatch(
        torch.cat(qs), torch.cat(ks), torch.cat(vs), k_cache, v_cache, meta, torch.cat(refs), scale
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("heads", [(4, 4), (8, 2)])
def test_paged_matches_dense(seed: int, heads: tuple[int, int]) -> None:
    b = random_paged_batch(seed, num_heads=heads[0], num_kv_heads=heads[1])
    assert b.meta.slot_mapping is not None
    write_kv_cache(b.k, b.v, b.k_cache, b.v_cache, b.meta.slot_mapping)
    out = torch_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    torch.testing.assert_close(out, b.reference, rtol=1e-5, atol=1e-5)


def test_cacheless_path_matches_dense() -> None:
    """The profiling run attends over the step's own K/V with no cache at all."""
    torch.manual_seed(0)
    lens = [5, 1, 9]
    qs = [torch.randn(n, 4, 16) for n in lens]
    ks = [torch.randn(n, 2, 16) for n in lens]
    vs = [torch.randn(n, 2, 16) for n in lens]
    lens_t = torch.tensor(lens, dtype=torch.int32)
    cu = torch.tensor([0, 5, 6, 15], dtype=torch.int32)
    meta = AttentionMetadata("torch", 3, 9, 9, lens_t, lens_t, cu, cu, None, None, False)
    out = torch_attention(torch.cat(qs), torch.cat(ks), torch.cat(vs), None, None, meta, 0.25)
    ref = torch.cat([dense_reference(q, k, v, 0.25) for q, k, v in zip(qs, ks, vs, strict=True)])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
