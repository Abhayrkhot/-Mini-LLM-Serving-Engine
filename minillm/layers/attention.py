"""Paged attention over a block-structured KV cache.

All sequences in a step are packed into one flat token dimension with no
padding. Sequence ``b`` contributes ``query_lens[b]`` new tokens and attends
to ``seq_lens_k[b]`` keys: its cached context plus the new tokens. A decode
step is the case ``query_len == 1``; a prompt chunk on top of a cached prefix
is ``1 < query_len < seq_len_k``. One code path handles both, so prefill
chunks and decodes can share a batch.

Cache layout per layer: ``[num_blocks, block_size, num_kv_heads, head_dim]``,
the layout flash-attn's paged kernels expect.

Backends:
  * ``torch``: gathers each sequence's blocks and calls SDPA with an explicit
    mask. Runs on CPU, MPS and CUDA; it is the reference implementation the
    tests check against dense attention.
  * ``flash_attn``: flash-attn 2 varlen and kv-cache kernels with block tables
    (CUDA only, block_size must be a multiple of 256).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch
import torch.nn.functional as F
from torch import nn

from minillm.utils import cdiv


@dataclass(eq=False)
class AttentionMetadata:
    backend: str
    num_seqs: int
    max_seqlen_q: int
    max_seqlen_k: int
    query_lens: torch.Tensor  # int32 [B]
    seq_lens_k: torch.Tensor  # int32 [B]
    cu_seqlens_q: torch.Tensor  # int32 [B + 1]
    cu_seqlens_k: torch.Tensor  # int32 [B + 1]
    slot_mapping: torch.Tensor | None  # int64 [T]: flat cache slot for each new token
    block_tables: torch.Tensor | None  # int32 [B, max_blocks]; None = no cache (profiling run)
    decode_only: bool

    @cached_property
    def token_to_seq(self) -> torch.Tensor:
        """Batch index of each packed token."""
        seq_ids = torch.arange(self.num_seqs, device=self.query_lens.device)
        return torch.repeat_interleave(seq_ids, self.query_lens.long())

    @cached_property
    def token_pos(self) -> torch.Tensor:
        """Index of each packed token within its sequence's new tokens."""
        total = self.token_to_seq.numel()
        flat = torch.arange(total, device=self.query_lens.device)
        return flat - self.cu_seqlens_q[:-1].long()[self.token_to_seq]


def write_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_kv_heads, head_dim = k_cache.shape[-2:]
    k_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slot_mapping, v)


def _pad(x: torch.Tensor, meta: AttentionMetadata, length: int) -> torch.Tensor:
    out = x.new_zeros(meta.num_seqs, length, *x.shape[1:])
    out[meta.token_to_seq, meta.token_pos] = x
    return out


def torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    num_heads = q.shape[1]
    max_q, max_k = meta.max_seqlen_q, meta.max_seqlen_k
    qp = _pad(q, meta, max_q)  # [B, max_q, H, D]

    if meta.block_tables is None:
        kp, vp = _pad(k, meta, max_k), _pad(v, meta, max_k)
    else:
        assert k_cache is not None and v_cache is not None
        block_size = k_cache.shape[1]
        num_blocks = cdiv(max_k, block_size)
        tables = meta.block_tables[:, :num_blocks].long()
        shape = (meta.num_seqs, num_blocks * block_size, *k_cache.shape[2:])
        kp = k_cache[tables].reshape(shape)[:, :max_k]
        vp = v_cache[tables].reshape(shape)[:, :max_k]
        # Slots past a sequence's length belong to padding or to other requests'
        # stale data. Zero them: a masked score still multiplies V, and 0 * NaN
        # would leak one request's numerical blow-up into another's output.
        beyond = torch.arange(max_k, device=q.device)[None, :] >= meta.seq_lens_k[:, None]
        kp = kp.masked_fill(beyond[:, :, None, None], 0)
        vp = vp.masked_fill(beyond[:, :, None, None], 0)

    # Query j of sequence b sits at absolute position (seq_len_k - query_len + j)
    # and may attend to keys at positions <= that (bottom-right aligned causal mask).
    device = q.device
    first_pos = (meta.seq_lens_k - meta.query_lens).long()
    q_pos = first_pos[:, None] + torch.arange(max_q, device=device)[None, :]
    k_pos = torch.arange(max_k, device=device)
    mask = k_pos[None, None, :] <= q_pos[:, :, None]  # [B, max_q, max_k]

    out = F.scaled_dot_product_attention(
        qp.transpose(1, 2),
        kp.transpose(1, 2),
        vp.transpose(1, 2),
        attn_mask=mask[:, None],
        scale=scale,
        enable_gqa=kp.shape[2] != num_heads,  # no 7x copy of K/V for grouped-query models
    )  # [B, H, max_q, D]
    return out.transpose(1, 2)[meta.token_to_seq, meta.token_pos]


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    if meta.block_tables is None:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=meta.cu_seqlens_q,
            cu_seqlens_k=meta.cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q,
            max_seqlen_k=meta.max_seqlen_k,
            softmax_scale=scale,
            causal=True,
        )  # fmt: skip
    if meta.decode_only:
        out = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=meta.seq_lens_k,
            block_table=meta.block_tables,
            softmax_scale=scale,
            causal=True,
        )
        return out.squeeze(1)
    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=meta.cu_seqlens_q,
        cu_seqlens_k=meta.cu_seqlens_k,
        max_seqlen_q=meta.max_seqlen_q,
        max_seqlen_k=meta.max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=meta.block_tables,
    )  # fmt: skip


_BACKENDS = {"torch": torch_attention, "flash_attn": flash_attention}


class PagedAttention(nn.Module):
    """Writes this step's K/V into the cache, then attends over the cached context."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, scale: float) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        # Views into the engine's KV pool, assigned by the model runner.
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        if meta.block_tables is not None:
            if self.k_cache is None or self.v_cache is None or meta.slot_mapping is None:
                raise RuntimeError("KV cache not allocated")
            write_kv_cache(k, v, self.k_cache, self.v_cache, meta.slot_mapping)
        return _BACKENDS[meta.backend](q, k, v, self.k_cache, self.v_cache, meta, self.scale)
