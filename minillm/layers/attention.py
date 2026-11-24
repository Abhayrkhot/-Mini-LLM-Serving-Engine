"""Paged attention over a block-structured KV cache.

All sequences in a step are packed into one flat token dimension with no
padding. Sequence ``b`` contributes ``query_lens[b]`` new tokens and attends
to ``seq_lens_k[b]`` keys: its cached context plus the new tokens. A decode
step is the case ``query_len == 1``; a prompt chunk on top of a cached prefix
is ``1 < query_len < seq_len_k``. One code path handles both, so prefill
chunks and decodes can share a batch.

Cache layout per layer: ``[num_blocks, block_size, num_kv_heads, head_dim]``,
the layout both flash-attn's and FlashInfer's paged kernels read ("NHD").

Backends:
  * ``torch``: gathers each sequence's blocks and calls SDPA with an explicit
    mask. Runs on CPU, MPS and CUDA; it is the reference implementation the
    tests check against dense attention.
  * ``flashinfer``: FlashInfer's paged prefill and decode kernels (CUDA). They
    read blocks in place through the block table; planned once per step.
  * ``flash_attn``: flash-attn 2 varlen and kv-cache kernels with block tables
    (CUDA only, block_size must be a multiple of 256).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

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
    flashinfer: Any = None  # a planned FlashInfer wrapper, shared by every layer this step

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


def paged_kv_layout(
    block_tables: list[list[int]],
    seq_lens: list[int],
    block_size: int,
    pad_rows: int = 0,
    pad_block: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """FlashInfer's CSR view of the batch's pages: (indptr, indices, last_page_len).

    ``pad_rows`` extra one-page sequences over ``pad_block`` fill a CUDA-graph batch.
    """
    indptr, indices, last = [0], [], []
    for table, seq_len in zip(block_tables, seq_lens, strict=True):
        pages = cdiv(seq_len, block_size)
        indices.extend(table[:pages])
        indptr.append(indptr[-1] + pages)
        last.append(seq_len - (pages - 1) * block_size)
    for _ in range(pad_rows):
        indices.append(pad_block)
        indptr.append(indptr[-1] + 1)
        last.append(1)
    return indptr, indices, last


def pinned_views(parts: list[list[int]], dtype: torch.dtype, pin: bool) -> list[torch.Tensor]:
    """Pack lists into one (pinned) host tensor; return a view per list."""
    flat = [x for part in parts for x in part]
    host = torch.tensor(flat, dtype=dtype, pin_memory=pin)
    views, offset = [], 0
    for part in parts:
        views.append(host[offset : offset + len(part)])
        offset += len(part)
    return views


def flashinfer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    if meta.block_tables is None:  # memory-profiling run before the cache exists
        return torch_attention(q, k, v, None, None, meta, scale)
    if meta.flashinfer is None:
        raise RuntimeError("flashinfer backend needs a planned wrapper in the metadata")
    out: torch.Tensor = meta.flashinfer.run(q, (k_cache, v_cache))
    return out


class FlashInferPlanner:
    """Owns FlashInfer's workspace and wrappers.

    FlashInfer splits work in a CPU-side ``plan`` step that depends only on the
    batch shape, so the engine plans once per step and every layer reuses it.
    Decode-only steps get the decode kernel (tensor cores, for grouped-query
    heads); anything with a prompt chunk gets the prefill kernel, whose causal
    mask is aligned to the end of each sequence like ours.

    Planning must not block on the GPU, or async scheduling stalls: every host
    tensor handed to ``plan`` is pinned (so its copy is asynchronous), and a
    CUDA-graph wrapper gets its page indices already on the device (from host
    memory it copies them synchronously).
    """

    def __init__(
        self,
        device: torch.device,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        workspace_mb: int = 256,
    ) -> None:
        import flashinfer

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.workspace = torch.zeros(workspace_mb * 2**20, dtype=torch.uint8, device=device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", use_tensor_cores=True
        )
        self.shape = (num_heads, num_kv_heads, head_dim, block_size)
        self.kwargs = dict(sm_scale=head_dim**-0.5, q_data_type=dtype, kv_data_type=dtype)
        self._planned: dict[int, torch.cuda.Event] = {}

    def run_plan(self, wrapper: Any, *args: Any, **kwargs: Any) -> None:
        """``wrapper.plan``, safe to call while earlier steps still run.

        ``plan`` writes its work schedule into a pinned host buffer and queues a
        copy to the GPU. Re-planning the same wrapper before that copy ran would
        overwrite the buffer under it, so wait for the previous one first. That
        copy is queued ahead of the step that used it, behind work the engine has
        already waited for, so in practice this never blocks.
        """
        done = self._planned.get(id(wrapper))
        if done is None:
            done = self._planned[id(wrapper)] = torch.cuda.Event()
        else:
            done.synchronize()
        wrapper.plan(*args, **kwargs)
        done.record()

    def plan_layout(
        self,
        indptr: list[int],
        indices: torch.Tensor,
        last_page_len: list[int],
        qo_indptr: list[int] | None = None,
    ) -> Any:
        """Plan a step from its page layout; ``indices`` may already be on the device.

        ``qo_indptr`` (query offsets) selects the prefill kernel; without it every
        sequence has one query token and the decode kernel runs.
        """
        if qo_indptr is None:
            indptr_h, last_h = pinned_views([indptr, last_page_len], torch.int32, pin=True)
            self.run_plan(self.decode, indptr_h, indices, last_h, *self.shape, **self.kwargs)
            return self.decode
        qo_h, indptr_h, last_h = pinned_views(
            [qo_indptr, indptr, last_page_len], torch.int32, pin=True
        )
        self.run_plan(
            self.prefill, qo_h, indptr_h, indices, last_h, *self.shape, causal=True, **self.kwargs
        )
        return self.prefill

    def plan_split(
        self,
        indptr: list[int],
        indices: torch.Tensor,
        last_page_len: list[int],
        qo_indptr: list[int],
        num_decode: int,
    ) -> Any:
        """Plan a step whose first ``num_decode`` sequences have one query token each
        and the rest prompt chunks: the decode kernel for the former, the prefill
        kernel for the latter.

        FlashInfer's prefill kernel tiles over query rows, so a decode row (one
        query token, a long context) wastes most of a tile: in a step mixing 192
        decodes with a 512-token chunk on Qwen2.5-7B, attention alone took 13 ms
        that way, more than the chunk's whole step on its own.
        """
        d = num_decode
        split = indptr[d]
        decode = self.plan_layout(indptr[: d + 1], indices[:split], last_page_len[:d])
        prefill = self.plan_layout(
            [p - split for p in indptr[d:]],
            indices[split:],
            last_page_len[d:],
            [q - d for q in qo_indptr[d:]],  # the decode rows hold one token each
        )
        return SplitAttention(decode, prefill, d)

    def plan(
        self, block_tables: list[list[int]], seq_lens: list[int], query_lens: list[int]
    ) -> Any:
        """Plan from Python block tables (tests; the engine uses ``plan_layout``)."""
        indptr, indices, last = paged_kv_layout(block_tables, seq_lens, self.block_size)
        (indices_h,) = pinned_views([indices], torch.int32, pin=True)
        qo_indptr = None
        if max(query_lens) > 1:
            qo_indptr = [0]
            for q_len in query_lens:
                qo_indptr.append(qo_indptr[-1] + q_len)
        return self.plan_layout(indptr, indices_h, last, qo_indptr)


class SplitAttention:
    """A planned mixed step: decode rows (first) and prompt chunks in separate kernels."""

    def __init__(self, decode: Any, prefill: Any, num_decode: int) -> None:
        self.decode = decode
        self.prefill = prefill
        self.num_decode = num_decode

    def run(self, q: torch.Tensor, kv_cache: Any) -> torch.Tensor:
        d = self.num_decode
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        self.decode.run(q[:d], kv_cache, out=out[:d])
        self.prefill.run(q[d:], kv_cache, out=out[d:])
        return out


_BACKENDS = {
    "torch": torch_attention,
    "flash_attn": flash_attention,
    "flashinfer": flashinfer_attention,
}


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
        # Views into the engine's KV pool, assigned by the model runner: K and V
        # separately, and both as one [2, blocks, block_size, heads, dim] view.
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: AttentionMetadata,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``kv``, if given, is k and v as one [tokens, 2, heads, dim] view (k already
        rotated); the cache write then takes one kernel instead of two."""
        if meta.block_tables is not None:
            if self.k_cache is None or self.v_cache is None or meta.slot_mapping is None:
                raise RuntimeError("KV cache not allocated")
            if kv is not None and self.kv_cache is not None:
                cache = self.kv_cache.view(2, -1, self.num_kv_heads, self.head_dim)
                cache.index_copy_(1, meta.slot_mapping, kv.transpose(0, 1))
            else:
                write_kv_cache(k, v, self.k_cache, self.v_cache, meta.slot_mapping)
        return _BACKENDS[meta.backend](q, k, v, self.k_cache, self.v_cache, meta, self.scale)
