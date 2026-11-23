"""CUDA graphs for decode steps.

A decode step for a small or medium model is dominated by launching ~30
kernels per layer from Python, not by the GPU work itself. Capturing the whole
forward pass as a CUDA graph turns those launches into one replay.

Graphs need fixed shapes and fixed memory addresses, so:
  * one graph is captured per batch size in ``sizes``, and a decode batch is
    padded up to the nearest one;
  * inputs (token ids, positions, KV slots) live in static buffers that are
    overwritten before each replay;
  * padded rows write their K/V to a scratch block the block manager never
    hands out, and attend over just that block;
  * each size gets its own FlashInfer decode wrapper with fixed index buffers;
    ``plan`` refills them for the real batch before the replay.

Only decode-only steps use graphs. Steps containing prompt chunks have
arbitrary shapes and run eagerly.
"""

from __future__ import annotations

import bisect
import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from minillm.layers.attention import AttentionMetadata
from minillm.utils import cdiv

if TYPE_CHECKING:
    from minillm.executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def capture_sizes(max_batch: int) -> list[int]:
    """Batch sizes to capture: dense at small sizes, every 16 above 32 (like vLLM)."""
    sizes = [1, 2, 4, 8, 16, 24, 32, *range(48, 513, 16)]
    return [s for s in sizes if s <= max_batch] or [1]


class DecodeGraphs:
    def __init__(self, runner: ModelRunner, max_batch: int, scratch_block: int) -> None:
        import flashinfer

        assert runner.flashinfer is not None, "decode graphs need the flashinfer backend"
        self.runner = runner
        self.sizes = capture_sizes(max_batch)
        self.scratch_block = scratch_block
        self.replays = 0
        mc = runner.model_config
        device = runner.device
        largest = self.sizes[-1]
        bs = runner.block_size
        self.max_pages = largest * cdiv(runner.max_model_len, bs)

        self.input_ids = torch.zeros(largest, dtype=torch.long, device=device)
        self.positions = torch.zeros(largest, dtype=torch.long, device=device)
        self.slots = torch.full((largest,), scratch_block * bs, dtype=torch.long, device=device)

        workspace = runner.flashinfer.workspace
        self.wrappers: dict[int, Any] = {}
        self.metas: dict[int, AttentionMetadata] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        for b in self.sizes:
            self.wrappers[b] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=torch.zeros(b + 1, dtype=torch.int32, device=device),
                paged_kv_indices_buffer=torch.zeros(
                    self.max_pages, dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buffer=torch.zeros(b, dtype=torch.int32, device=device),
            )
        self._plan_kwargs = dict(
            sm_scale=mc.head_dim**-0.5, q_data_type=runner.dtype, kv_data_type=runner.dtype
        )
        self._capture()

    def _plan(self, b: int, tables: list[list[int]], seq_lens: list[int]) -> None:
        """Plan size ``b``'s wrapper for the real sequences plus scratch padding rows."""
        bs = self.runner.block_size
        mc = self.runner.model_config
        indptr, indices, last = [0], [], []
        for table, seq_len in zip(tables, seq_lens, strict=True):
            pages = cdiv(seq_len, bs)
            indices.extend(table[:pages])
            indptr.append(indptr[-1] + pages)
            last.append(seq_len - (pages - 1) * bs)
        for _ in range(b - len(tables)):
            indices.append(self.scratch_block)
            indptr.append(indptr[-1] + 1)
            last.append(1)

        def i32(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.int32)

        self.wrappers[b].plan(
            i32(indptr),
            i32(indices),
            i32(last),
            mc.num_heads,
            mc.num_kv_heads,
            mc.head_dim,
            bs,
            **self._plan_kwargs,
        )

    @torch.inference_mode()
    def _capture(self) -> None:
        start = time.perf_counter()
        model = self.runner.model
        device = self.runner.device
        pool = torch.cuda.graph_pool_handle()
        one = torch.ones(1, dtype=torch.int32, device=device)
        for b in reversed(self.sizes):  # largest first, so smaller graphs reuse its memory
            meta = AttentionMetadata(
                backend="flashinfer",
                num_seqs=b,
                max_seqlen_q=1,
                max_seqlen_k=1,
                query_lens=one.expand(b),
                seq_lens_k=one.expand(b),
                cu_seqlens_q=torch.arange(b + 1, dtype=torch.int32, device=device),
                cu_seqlens_k=torch.arange(b + 1, dtype=torch.int32, device=device),
                slot_mapping=self.slots[:b],
                block_tables=torch.zeros(b, 1, dtype=torch.int32, device=device),
                decode_only=True,
                flashinfer=self.wrappers[b],
            )
            self._plan(b, [], [])  # all rows padding: attend over the scratch block
            model(self.input_ids[:b], self.positions[:b], meta)  # warm-up outside the graph
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = model(self.input_ids[:b], self.positions[:b], meta)
            self.graphs[b] = graph
            self.outputs[b] = out
            self.metas[b] = meta
        torch.cuda.synchronize()
        logger.info(
            "Captured decode CUDA graphs for %d batch sizes (1..%d) in %.1fs",
            len(self.sizes),
            self.sizes[-1],
            time.perf_counter() - start,
        )

    def can_run(self, num_seqs: int) -> bool:
        return num_seqs <= self.sizes[-1]

    def run(
        self,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        tables: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Replay the graph for this decode batch; returns hidden states for the real rows."""
        n = len(input_ids)
        b = self.sizes[bisect.bisect_left(self.sizes, n)]
        pad = b - n
        scratch_slot = self.scratch_block * self.runner.block_size

        def staged(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.long, pin_memory=True).to(
                self.input_ids.device, non_blocking=True
            )

        self.input_ids[:b].copy_(staged(input_ids + [0] * pad))
        self.positions[:b].copy_(staged(positions + [0] * pad))
        self.slots[:b].copy_(staged(slots + [scratch_slot] * pad))
        self._plan(b, tables, seq_lens)
        self.graphs[b].replay()
        self.replays += 1
        return self.outputs[b][:n]
