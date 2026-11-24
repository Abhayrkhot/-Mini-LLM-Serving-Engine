"""Runs the real model: builds batch tensors, owns the KV cache, samples tokens."""

from __future__ import annotations

import importlib
import logging
import os
import time
from dataclasses import dataclass, field

import torch

from minillm.config import EngineConfig
from minillm.executor.base import DoneStep, StepHandle
from minillm.executor.cuda_graphs import DecodeGraphs
from minillm.layers.attention import AttentionMetadata, FlashInferPlanner
from minillm.layers.sampler import is_greedy, sample
from minillm.models.llama import ModelConfig
from minillm.models.loader import (
    build_model,
    load_weights,
    read_eos_token_ids,
    read_json,
    resolve_model_path,
)
from minillm.scheduler import ScheduledSeq
from minillm.sequence import Sequence
from minillm.utils import cdiv

logger = logging.getLogger(__name__)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        if name not in _DTYPES:
            raise ValueError(f"dtype must be auto or one of {sorted(_DTYPES)}")
        return _DTYPES[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def _importable(module: str) -> bool:
    # A real import, not find_spec: a flash-attn wheel built for a different
    # torch is "found" but fails to load its CUDA extension.
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def resolve_backend(name: str, device: torch.device) -> str:
    # Preference order for "auto": FlashInfer is the backend verified on H200 hardware.
    gpu_backends = {"flashinfer": "flashinfer", "flash_attn": "flash_attn"}
    if name == "auto":
        if device.type == "cuda":
            for backend, module in gpu_backends.items():
                if _importable(module):
                    return backend
        return "torch"
    if name in gpu_backends and (device.type != "cuda" or not _importable(gpu_backends[name])):
        raise ValueError(
            f"attention_backend={name} needs a CUDA device and a working {name} install"
        )
    return name


@dataclass
class _Batch:
    """One step's inputs: host lists, then the device tensors made from them."""

    num_seqs: int
    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=list)
    tables: list[list[int]] = field(default_factory=list)
    sample_rows: list[int] = field(default_factory=list)  # flat index of each sampler's last input
    sampling: list[Sequence] = field(default_factory=list)
    temps: list[float] = field(default_factory=list)
    top_ps: list[float] = field(default_factory=list)
    top_ks: list[int] = field(default_factory=list)
    # Input positions holding a token the previous step sampled (still on the
    # GPU), and that token's row in the previous step's output.
    pending_dst: list[int] = field(default_factory=list)
    pending_src: list[int] = field(default_factory=list)
    dev: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def decode_only(self) -> bool:
        return max(self.query_lens) == 1


class _CudaStep:
    """Sampled tokens on their way to a pinned host buffer."""

    def __init__(self, host: torch.Tensor, done: torch.cuda.Event) -> None:
        self.host = host
        self.done = done

    def result(self) -> list[int]:
        self.done.synchronize()
        tokens: list[int] = self.host.tolist()
        return tokens


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.model_dir = resolve_model_path(config.model)
        hf_config = read_json(os.path.join(self.model_dir, "config.json"))
        self.model_config = ModelConfig.from_hf(hf_config)
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.backend = resolve_backend(config.attention_backend, self.device)
        self.block_size = config.block_size or (256 if self.backend == "flash_attn" else 16)
        if self.backend == "flash_attn" and self.block_size % 256:
            raise ValueError("flash_attn paged kernels need block_size to be a multiple of 256")

        mc = self.model_config
        limit = mc.max_position_embeddings
        if config.max_model_len is not None and config.max_model_len > limit:
            raise ValueError(f"max_model_len {config.max_model_len} exceeds the model's {limit}")
        self.max_model_len = config.max_model_len or limit
        self.vocab_size = mc.vocab_size
        self.eos_token_ids = read_eos_token_ids(self.model_dir, hf_config)
        self._pin = self.device.type == "cuda"

        self.flashinfer: FlashInferPlanner | None = None
        if self.backend == "flashinfer":
            self.flashinfer = FlashInferPlanner(
                self.device, mc.num_heads, mc.num_kv_heads, mc.head_dim, self.block_size, self.dtype
            )

        torch.manual_seed(config.seed)
        start = time.perf_counter()
        self.model = build_model(mc, self.device, self.dtype)
        load_weights(self.model, self.model_dir)
        self.fused_kernels = self.backend == "flashinfer" and config.fused_kernels
        self.model.enable_fused_kernels(self.fused_kernels)
        weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        logger.info(
            "Loaded %s (%s) in %.1fs: %.2f GiB of weights, %s on %s, attention=%s%s",
            config.model,
            mc.model_type,
            time.perf_counter() - start,
            weight_bytes / 2**30,
            str(self.dtype).removeprefix("torch."),
            self.device,
            self.backend,
            ", fused kernels" if self.fused_kernels else "",
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.num_kv_blocks = self._init_kv_cache()

        # Tokens the latest step sampled, still on the device, and each sampling
        # sequence's row in them: the next step may need them as inputs.
        self._prev_tokens: torch.Tensor | None = None
        self._prev_rows: dict[Sequence, int] = {}
        # FlashInfer reads pages through its planned wrapper, not a block-table
        # tensor; this empty one only marks that the KV cache is in use.
        self._cache_marker = torch.empty(0, 0, dtype=torch.int32, device=self.device)

        # Decode CUDA graphs: CUDA + FlashInfer only (its decode kernel supports capture).
        self.graphs: DecodeGraphs | None = None
        if self.backend == "flashinfer" and not config.enforce_eager:
            self.graphs = DecodeGraphs(self, min(config.max_num_seqs, 512), self.scratch_block)

    # ------------------------------------------------------------------ KV cache

    def _block_bytes(self) -> int:
        mc = self.model_config
        elem = torch.empty((), dtype=self.dtype).element_size()
        return 2 * mc.num_layers * self.block_size * mc.num_kv_heads * mc.head_dim * elem

    def _init_kv_cache(self) -> int:
        cfg, mc = self.config, self.model_config
        block_bytes = self._block_bytes()
        if cfg.num_kv_blocks is not None:
            num_blocks = cfg.num_kv_blocks
        else:
            if self.device.type == "cuda":
                budget = self._profile_cuda_kv_budget()
            else:
                budget = int(cfg.kv_cache_memory_gb * 2**30)
            num_blocks = budget // block_bytes
            # No point holding more than every running sequence could ever fill.
            num_blocks = min(
                num_blocks, cfg.max_num_seqs * cdiv(self.max_model_len, self.block_size)
            )

        capacity = num_blocks * self.block_size
        if capacity < self.max_model_len:
            if cfg.max_model_len is not None or capacity < 1024:
                raise ValueError(
                    f"KV cache holds {capacity} tokens but max_model_len is {self.max_model_len}. "
                    "Give the cache more memory or lower max_model_len."
                )
            logger.warning(
                "Lowering max_model_len from %d to %d to fit the KV cache",
                self.max_model_len,
                capacity,
            )
            self.max_model_len = capacity

        # Zeroed, not empty: the torch backend reads masked padding slots, and a
        # masked weight of 0 times a NaN from uninitialised memory is still NaN.
        # One extra block past the block manager's range is scratch space that
        # padded rows of a CUDA-graph batch write into.
        self.scratch_block = num_blocks
        kv = torch.zeros(
            2,
            mc.num_layers,
            num_blocks + 1,
            self.block_size,
            mc.num_kv_heads,
            mc.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        for i, attn in enumerate(self.model.attention_layers):
            attn.k_cache = kv[0, i]
            attn.v_cache = kv[1, i]
        self.kv_cache = kv
        logger.info(
            "KV cache: %d blocks x %d tokens = %d tokens (%.2f GiB), max_model_len=%d",
            num_blocks,
            self.block_size,
            capacity,
            num_blocks * block_bytes / 2**30,
            self.max_model_len,
        )
        return num_blocks

    def _profile_cuda_kv_budget(self) -> int:
        """Bytes left for the KV cache after weights and a worst-case forward pass."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self._dummy_run()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        activation_peak = stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"]
        used = total - free
        budget = int(total * self.config.gpu_memory_utilization - used - activation_peak)
        torch.cuda.empty_cache()
        if budget <= 0:
            raise RuntimeError(
                "No GPU memory left for the KV cache; raise gpu_memory_utilization "
                "or lower max_num_batched_tokens"
            )
        return budget

    @torch.inference_mode()
    def _dummy_run(self) -> None:
        """A forward pass at the largest batch the scheduler can produce, without a KV cache."""
        cfg = self.config
        total = min(cfg.max_num_batched_tokens, cfg.max_num_seqs * self.max_model_len)
        num_seqs = min(cfg.max_num_seqs, total)
        lens = [total // num_seqs + (1 if i < total % num_seqs else 0) for i in range(num_seqs)]
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        cu_t = self._tensor(cu, torch.int32)
        lens_t = self._tensor(lens, torch.int32)
        meta = AttentionMetadata(
            backend=self.backend,
            num_seqs=num_seqs,
            max_seqlen_q=max(lens),
            max_seqlen_k=max(lens),
            query_lens=lens_t,
            seq_lens_k=lens_t,
            cu_seqlens_q=cu_t,
            cu_seqlens_k=cu_t,
            slot_mapping=None,
            block_tables=None,
            decode_only=False,
        )
        input_ids = torch.zeros(total, dtype=torch.long, device=self.device)
        positions = torch.cat([torch.arange(n, device=self.device) for n in lens])
        hidden = self.model(input_ids, positions, meta)
        last = self._tensor([c - 1 for c in cu[1:]], torch.long)
        logits = self.model.compute_logits(hidden[last]).float()
        ones = torch.ones(num_seqs, device=self.device)
        sample(logits, ones, ones, torch.full_like(ones, 50, dtype=torch.long), self.generator)

    # ------------------------------------------------------------------ execution

    def _tensor(self, data: list[int] | list[float], dtype: torch.dtype) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype, pin_memory=self._pin)
        return t.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def execute_async(self, scheduled: list[ScheduledSeq]) -> StepHandle:
        """Launch one step and return without waiting for the GPU.

        Nothing on this path synchronizes with the device: inputs go over
        through pinned memory, tokens the previous step sampled are gathered on
        the GPU, and this step's tokens come back through a pinned buffer whose
        copy a CUDA event tracks.
        """
        if not scheduled:
            return DoneStep([])
        batch = self._prepare(scheduled)
        hidden = self._run_model(batch)
        if not batch.sampling:
            self._prev_tokens, self._prev_rows = None, {}
            return DoneStep([])
        tokens = self._sample(hidden, batch)
        self._prev_tokens = tokens
        self._prev_rows = {seq: row for row, seq in enumerate(batch.sampling)}
        return self._to_host(tokens)

    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        return self.execute_async(scheduled).result()

    @torch.inference_mode()
    def compute_logits(self, scheduled: list[ScheduledSeq]) -> torch.Tensor | None:
        """Run the batch and return float32 logits for the sequences that sample (for tests)."""
        batch = self._prepare(scheduled)
        hidden = self._run_model(batch)
        if not batch.sampling:
            return None
        logits: torch.Tensor = self.model.compute_logits(hidden[batch.dev["rows"]]).float()
        return logits

    def _prepare(self, scheduled: list[ScheduledSeq]) -> _Batch:
        bs = self.block_size
        batch = _Batch(num_seqs=len(scheduled))
        for item in scheduled:
            seq = item.seq
            start = item.start
            end = start + item.num_new_tokens
            known = len(seq.token_ids)
            if end <= known:
                batch.input_ids.extend(seq.token_ids[start:end])
            else:
                # The last input is the token the previous step sampled for this
                # sequence; the CPU has not seen it. Leave a hole to fill on the GPU.
                row = self._prev_rows.get(seq)
                if end != known + 1 or row is None:
                    raise RuntimeError(
                        f"{seq.request_id}: input position {end - 1} is neither known to "
                        "the CPU nor a token the previous step sampled"
                    )
                batch.input_ids.extend(seq.token_ids[start:known])
                batch.pending_dst.append(len(batch.input_ids))
                batch.pending_src.append(row)
                batch.input_ids.append(0)
            batch.positions.extend(range(start, end))
            pos = start
            while pos < end:  # one contiguous slot range per block
                block_idx, offset = divmod(pos, bs)
                span = min(bs - offset, end - pos)
                base = seq.block_table[block_idx] * bs + offset
                batch.slots.extend(range(base, base + span))
                pos += span
            batch.query_lens.append(end - start)
            batch.seq_lens.append(end)
            batch.tables.append(seq.block_table)
            if item.sample:
                batch.sample_rows.append(len(batch.input_ids) - 1)
                batch.sampling.append(seq)
                params = seq.params
                batch.temps.append(params.temperature)
                batch.top_ps.append(params.top_p)
                batch.top_ks.append(params.top_k)
        return batch

    def _upload(
        self, parts: list[list[int]] | list[list[float]], dtype: torch.dtype
    ) -> list[torch.Tensor]:
        """Copy several host lists to the device in one transfer; return a view of each."""
        flat = [x for part in parts for x in part]
        dev = torch.tensor(flat, dtype=dtype, pin_memory=self._pin).to(
            self.device, non_blocking=True
        )
        views, offset = [], 0
        for part in parts:
            views.append(dev[offset : offset + len(part)])
            offset += len(part)
        return views

    def _upload_ints(self, batch: _Batch, *leading: list[int]) -> list[torch.Tensor]:
        """One copy for the step's int64 inputs: ``leading`` lists, then the
        sampling rows, top-k values and pending-token indices (kept in ``batch.dev``)."""
        names = ("rows", "top_ks", "pending_dst", "pending_src")
        extras = [batch.sample_rows, batch.top_ks, batch.pending_dst, batch.pending_src]
        views = self._upload([*leading, *extras], torch.long)
        batch.dev.update(zip(names, views[len(leading) :], strict=True))
        return views[: len(leading)]

    def gather_pending(self, input_ids: torch.Tensor, batch: _Batch) -> None:
        """Fill in input tokens the previous step sampled, straight from device memory."""
        if batch.pending_dst:
            assert self._prev_tokens is not None
            tokens = self._prev_tokens.index_select(0, batch.dev["pending_src"])
            input_ids.index_copy_(0, batch.dev["pending_dst"], tokens)

    def _run_model(self, batch: _Batch) -> torch.Tensor:
        if self.graphs is not None and batch.decode_only and self.graphs.can_run(batch.num_seqs):
            self._upload_ints(batch)
            return self.graphs.run(batch)
        return self._eager_forward(batch)

    def _eager_forward(self, batch: _Batch) -> torch.Tensor:
        ids, positions, slots = self._upload_ints(
            batch, batch.input_ids, batch.positions, batch.slots
        )
        self.gather_pending(ids, batch)
        cu_q, cu_k = [0], [0]
        for q_len, k_len in zip(batch.query_lens, batch.seq_lens, strict=True):
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
        parts = [batch.query_lens, batch.seq_lens, cu_q, cu_k]
        width = max(len(t) for t in batch.tables)
        if self.flashinfer is None:  # the torch and flash-attn kernels take block tables
            parts.append([x for t in batch.tables for x in (*t, *[0] * (width - len(t)))])
        views = self._upload(parts, torch.int32)
        meta = AttentionMetadata(
            backend=self.backend,
            num_seqs=batch.num_seqs,
            max_seqlen_q=max(batch.query_lens),
            max_seqlen_k=max(batch.seq_lens),
            query_lens=views[0],
            seq_lens_k=views[1],
            cu_seqlens_q=views[2],
            cu_seqlens_k=views[3],
            slot_mapping=slots,
            block_tables=views[4].view(batch.num_seqs, width)
            if self.flashinfer is None
            else self._cache_marker,
            decode_only=batch.decode_only,
        )
        if self.flashinfer is not None:
            meta.flashinfer = self.flashinfer.plan(batch.tables, batch.seq_lens, batch.query_lens)
        hidden: torch.Tensor = self.model(ids, positions, meta)
        return hidden

    def _sample(self, hidden: torch.Tensor, batch: _Batch) -> torch.Tensor:
        logits = self.model.compute_logits(hidden[batch.dev["rows"]]).float()
        if all(is_greedy(t) for t in batch.temps):
            tokens: torch.Tensor = logits.argmax(dim=-1)
            return tokens
        temps, top_ps = self._upload([batch.temps, batch.top_ps], torch.float32)
        needs_filter = any(0 < k < self.vocab_size for k in batch.top_ks) or any(
            p < 1.0 for p in batch.top_ps
        )
        return sample(
            logits,
            temps,
            top_ps,
            batch.dev["top_ks"],
            self.generator,
            all_greedy=False,
            needs_filter=needs_filter,
        )

    def _to_host(self, tokens: torch.Tensor) -> StepHandle:
        if self.device.type != "cuda":
            return DoneStep(tokens.tolist())
        host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=True)
        host.copy_(tokens, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return _CudaStep(host, done)
