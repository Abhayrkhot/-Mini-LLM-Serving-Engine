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
from minillm.executor.block_tables import DeviceBlockTables
from minillm.executor.cuda_graphs import DecodeGraphs
from minillm.layers.attention import AttentionMetadata, FlashInferPlanner, pinned_views
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
    """One step's inputs as host lists, O(sequences) apart from prompt tokens.

    Per-token positions and KV slots, and FlashInfer's page lists, are computed
    on the device from these (see block_tables.py); ``dev`` holds the results.
    """

    num_seqs: int
    num_decode: int = 0  # sequences with one new token; they come first
    input_ids: list[int] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)  # first position each sequence computes
    query_lens: list[int] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)  # block-table row of each sequence
    tables: list[list[int]] = field(default_factory=list)  # the block manager's tables
    slots: list[int] = field(default_factory=list)  # KV slot per sequence (decode steps)
    sample_rows: list[int] = field(default_factory=list)  # flat index of each sampler's last input
    sampling: list[Sequence] = field(default_factory=list)
    temps: list[float] = field(default_factory=list)
    top_ps: list[float] = field(default_factory=list)
    top_ks: list[int] = field(default_factory=list)
    # Input positions holding a token the previous step sampled (still on the
    # GPU), and that token's row in the previous step's output.
    pending_dst: list[int] = field(default_factory=list)
    pending_src: list[int] = field(default_factory=list)
    # Block-table entries the device does not have yet.
    table_rows: list[int] = field(default_factory=list)
    table_cols: list[int] = field(default_factory=list)
    table_vals: list[int] = field(default_factory=list)
    # FlashInfer's page layout, possibly padded to a CUDA-graph batch size.
    indptr: list[int] = field(default_factory=list)
    last_page_len: list[int] = field(default_factory=list)
    dev: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def decode_only(self) -> bool:
        return len(self.input_ids) == self.num_seqs


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
        # FlashInfer reads pages through its planned wrapper, not block tables or
        # length tensors; these empty ones fill the metadata fields it ignores.
        self._cache_marker = torch.empty(0, 0, dtype=torch.int32, device=self.device)
        self._unused = torch.empty(0, dtype=torch.int32, device=self.device)
        self.block_tables = DeviceBlockTables(
            config.max_num_seqs,
            cdiv(self.max_model_len, self.block_size),
            self.scratch_block,
            self.device,
        )

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
            attn.kv_cache = kv[:, i]
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
        logits: torch.Tensor = self.model.compute_logits(hidden[batch.dev["sample_rows"]]).float()
        return logits

    def _prepare(self, scheduled: list[ScheduledSeq]) -> _Batch:
        bs = self.block_size
        # Sequences with one new token first, prompt chunks after: in a mixed step
        # the two groups run through different attention kernels (plan_split).
        # Sampled tokens still come back in scheduled order.
        decodes = [i for i, item in enumerate(scheduled) if item.num_new_tokens == 1]
        order = decodes
        if len(decodes) < len(scheduled):
            order = decodes + [i for i, item in enumerate(scheduled) if item.num_new_tokens > 1]
        batch = _Batch(num_seqs=len(scheduled), num_decode=len(decodes))
        tables = self.block_tables
        tables.begin_step()
        ids = batch.input_ids
        sample_row: dict[int, int] = {}
        for i in order:
            item = scheduled[i]
            seq = item.seq
            start = item.start
            end = start + item.num_new_tokens
            known = len(seq.token_ids)
            if end <= known:
                ids.extend(seq.token_ids[start:end])
            else:
                # The last input is the token the previous step sampled for this
                # sequence; the CPU has not seen it. Leave a hole to fill on the GPU.
                row = self._prev_rows.get(seq)
                if end != known + 1 or row is None:
                    raise RuntimeError(
                        f"{seq.request_id}: input position {end - 1} is neither known to "
                        "the CPU nor a token the previous step sampled"
                    )
                ids.extend(seq.token_ids[start:known])
                batch.pending_dst.append(len(ids))
                batch.pending_src.append(row)
                ids.append(0)
            batch.starts.append(start)
            batch.query_lens.append(end - start)
            batch.seq_lens.append(end)
            table = seq.block_table
            batch.tables.append(table)
            batch.slots.append(table[start // bs] * bs + start % bs)  # used if decode-only
            batch.rows.append(tables.row(seq, batch.table_rows, batch.table_cols, batch.table_vals))
            if item.sample:
                sample_row[i] = len(ids) - 1
        for i, item in enumerate(scheduled):
            if item.sample:
                batch.sample_rows.append(sample_row[i])
                batch.sampling.append(item.seq)
                params = item.seq.params
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

    # Up to this many pages, FlashInfer's page list is built in Python; beyond it,
    # on the device from the block tables. Small lists cost less to build here than
    # the extra kernels cost the GPU; large ones (a big decode batch with long
    # contexts: tens of thousands of entries) the other way round.
    HOST_PAGES_MAX = 4096

    def _upload_inputs(self, batch: _Batch, padded: int, graph: bool = False) -> None:
        """Send the step's inputs to the device and lay out FlashInfer's pages.

        Decode steps carry one token per sequence, so positions and KV slots are
        computed here. Steps with prompt chunks can carry thousands of tokens:
        their positions and slots are computed on the device from the block
        tables. ``padded`` >= num_seqs adds scratch-page rows for a CUDA-graph
        batch, whose graph copies the inputs into its own buffers (``graph``).
        """
        bs, n = self.block_size, batch.num_seqs
        pad = padded - n
        pages = [(length + bs - 1) // bs for length in batch.seq_lens]
        indptr = [0]
        for p in pages:
            indptr.append(indptr[-1] + p)
        for _ in range(pad):
            indptr.append(indptr[-1] + 1)
        batch.indptr = indptr
        batch.last_page_len = [
            length - (p - 1) * bs for length, p in zip(batch.seq_lens, pages, strict=True)
        ] + [1] * pad
        decode = batch.decode_only
        device_pages = self.flashinfer is not None and indptr[-1] > self.HOST_PAGES_MAX

        parts: dict[str, list[int]] = {}
        if not graph:
            parts["ids"] = batch.input_ids
            if decode:
                parts["positions"] = batch.starts
                parts["slots"] = batch.slots
        if not decode:
            qstart = [0] * n
            for i in range(1, n):
                qstart[i] = qstart[i - 1] + batch.query_lens[i - 1]
            parts["starts"], parts["qlens"], parts["qstart"] = (
                batch.starts,
                batch.query_lens,
                qstart,
            )
        if not decode or device_pages or self.flashinfer is None:
            parts["rows"] = batch.rows + [self.block_tables.pad_row] * pad
        if device_pages:
            parts["pages"] = pages + [1] * pad
            parts["page_start"] = indptr[:-1]
        parts.update(
            table_rows=batch.table_rows,
            table_cols=batch.table_cols,
            table_vals=batch.table_vals,
            sample_rows=batch.sample_rows,
            top_ks=batch.top_ks,
            pending_dst=batch.pending_dst,
            pending_src=batch.pending_src,
        )
        dev = batch.dev
        dev.update(zip(parts, self._upload(list(parts.values()), torch.long), strict=True))
        tables = self.block_tables
        tables.write(dev["table_rows"], dev["table_cols"], dev["table_vals"])
        if not decode:
            total = len(batch.input_ids)
            seq = torch.repeat_interleave(
                torch.arange(n, device=self.device), dev["qlens"], output_size=total
            )
            positions = dev["starts"][seq] + torch.arange(total, device=self.device)
            positions -= dev["qstart"][seq]
            dev["positions"] = positions
            dev["slots"] = tables.slots(dev["rows"][seq], positions, bs)
        if self.flashinfer is None:
            return
        if device_pages:
            dev["page_indices"] = tables.page_indices(
                dev["rows"], dev["pages"], dev["page_start"], indptr[-1]
            )
        else:
            flat = [b for table, p in zip(batch.tables, pages, strict=True) for b in table[:p]]
            flat += [self.scratch_block] * pad
            (indices,) = pinned_views([flat], torch.int32, pin=self._pin)
            dev["page_indices"] = indices.to(self.device, non_blocking=True)

    def gather_pending(self, input_ids: torch.Tensor, batch: _Batch) -> None:
        """Fill in input tokens the previous step sampled, straight from device memory."""
        if batch.pending_dst:
            assert self._prev_tokens is not None
            tokens = self._prev_tokens.index_select(0, batch.dev["pending_src"])
            input_ids.index_copy_(0, batch.dev["pending_dst"], tokens)

    def _run_model(self, batch: _Batch) -> torch.Tensor:
        graphs = self.graphs
        if graphs is not None and batch.decode_only and graphs.can_run(batch.num_seqs):
            self._upload_inputs(batch, graphs.padded_size(batch.num_seqs), graph=True)
            return graphs.run(batch)
        self._upload_inputs(batch, batch.num_seqs)
        return self._eager_forward(batch)

    def _eager_forward(self, batch: _Batch) -> torch.Tensor:
        dev, n = batch.dev, batch.num_seqs
        ids = dev["ids"]
        self.gather_pending(ids, batch)
        if self.flashinfer is not None:
            lens: list[torch.Tensor] = [self._unused] * 4
            tables = self._cache_marker
        else:  # the torch and flash-attn kernels take lengths and block tables
            cu_q, cu_k = [0], [0]
            for q_len, k_len in zip(batch.query_lens, batch.seq_lens, strict=True):
                cu_q.append(cu_q[-1] + q_len)
                cu_k.append(cu_k[-1] + k_len)
            lens = self._upload([batch.query_lens, batch.seq_lens, cu_q, cu_k], torch.int32)
            width = max(batch.indptr[i + 1] - batch.indptr[i] for i in range(n))
            tables = self.block_tables.table[dev["rows"], :width]
        meta = AttentionMetadata(
            backend=self.backend,
            num_seqs=n,
            max_seqlen_q=max(batch.query_lens),
            max_seqlen_k=max(batch.seq_lens),
            query_lens=lens[0],
            seq_lens_k=lens[1],
            cu_seqlens_q=lens[2],
            cu_seqlens_k=lens[3],
            slot_mapping=dev["slots"],
            block_tables=tables,
            decode_only=batch.decode_only,
        )
        if self.flashinfer is not None:
            indices = dev["page_indices"]
            if batch.decode_only:
                meta.flashinfer = self.flashinfer.plan_layout(
                    batch.indptr, indices, batch.last_page_len
                )
            else:
                qo_indptr = [0]
                for q_len in batch.query_lens:
                    qo_indptr.append(qo_indptr[-1] + q_len)
                if batch.num_decode:
                    meta.flashinfer = self.flashinfer.plan_split(
                        batch.indptr, indices, batch.last_page_len, qo_indptr, batch.num_decode
                    )
                else:
                    meta.flashinfer = self.flashinfer.plan_layout(
                        batch.indptr, indices, batch.last_page_len, qo_indptr
                    )
        hidden: torch.Tensor = self.model(ids, dev["positions"], meta)
        return hidden

    def _sample(self, hidden: torch.Tensor, batch: _Batch) -> torch.Tensor:
        logits = self.model.compute_logits(hidden[batch.dev["sample_rows"]]).float()
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
