"""Runs the real model: builds batch tensors, owns the KV cache, samples tokens."""

from __future__ import annotations

import importlib
import logging
import os
import time

import torch

from minillm.config import EngineConfig
from minillm.executor.cuda_graphs import DecodeGraphs
from minillm.layers.attention import AttentionMetadata, FlashInferPlanner
from minillm.layers.sampler import sample
from minillm.models.llama import ModelConfig
from minillm.models.loader import (
    build_model,
    load_weights,
    read_eos_token_ids,
    read_json,
    resolve_model_path,
)
from minillm.scheduler import ScheduledSeq
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
        weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        logger.info(
            "Loaded %s (%s) in %.1fs: %.2f GiB of weights, %s on %s, attention=%s",
            config.model,
            mc.model_type,
            time.perf_counter() - start,
            weight_bytes / 2**30,
            str(self.dtype).removeprefix("torch."),
            self.device,
            self.backend,
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.num_kv_blocks = self._init_kv_cache()

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
    def execute(self, scheduled: list[ScheduledSeq]) -> list[int]:
        if not scheduled:
            return []
        logits, temps, top_ps, top_ks = self._forward(scheduled)
        if logits is None:
            return []
        tokens = sample(logits, temps, top_ps, top_ks, self.generator)
        result: list[int] = tokens.tolist()
        return result

    @torch.inference_mode()
    def compute_logits(self, scheduled: list[ScheduledSeq]) -> torch.Tensor | None:
        """Run the batch and return float32 logits for the sequences that sample (for tests)."""
        return self._forward(scheduled)[0]

    def _forward(
        self, scheduled: list[ScheduledSeq]
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = self.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        cu_q, cu_k = [0], [0]
        tables: list[list[int]] = []
        sample_idx: list[int] = []
        temps: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []

        for item in scheduled:
            seq = item.seq
            start = seq.num_computed_tokens
            end = start + item.num_new_tokens
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            pos = start
            while pos < end:  # one contiguous slot range per block
                block_idx, offset = divmod(pos, bs)
                span = min(bs - offset, end - pos)
                base = seq.block_table[block_idx] * bs + offset
                slots.extend(range(base, base + span))
                pos += span
            query_lens.append(end - start)
            seq_lens.append(end)
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(cu_k[-1] + end)
            tables.append(seq.block_table)
            if item.sample:
                sample_idx.append(cu_q[-1] - 1)
                temps.append(seq.params.temperature)
                top_ps.append(seq.params.top_p)
                top_ks.append(seq.params.top_k)

        if self.graphs is not None and max(query_lens) == 1 and self.graphs.can_run(len(scheduled)):
            hidden = self.graphs.run(input_ids, positions, slots, tables, seq_lens)
        else:
            hidden = self._eager_forward(
                input_ids, positions, slots, tables, query_lens, seq_lens, cu_q, cu_k
            )
        temps_t = self._tensor(temps, torch.float32)
        top_ps_t = self._tensor(top_ps, torch.float32)
        top_ks_t = self._tensor(top_ks, torch.long)
        if not sample_idx:
            return None, temps_t, top_ps_t, top_ks_t
        logits = self.model.compute_logits(hidden[self._tensor(sample_idx, torch.long)]).float()
        return logits, temps_t, top_ps_t, top_ks_t

    def _eager_forward(
        self,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        tables: list[list[int]],
        query_lens: list[int],
        seq_lens: list[int],
        cu_q: list[int],
        cu_k: list[int],
    ) -> torch.Tensor:
        width = max(len(t) for t in tables)
        padded = [t + [0] * (width - len(t)) for t in tables]
        meta = AttentionMetadata(
            backend=self.backend,
            num_seqs=len(tables),
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(seq_lens),
            query_lens=self._tensor(query_lens, torch.int32),
            seq_lens_k=self._tensor(seq_lens, torch.int32),
            cu_seqlens_q=self._tensor(cu_q, torch.int32),
            cu_seqlens_k=self._tensor(cu_k, torch.int32),
            slot_mapping=self._tensor(slots, torch.long),
            block_tables=torch.tensor(padded, dtype=torch.int32, pin_memory=self._pin).to(
                self.device, non_blocking=True
            ),
            decode_only=max(query_lens) == 1,
        )
        if self.flashinfer is not None:
            meta.flashinfer = self.flashinfer.plan(tables, seq_lens, query_lens)
        hidden: torch.Tensor = self.model(
            self._tensor(input_ids, torch.long), self._tensor(positions, torch.long), meta
        )
        return hidden
