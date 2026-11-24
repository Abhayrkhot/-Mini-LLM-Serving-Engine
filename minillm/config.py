"""Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    model: str
    """Local directory or Hugging Face Hub id of a safetensors checkpoint."""
    tokenizer: str | None = None
    """Tokenizer location; defaults to ``model``."""

    device: str = "auto"
    """``auto`` picks cuda, then mps, then cpu."""
    dtype: str = "auto"
    """``auto``: bfloat16 on cuda/mps, float32 on cpu. Or float16, bfloat16, float32."""
    attention_backend: str = "auto"
    """``auto``: on CUDA, flashinfer, then flash_attn, whichever loads; otherwise torch."""

    block_size: int | None = None
    """Tokens per KV block. Default 256 for flash_attn (its paged kernels require it), else 16."""
    max_model_len: int | None = None
    """Longest prompt + output. Default: the model's limit, lowered to what the KV cache holds."""
    max_num_seqs: int = 128
    """Most sequences running at once."""
    max_num_batched_tokens: int = 2048
    """Token budget per engine step, shared between prompt chunks and decodes."""
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True

    gpu_memory_utilization: float = 0.90
    """CUDA only: fraction of device memory the engine may use, weights and KV cache included."""
    kv_cache_memory_gb: float = 1.0
    """CPU/MPS only: memory reserved for the KV cache."""
    num_kv_blocks: int | None = None
    """Override the KV-cache size in blocks (skips memory profiling)."""

    enforce_eager: bool = False
    """Disable decode CUDA graphs (used with the flashinfer backend on CUDA)."""
    fused_kernels: bool = True
    """With the flashinfer backend, run RMSNorm (with the residual add), rotary
    embeddings and SiLU-and-mul as FlashInfer's fused kernels instead of PyTorch ops."""
    speculative_tokens: int = 0
    """Speculative decoding: draft up to this many tokens per decoding sequence by
    n-gram lookup in its own context, and verify them all in one step. Greedy
    requests only (others decode normally); outputs are unchanged. Turns async
    scheduling off, since a step's token count is not known until it finishes."""
    speculative_ngram_max: int = 3
    """Longest n-gram matched when drafting."""
    async_scheduling: bool = True
    """Schedule and launch step n+1 while the GPU runs step n. A request that stops on
    EOS or a stop token then costs one discarded extra token of compute."""

    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be >= 1")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.attention_backend not in ("auto", "torch", "flash_attn", "flashinfer"):
            raise ValueError("attention_backend must be auto, torch, flash_attn or flashinfer")
