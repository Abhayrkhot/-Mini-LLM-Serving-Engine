"""Decoder-only transformer covering the Llama family: Llama 2/3, Mistral, Qwen2/2.5 and Qwen3.

Written against the Hugging Face reference implementations and checked token
for token against them in tests/test_model_parity.py. The differences between
these architectures are small: Qwen2 adds a bias to the q/k/v projections,
Qwen3 adds per-head RMSNorm on q and k, Llama 3.x rescales RoPE frequencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from minillm.layers.attention import AttentionMetadata, PagedAttention
from minillm.layers.fused import FusedOps

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3")


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    qkv_bias: bool
    o_bias: bool
    mlp_bias: bool
    qk_norm: bool

    @classmethod
    def from_hf(cls, cfg: dict[str, Any]) -> ModelConfig:
        model_type = cfg.get("model_type", "")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"unsupported model_type {model_type!r}; supported: {SUPPORTED_MODEL_TYPES}"
            )
        # transformers >= 5 nests RoPE settings under rope_parameters.
        rope_params = cfg.get("rope_parameters") or {}
        rope_theta = cfg.get("rope_theta") or rope_params.get("rope_theta") or 10000.0
        scaling = cfg.get("rope_scaling") or rope_params or None
        if scaling is not None:
            rope_type = scaling.get("rope_type", scaling.get("type", "default"))
            if rope_type == "default":
                scaling = None
            elif rope_type != "llama3":
                raise ValueError(f"unsupported rope scaling type {rope_type!r}")

        uses_sliding_window = (
            cfg.get("use_sliding_window", False)
            if model_type in ("qwen2", "qwen3")
            else model_type == "mistral" and cfg.get("sliding_window") is not None
        )
        if uses_sliding_window:
            raise ValueError("sliding-window attention is not supported")

        num_heads = cfg["num_attention_heads"]
        attention_bias = bool(cfg.get("attention_bias", False))
        return cls(
            model_type=model_type,
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=cfg.get("num_key_value_heads") or num_heads,
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            rope_scaling=scaling,
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            qkv_bias=True if model_type == "qwen2" else attention_bias,
            o_bias=False if model_type == "qwen2" else attention_bias,
            mlp_bias=bool(cfg.get("mlp_bias", False)),
            qk_norm=model_type == "qwen3",
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dtype)


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling: dict[str, Any]) -> torch.Tensor:
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables (float32), gathered once per step by position.

    ``cos_sin_cache`` holds the same tables side by side, [max_position, head_dim],
    the layout FlashInfer's fused rotary kernel reads.
    """

    def __init__(
        self, head_dim: int, max_position: int, base: float, scaling: dict[str, Any] | None
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scaling = scaling
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)
        self.build()

    def build(self, device: torch.device | None = None) -> None:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float()
        inv_freq = 1.0 / (self.base ** (exponent / self.head_dim))
        if self.scaling is not None:
            inv_freq = _llama3_inv_freq(inv_freq, self.scaling)
        t = torch.arange(self.max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos = freqs.cos()
        self.sin = freqs.sin()
        self.cos_sin_cache = torch.cat((self.cos, self.sin), dim=-1)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Half-split ("rotate_half") convention used by Hugging Face checkpoints.
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = cfg.num_heads * cfg.head_dim
        self.kv_size = cfg.num_kv_heads * cfg.head_dim
        # q, k and v share one GEMM; the loader fills the three slices.
        self.qkv_proj = nn.Linear(
            cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=cfg.qkv_bias
        )
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=cfg.o_bias)
        if cfg.qk_norm:
            self.q_norm: RMSNorm | None = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
            self.k_norm: RMSNorm | None = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None
        self.attn = PagedAttention(
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, scale=cfg.head_dim**-0.5
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = rope
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        out = self.attn(q, k, v, meta)
        return self.o_proj(out.reshape(-1, self.q_size))

    def forward_fused(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        meta: AttentionMetadata,
        ops: FusedOps,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        kv = None
        if self.q_norm is not None and self.k_norm is not None:
            # Per-head norms over contiguous [tokens * heads, head_dim] rows.
            q = ops.rmsnorm(q.reshape(-1, self.head_dim), self.q_norm.weight, self.q_norm.eps)
            k = ops.rmsnorm(k.reshape(-1, self.head_dim), self.k_norm.weight, self.k_norm.eps)
            q, k = q.view(-1, self.q_size), k.view(-1, self.kv_size)
        else:
            # k and v sit side by side in the qkv buffer: one cache write covers both.
            kv = qkv[:, self.q_size :].view(-1, 2, self.num_kv_heads, self.head_dim)
        ops.rope(positions, q, k, self.head_dim, cos_sin_cache)  # in place, on the qkv buffer
        out = self.attn(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            meta,
            kv=kv,
        )
        return self.o_proj(out.reshape(-1, self.q_size))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def forward_fused(self, x: torch.Tensor, ops: FusedOps) -> torch.Tensor:
        return self.down_proj(ops.silu_and_mul(self.gate_up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), rope, meta)
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward_fused(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        meta: AttentionMetadata,
        ops: FusedOps,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carries the residual stream separately: ``x`` is the previous sublayer's
        output, still to be added to ``residual`` (which is None at the first layer)."""
        norm1, norm2 = self.input_layernorm, self.post_attention_layernorm
        if residual is None:
            residual = x
            h = ops.rmsnorm(x, norm1.weight, norm1.eps)
        else:
            h = ops.add_rmsnorm(x, residual, norm1.weight, norm1.eps)
        h = self.self_attn.forward_fused(h, positions, cos_sin_cache, meta, ops)
        h = ops.add_rmsnorm(h, residual, norm2.weight, norm2.eps)
        return self.mlp.forward_fused(h, ops), residual


class TransformerModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, cfg.rope_scaling
        )
        self.fused: FusedOps | None = None

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if self.fused is not None:
            return self._forward_fused(x, positions, meta, self.fused)
        rope = self.rotary_emb(positions)
        for layer in self.layers:
            x = layer(x, rope, meta)
        return self.norm(x)

    def _forward_fused(
        self, x: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata, ops: FusedOps
    ) -> torch.Tensor:
        cache = self.rotary_emb.cos_sin_cache
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = cast(DecoderLayer, layer).forward_fused(
                x, residual, positions, cache, meta, ops
            )
        assert residual is not None
        return ops.add_rmsnorm(x, residual, self.norm.weight, self.norm.eps)


class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = TransformerModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        """Returns final hidden states for every packed token, [T, hidden]."""
        return self.model(input_ids, positions, meta)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def enable_fused_kernels(self, enabled: bool = True) -> None:
        """Run norms, rotary embeddings and SiLU-and-mul as FlashInfer's fused kernels."""
        self.model.fused = FusedOps() if enabled else None

    @property
    def attention_layers(self) -> list[PagedAttention]:
        return [cast(DecoderLayer, layer).self_attn.attn for layer in self.model.layers]
