"""Load Hugging Face safetensors checkpoints into CausalLM."""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

import torch
from safetensors import safe_open

from minillm.models.llama import CausalLM, ModelConfig

logger = logging.getLogger(__name__)

# Checkpoint projection -> (fused parameter, shard name).
_FUSED = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", "gate"),
    "up_proj": ("gate_up_proj", "up"),
}
_SHARDS = {"qkv_proj": ("q", "k", "v"), "gate_up_proj": ("gate", "up")}


def resolve_model_path(model: str) -> str:
    """Return a local directory for ``model``, downloading from the Hugging Face Hub if needed."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    logger.info("Fetching %s from the Hugging Face Hub", model)
    return snapshot_download(
        model,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
    )


def read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def read_eos_token_ids(model_dir: str, hf_config: dict[str, Any]) -> list[int]:
    """EOS ids from generation_config.json (which may list several), else config.json."""
    eos: Any = None
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_path):
        eos = read_json(gen_path).get("eos_token_id")
    if eos is None:
        eos = hf_config.get("eos_token_id")
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, list) else [int(eos)]


def _shard_range(cfg: ModelConfig, fused: str, shard: str) -> tuple[int, int]:
    if fused == "qkv_proj":
        q, kv = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
        return {"q": (0, q), "k": (q, kv), "v": (q + kv, kv)}[shard]
    inter = cfg.intermediate_size
    return {"gate": (0, inter), "up": (inter, inter)}[shard]


def build_model(cfg: ModelConfig, device: torch.device, dtype: torch.dtype) -> CausalLM:
    """Construct the model on ``device`` without spending time on random init."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = CausalLM(cfg)
    finally:
        torch.set_default_dtype(previous)
    model = model.to_empty(device=device)
    model.tie_weights()
    model.model.rotary_emb.build(device)
    return model.eval()


def load_weights(model: CausalLM, model_dir: str) -> None:
    cfg = model.config
    params = dict(model.named_parameters())
    expected: set[tuple[str, str | None]] = set()
    for name in params:
        fused = name.split(".")[-2]
        if fused in _SHARDS:
            expected.update((name, shard) for shard in _SHARDS[fused])
        else:
            expected.add((name, None))

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")

    loaded: set[tuple[str, str | None]] = set()
    with torch.no_grad():
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for ckpt_name in f.keys():  # noqa: SIM118 - safe_open is not a dict
                    if ckpt_name.endswith("rotary_emb.inv_freq"):
                        continue
                    if cfg.tie_word_embeddings and ckpt_name == "lm_head.weight":
                        continue
                    target, shard = ckpt_name, None
                    parts = ckpt_name.split(".")
                    if len(parts) >= 2 and parts[-2] in _FUSED:
                        fused, shard = _FUSED[parts[-2]]
                        target = ".".join([*parts[:-2], fused, parts[-1]])
                    param = params.get(target)
                    if param is None:
                        raise KeyError(f"checkpoint tensor {ckpt_name!r} has no matching parameter")
                    tensor = f.get_tensor(ckpt_name)
                    dest = param.data
                    if shard is not None:
                        offset, size = _shard_range(cfg, target.split(".")[-2], shard)
                        dest = dest.narrow(0, offset, size)
                    if dest.shape != tensor.shape:
                        raise ValueError(
                            f"{ckpt_name}: checkpoint shape {tuple(tensor.shape)} "
                            f"!= parameter shape {tuple(dest.shape)}"
                        )
                    dest.copy_(tensor)
                    loaded.add((target, shard))

    missing = expected - loaded
    if missing:
        names = sorted(f"{n}[{s}]" if s else n for n, s in missing)
        raise ValueError(f"checkpoint is missing {len(names)} tensors, e.g. {names[:5]}")
