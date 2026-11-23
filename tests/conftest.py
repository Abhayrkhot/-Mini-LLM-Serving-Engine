from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from minillm.executor.fake import ByteTokenizer

TINY = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    # Larger than the usual 0.02 so logits are peaked and greedy decoding is
    # decided by the model, not by float rounding.
    initializer_range=0.2,
)


def _tiny_models() -> dict[str, tuple[Any, Any]]:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    llama3_rope = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 32,
    }
    models: dict[str, tuple[Any, Any]] = {
        "qwen2": (Qwen2Config(**TINY, tie_word_embeddings=True), Qwen2ForCausalLM),
        "llama3": (LlamaConfig(**TINY, rope_scaling=llama3_rope), LlamaForCausalLM),
    }
    try:  # Qwen3 arrived in transformers 4.51; older installs (e.g. vLLM images) skip it.
        from transformers import Qwen3Config, Qwen3ForCausalLM

        models["qwen3"] = (Qwen3Config(**TINY, head_dim=16), Qwen3ForCausalLM)
    except ImportError:
        pass
    return models


TINY_MODEL_NAMES = sorted(_tiny_models())


def load_hf_model(path: Any, dtype: torch.dtype) -> Any:
    """AutoModelForCausalLM.from_pretrained across transformers 4.x (torch_dtype) and 5.x (dtype)."""
    import transformers
    from transformers import AutoModelForCausalLM

    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(path, **{key: dtype}).eval()


@pytest.fixture(scope="session")
def tiny_model_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Randomly initialised Hugging Face checkpoints, saved as safetensors."""
    root = tmp_path_factory.mktemp("tiny-models")
    dirs = {}
    for name, (config, cls) in _tiny_models().items():
        torch.manual_seed(0)
        model = cls(config).eval()
        path = root / name
        model.save_pretrained(path)
        dirs[name] = path
    return dirs


@pytest.fixture(scope="session")
def hf_models(tiny_model_dirs: dict[str, Path]) -> dict[str, Any]:
    return {name: load_hf_model(path, torch.float32) for name, path in tiny_model_dirs.items()}


def hf_greedy(model: Any, prompt: list[int], max_tokens: int) -> list[int]:
    """Reference greedy decoding: full forward each step, no KV cache, no batching."""
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(torch.tensor([ids])).logits[0, -1]
            ids.append(int(logits.argmax()))
    return ids[len(prompt) :]


# Byte-level tokenizer shared with the fake-server benchmark.
CharTokenizer = ByteTokenizer
