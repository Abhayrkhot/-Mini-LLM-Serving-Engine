"""CUDA-only checks: the flash-attn backend must agree with the torch reference backend.

Skipped unless a CUDA device and the flash-attn package are present
(run them with scripts/gpu_validate.sh).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.layers.attention import flash_attention, torch_attention, write_kv_cache
from tests.test_attention import random_paged_batch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and importlib.util.find_spec("flash_attn")),
        reason="needs CUDA and flash-attn",
    ),
]


@pytest.mark.parametrize("seed", range(8))
def test_flash_backend_matches_torch_backend(seed: int) -> None:
    b = random_paged_batch(
        seed,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=256,
        max_len=700,
        device="cuda",
        dtype=torch.bfloat16,
        backend="flash_attn",
    )
    assert b.meta.slot_mapping is not None
    write_kv_cache(b.k, b.v, b.k_cache, b.v_cache, b.meta.slot_mapping)
    flash = flash_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    ref = torch_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    torch.testing.assert_close(flash.float(), ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(flash.float(), b.reference, atol=3e-2, rtol=3e-2)


def test_flash_engine_agrees_with_torch_engine(tiny_model_dirs) -> None:
    prompts = [[(7 * i + j) % 250 + 3 for j in range(10 + 13 * i)] for i in range(6)]
    params = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)

    def run(backend: str) -> list[list[int]]:
        engine = LLMEngine(
            EngineConfig(
                model=str(tiny_model_dirs["qwen2"]),
                device="cuda",
                dtype="bfloat16",
                attention_backend=backend,
                block_size=256,
                max_model_len=256,
                num_kv_blocks=16,
                max_num_batched_tokens=64,  # forces chunked prefill mixed with decodes
            )
        )
        return [r.output_token_ids for r in engine.generate(prompts, params)]

    flash, ref = run("flash_attn"), run("torch")
    agree = sum(
        a == b for fa, ra in zip(flash, ref, strict=True) for a, b in zip(fa, ra, strict=True)
    )
    # bf16 kernels round differently, so allow a rare flipped near-tie.
    assert agree / sum(len(r) for r in ref) > 0.95
