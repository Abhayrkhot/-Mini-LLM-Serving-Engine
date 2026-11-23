"""Greedy parity on a real checkpoint (opt-in: set MINILLM_TEST_MODEL).

    MINILLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m slow tests/test_real_model.py

The engine generates greedily (with chunked prefill forced on). Then one
Hugging Face forward pass over prompt + generated tokens gives the reference
logits at every position, and each generated token must be HF's argmax there.
Teacher forcing keeps one numerical near-tie from cascading into a different
continuation, and avoids ``generate()``, which merges the checkpoint's
generation_config (Qwen2.5-Instruct sets repetition_penalty=1.1) into greedy
decoding.
"""

from __future__ import annotations

import os

import pytest
import torch

from minillm import EngineConfig, LLMEngine, SamplingParams

MODEL = os.environ.get("MINILLM_TEST_MODEL")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(MODEL is None, reason="set MINILLM_TEST_MODEL to run"),
]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain paged attention in one sentence:",
    "1, 1, 2, 3, 5, 8,",
    "Translate to German: The weather is nice today.",
    "A haiku about GPUs:",
]
# A disagreement is tolerated only where HF's own top-2 logits are this close.
NEAR_TIE = 1e-3


def test_greedy_tokens_are_hf_argmax() -> None:
    from tests.conftest import load_hf_model

    assert MODEL is not None
    device = os.environ.get("MINILLM_TEST_DEVICE", "cpu")
    engine = LLMEngine(
        EngineConfig(
            model=MODEL,
            device=device,
            dtype="float32",
            attention_backend="torch",
            max_model_len=512,
            max_num_batched_tokens=24,  # chunked prefill mixed with decodes
        )
    )
    params = SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True)
    results = engine.generate(PROMPTS, params)

    hf = load_hf_model(MODEL, torch.float32).to(device)
    checked = 0
    for res in results:
        full = res.prompt_token_ids + res.output_token_ids
        with torch.inference_mode():
            logits = hf(torch.tensor([full], device=device)).logits[0].float()
        start = len(res.prompt_token_ids) - 1
        for t, token in enumerate(res.output_token_ids):
            row = logits[start + t]
            best = int(row.argmax())
            if best != token:
                gap = float(row[best] - row[token])
                assert gap < NEAR_TIE, f"step {t} of {res.text!r}: logit gap {gap:.2e}"
            checked += 1
    assert checked == len(PROMPTS) * 48


@pytest.mark.parametrize("backend", ["flashinfer", "flash_attn"])
def test_gpu_backend_matches_hf_in_bf16(backend: str) -> None:
    """The fast CUDA kernels in bfloat16, teacher-forced against HF in bfloat16."""
    from minillm.executor.model_runner import _importable
    from tests.conftest import load_hf_model

    if not torch.cuda.is_available() or not _importable(backend):
        pytest.skip(f"needs CUDA and {backend}")
    assert MODEL is not None
    engine = LLMEngine(
        EngineConfig(
            model=MODEL,
            device="cuda",
            dtype="bfloat16",
            attention_backend=backend,
            max_model_len=1024,
            max_num_batched_tokens=24,
        )
    )
    results = engine.generate(
        PROMPTS, SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True)
    )
    hf = load_hf_model(MODEL, torch.bfloat16).to("cuda")
    total = agree = 0
    worst = 0.0
    for res in results:
        full = res.prompt_token_ids + res.output_token_ids
        with torch.inference_mode():
            logits = hf(torch.tensor([full], device="cuda")).logits[0].float()
        start = len(res.prompt_token_ids) - 1
        for t, token in enumerate(res.output_token_ids):
            row = logits[start + t]
            best = int(row.argmax())
            total += 1
            if best == token:
                agree += 1
            else:
                worst = max(worst, float(row[best] - row[token]))
    # bf16 kernels round differently from HF's SDPA, so only near-ties may flip.
    assert agree / total > 0.97, f"{agree}/{total} agree"
    assert worst < 0.5, f"largest logit gap at a disagreement: {worst:.3f}"
