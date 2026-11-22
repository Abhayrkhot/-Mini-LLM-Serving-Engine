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
    from transformers import AutoModelForCausalLM

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

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
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
