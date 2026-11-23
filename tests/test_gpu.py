"""CUDA-only checks: each GPU attention backend must agree with the torch reference.

Each backend is skipped unless a CUDA device is present and the backend's
package actually loads (run them with scripts/gpu_validate.sh).
"""

from __future__ import annotations

import pytest
import torch

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.executor.model_runner import _importable
from minillm.layers.attention import (
    FlashInferPlanner,
    flash_attention,
    flashinfer_attention,
    torch_attention,
    write_kv_cache,
)
from tests.test_attention import random_paged_batch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

# backend -> (package, block size its paged kernels accept)
BACKENDS = {"flash_attn": ("flash_attn", 256), "flashinfer": ("flashinfer", 16)}


def backend_param(name: str) -> object:
    package = BACKENDS[name][0]
    available = torch.cuda.is_available() and _importable(package)
    return pytest.param(
        name, marks=pytest.mark.skipif(not available, reason=f"{package} not usable")
    )


GPU_BACKENDS = [backend_param(name) for name in BACKENDS]


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seed", range(6))
def test_kernel_matches_torch_reference(backend: str, head_dim: int, seed: int) -> None:
    block_size = BACKENDS[backend][1]
    b = random_paged_batch(
        seed,
        num_heads=14,
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=block_size,
        max_len=700,
        device="cuda",
        dtype=torch.bfloat16,
        backend=backend,
    )
    assert b.meta.slot_mapping is not None
    write_kv_cache(b.k, b.v, b.k_cache, b.v_cache, b.meta.slot_mapping)
    if backend == "flashinfer":
        planner = FlashInferPlanner(
            torch.device("cuda"), 14, 2, head_dim, block_size, torch.bfloat16, workspace_mb=64
        )
        b.meta.flashinfer = planner.plan(b.tables, b.k_lens, b.q_lens)
        out = flashinfer_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    else:
        out = flash_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    ref = torch_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out.float(), b.reference, atol=3e-2, rtol=3e-2)


@pytest.fixture(scope="module")
def tiny_qwen2_hd64(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A random Qwen2 with head_dim 64, a size FlashInfer ships prebuilt kernels for."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        initializer_range=0.2,
        tie_word_embeddings=True,
    )
    path = tmp_path_factory.mktemp("qwen2-hd64")
    Qwen2ForCausalLM(config).save_pretrained(path)
    return str(path)


def _teacher_forced_agreement(engine: LLMEngine, hf: object, prompts: list[list[int]]) -> float:
    """Fraction of greedy tokens that are the fp32 reference's argmax at their position."""
    params = SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True)
    total = agree = 0
    for res in engine.generate(prompts, params):
        full = res.prompt_token_ids + res.output_token_ids
        with torch.inference_mode():
            logits = hf(torch.tensor([full], device="cuda")).logits[0]  # type: ignore[operator]
        start = len(res.prompt_token_ids) - 1
        best = logits[start : start + len(res.output_token_ids)].argmax(-1).tolist()
        agree += sum(b == t for b, t in zip(best, res.output_token_ids, strict=True))
        total += len(res.output_token_ids)
    return agree / total


ENGINE_CASES = [
    pytest.param(
        "flashinfer", False, id="flashinfer-cudagraphs", marks=backend_param("flashinfer").marks
    ),
    pytest.param(
        "flashinfer", True, id="flashinfer-eager", marks=backend_param("flashinfer").marks
    ),
    pytest.param(
        "flash_attn", True, id="flash_attn-eager", marks=backend_param("flash_attn").marks
    ),
]


@pytest.mark.parametrize(("backend", "eager"), ENGINE_CASES)
def test_engine_is_as_accurate_as_reference_backend(
    backend: str, eager: bool, tiny_qwen2_hd64: str
) -> None:
    """In bf16, the fast backend must pick the fp32 model's argmax as often as the torch backend.

    Teacher-forced, like tests/test_real_model.py: comparing two free-running bf16
    generations would let one flipped near-tie rewrite the rest of a sequence.
    bf16 alone costs this random model a few percent of argmax agreement, so the
    torch backend (exact in fp32) is the yardstick rather than a fixed threshold.
    The cache is small enough that preemption and recompute happen, and with
    CUDA graphs on, decode batches of every size get padded and replayed.
    """
    from tests.conftest import load_hf_model

    prompts = [[(7 * i + j) % 500 + 3 for j in range(10 + 37 * i)] for i in range(8)]
    hf = load_hf_model(tiny_qwen2_hd64, torch.float32).to("cuda")
    scores = {}
    for name in (backend, "torch"):
        block_size = BACKENDS[backend][1]
        engine = LLMEngine(
            EngineConfig(
                model=tiny_qwen2_hd64,
                device="cuda",
                dtype="bfloat16",
                attention_backend=name,
                block_size=block_size,
                max_model_len=512,
                num_kv_blocks=1024 // block_size if block_size < 256 else 8,
                max_num_batched_tokens=96,  # chunked prefill mixed with decodes
                enforce_eager=eager,
            )
        )
        scores[name] = _teacher_forced_agreement(engine, hf, prompts)
        if block_size < 256:  # 1,024 cache slots for ~2,300 tokens
            assert engine.scheduler.num_preemptions > 0
        graphs = getattr(engine.executor, "graphs", None)
        if name == backend and not eager:
            assert graphs is not None and graphs.replays > 0
    assert scores[backend] >= scores["torch"] - 0.02, scores
