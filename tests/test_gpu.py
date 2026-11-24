"""CUDA-only checks: each GPU attention backend must agree with the torch reference.

Each backend is skipped unless a CUDA device is present and the backend's
package actually loads (run them with scripts/gpu_validate.sh).
"""

from __future__ import annotations

import time

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


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.skipif(not _importable("flashinfer"), reason="flashinfer not usable")
def test_split_mixed_attention_matches_torch_reference(head_dim: int, seed: int) -> None:
    """A mixed step planned split (decodes through the decode kernel, chunks through
    the prefill kernel) computes the same attention as the reference."""
    b = random_paged_batch(
        seed,
        num_heads=14,
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=16,
        max_len=700,
        device="cuda",
        dtype=torch.bfloat16,
        backend="flashinfer",
        decodes_first=True,
    )
    assert b.meta.slot_mapping is not None
    write_kv_cache(b.k, b.v, b.k_cache, b.v_cache, b.meta.slot_mapping)
    planner = FlashInferPlanner(
        torch.device("cuda"), 14, 2, head_dim, 16, torch.bfloat16, workspace_mb=64
    )
    from minillm.layers.attention import paged_kv_layout, pinned_views

    indptr, indices, last = paged_kv_layout(b.tables, b.k_lens, 16)
    qo = [0]
    for q_len in b.q_lens:
        qo.append(qo[-1] + q_len)
    (indices_h,) = pinned_views([indices], torch.int32, pin=True)
    num_decode = len(b.q_lens) - 2  # decodes first, then a chunk and a full prompt
    b.meta.flashinfer = planner.plan_split(indptr, indices_h.cuda(), last, qo, num_decode)
    out = flashinfer_attention(b.q, b.k, b.v, b.k_cache, b.v_cache, b.meta, b.scale)
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


FLASHINFER = backend_param("flashinfer").marks  # type: ignore[attr-defined]
ENGINE_CASES = [  # (backend, eager, fused kernels)
    pytest.param("flashinfer", False, True, id="flashinfer-cudagraphs", marks=FLASHINFER),
    pytest.param("flashinfer", True, True, id="flashinfer-eager", marks=FLASHINFER),
    pytest.param("flashinfer", False, False, id="flashinfer-unfused", marks=FLASHINFER),
    pytest.param(
        "flash_attn", True, False, id="flash_attn-eager", marks=backend_param("flash_attn").marks
    ),
]


@pytest.mark.parametrize(("backend", "eager", "fused"), ENGINE_CASES)
def test_engine_is_as_accurate_as_reference_backend(
    backend: str, eager: bool, fused: bool, tiny_qwen2_hd64: str
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

    # 16 prompts x 48 tokens: on this random model ~7% of tokens are near-ties that bf16
    # can flip either way, so fewer tokens leave the comparison mostly noise.
    prompts = [[(7 * i + j) % 500 + 3 for j in range(10 + 19 * i)] for i in range(16)]
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
                fused_kernels=fused,
            )
        )
        scores[name] = _teacher_forced_agreement(engine, hf, prompts)
        if block_size < 256:  # 1,024 cache slots for ~3,200 tokens
            assert engine.scheduler.num_preemptions > 0
        graphs = getattr(engine.executor, "graphs", None)
        if name == backend and not eager:
            assert graphs is not None and graphs.replays > 0
    assert scores[backend] >= scores["torch"] - 0.02, scores


@pytest.mark.parametrize(("backend", "eager", "fused"), ENGINE_CASES[:3])
def test_async_scheduling_matches_sync_exactly(
    backend: str, eager: bool, fused: bool, tiny_qwen2_hd64: str
) -> None:
    """Async scheduling changes when the CPU sees each token, not what is computed.

    With a roomy cache and only length limits, both modes run the same batch at
    every step, so even bf16 outputs (and sampled ones, whose random draws line
    up) must match exactly. This exercises the GPU-side gather of the previous
    step's tokens in graph replays and in eager steps that mix prompt chunks in.
    """
    prompts = [[(11 * i + j) % 500 + 3 for j in range(5 + 29 * i)] for i in range(8)]
    params = [
        SamplingParams(max_tokens=24 + 3 * i, temperature=0.0, ignore_eos=True) for i in range(7)
    ]
    params.append(
        SamplingParams(max_tokens=40, temperature=0.8, top_p=0.9, top_k=20, ignore_eos=True)
    )
    outputs = {}
    for async_scheduling in (False, True):
        engine = LLMEngine(
            EngineConfig(
                model=tiny_qwen2_hd64,
                device="cuda",
                dtype="bfloat16",
                attention_backend=backend,
                max_model_len=512,
                num_kv_blocks=512,
                max_num_batched_tokens=64,
                enforce_eager=eager,
                fused_kernels=fused,
                async_scheduling=async_scheduling,
            )
        )
        outputs[async_scheduling] = [r.output_token_ids for r in engine.generate(prompts, params)]
    assert outputs[True] == outputs[False]


def _spin_cycles_per_ms() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1_000_000)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    return 10_000_000 / start.elapsed_time(end)


@pytest.mark.parametrize("eager", [False, True], ids=["cudagraphs", "eager"])
@pytest.mark.skipif(not _importable("flashinfer"), reason="flashinfer not usable")
def test_launching_a_step_never_waits_for_the_gpu(eager: bool, tiny_qwen2_hd64: str) -> None:
    """Any device-to-host sync while launching a step would stall async scheduling.

    Queue half a second of GPU spinning, then step the engine: it launches the
    next step behind the spin and collects the previous one, queued before it.
    A launch that synchronized anywhere (PyTorch, FlashInfer's planning) would
    take the whole half second. Covers decode steps (graph replay or eager),
    a step with a prompt chunk, and the sampling path for temperature > 0.
    """
    engine = LLMEngine(
        EngineConfig(
            model=tiny_qwen2_hd64,
            device="cuda",
            dtype="bfloat16",
            attention_backend="flashinfer",
            max_model_len=1024,
            max_num_batched_tokens=64,
            enforce_eager=eager,
        )
    )
    sampled = SamplingParams(max_tokens=300, temperature=0.8, top_p=0.9, top_k=20, ignore_eos=True)
    engine.add_request(list(range(3, 40)), sampled)
    for _ in range(4):
        engine.step()
    spin_ms = 500
    cycles = int(spin_ms * _spin_cycles_per_ms())
    for new_prompt in (0, 200):  # a decode-only step, then one with a prompt chunk
        if new_prompt:
            engine.add_request(list(range(5, 5 + new_prompt)), SamplingParams(max_tokens=4))
        engine.step()
        torch.cuda._sleep(cycles)
        t0 = time.perf_counter()
        engine.step()
        elapsed_ms = 1e3 * (time.perf_counter() - t0)
        behind = torch.cuda.Event()
        behind.record()
        assert not behind.query(), "the GPU finished spinning before the step returned"
        assert elapsed_ms < spin_ms / 5, f"step took {elapsed_ms:.0f} ms: something synchronized"
    while engine.has_unfinished_requests():
        engine.step()


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.skipif(not _importable("flashinfer"), reason="flashinfer not usable")
def test_fused_ops_match_pytorch_reference(head_dim: int) -> None:
    """Each fused kernel computes what the reference PyTorch ops do, up to bf16 rounding.

    Inputs mimic the real call sites: q and k are strided slices of one qkv tensor,
    and positions go up to the model's longest context.
    """
    from minillm.layers.fused import FusedOps
    from minillm.models.llama import RMSNorm, RotaryEmbedding, apply_rotary

    torch.manual_seed(head_dim)
    ops = FusedOps()
    dev, bf16 = "cuda", torch.bfloat16
    tokens, hidden, heads, kv_heads, inter = 37, 896, 14, 2, 4864

    def close(actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    norm = RMSNorm(hidden, 1e-6).to(dev, bf16)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(tokens, hidden, device=dev, dtype=bf16)
    residual = torch.randn_like(x)
    close(ops.rmsnorm(x, norm.weight, norm.eps), norm(x))
    expected_residual, expected_out = x + residual, norm(x + residual)
    out = ops.add_rmsnorm(x.clone(), residual, norm.weight, norm.eps)
    close(residual, expected_residual)
    close(out, expected_out)

    rope = RotaryEmbedding(head_dim, 32768, 1e6, None)
    rope.build(torch.device(dev))
    q_size, kv_size = heads * head_dim, kv_heads * head_dim
    qkv = torch.randn(tokens, q_size + 2 * kv_size, device=dev, dtype=bf16)
    positions = torch.randint(0, 32768, (tokens,), device=dev)
    q, k, _ = qkv.clone().split([q_size, kv_size, kv_size], dim=-1)
    cos, sin = rope(positions)
    expected_q = apply_rotary(q.reshape(tokens, heads, head_dim), cos, sin)
    expected_k = apply_rotary(k.reshape(tokens, kv_heads, head_dim), cos, sin)
    ops.rope(positions, q, k, head_dim, rope.cos_sin_cache)  # in place on the strided views
    close(q.reshape(tokens, heads, head_dim), expected_q)
    close(k.reshape(tokens, kv_heads, head_dim), expected_k)

    gate_up = torch.randn(tokens, 2 * inter, device=dev, dtype=bf16)
    gate, up = gate_up.chunk(2, dim=-1)
    close(ops.silu_and_mul(gate_up), torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize("eager", [False, True], ids=["cudagraphs", "eager"])
@pytest.mark.skipif(not _importable("flashinfer"), reason="flashinfer not usable")
def test_device_built_page_lists_match_host_built(
    eager: bool, tiny_qwen2_hd64: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FlashInfer's page list is built in Python for small batches and on the GPU
    from the block tables for large ones; both must give identical outputs."""
    from minillm.executor.model_runner import ModelRunner

    prompts = [[(13 * i + j) % 500 + 3 for j in range(7 + 31 * i)] for i in range(8)]
    params = SamplingParams(max_tokens=40, temperature=0.0, ignore_eos=True)
    outputs = {}
    for limit in (ModelRunner.HOST_PAGES_MAX, 0):
        monkeypatch.setattr(ModelRunner, "HOST_PAGES_MAX", limit)
        engine = LLMEngine(
            EngineConfig(
                model=tiny_qwen2_hd64,
                device="cuda",
                dtype="bfloat16",
                attention_backend="flashinfer",
                max_model_len=512,
                num_kv_blocks=40,  # 640 slots for ~1,240 tokens: preemption rebuilds rows
                max_num_batched_tokens=64,
                enforce_eager=eager,
            )
        )
        outputs[limit] = [r.output_token_ids for r in engine.generate(prompts, params)]
        assert engine.scheduler.num_preemptions > 0
    assert outputs[0] == outputs[ModelRunner.HOST_PAGES_MAX]


@pytest.mark.skipif(not _importable("flashinfer"), reason="flashinfer not usable")
def test_speculative_decoding_is_as_accurate_as_reference_backend(tiny_qwen2_hd64: str) -> None:
    """With n-gram drafts verified in mixed steps (CUDA graphs for the plain decode
    steps), greedy tokens stay as faithful to the fp32 model as the torch backend's."""
    from tests.conftest import load_hf_model

    hf = load_hf_model(tiny_qwen2_hd64, torch.float32).to("cuda")
    prompts = [
        ([(5 * i + j) % 7 + 3 for j in range(3 + i % 4)] * 12)[: 20 + 7 * i] for i in range(16)
    ]
    scores, accepted = {}, 0
    for name, spec in (("flashinfer", 4), ("torch", 0)):
        engine = LLMEngine(
            EngineConfig(
                model=tiny_qwen2_hd64,
                device="cuda",
                dtype="bfloat16",
                attention_backend=name,
                max_model_len=512,
                num_kv_blocks=1024 // 16,
                max_num_batched_tokens=96,
                speculative_tokens=spec,
            )
        )
        scores[name] = _teacher_forced_agreement(engine, hf, prompts)
        accepted += engine.scheduler.spec_accepted
    assert accepted > 0
    assert scores["flashinfer"] >= scores["torch"] - 0.02, scores
