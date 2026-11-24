"""The engine must produce exactly what the Hugging Face reference produces.

Greedy decoding is compared token for token against a no-cache, no-batching
reference, across the code paths that could corrupt the KV cache: chunked
prefill, preemption with recompute, and prefix-cache hits.
"""

from __future__ import annotations

import random

import pytest
import torch

from minillm import EngineConfig, LLMEngine, SamplingParams
from minillm.block_manager import BlockManager
from minillm.executor.model_runner import ModelRunner
from minillm.scheduler import ScheduledSeq
from minillm.sequence import Sequence
from tests.conftest import TINY_MODEL_NAMES, hf_greedy

MODELS = TINY_MODEL_NAMES
GREEDY = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)


def _prompts(n: int, seed: int, lo: int = 3, hi: int = 30) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(3, 256) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def _engine(path, **overrides) -> LLMEngine:
    kwargs = dict(
        model=str(path),
        device="cpu",
        dtype="float32",
        block_size=4,
        max_model_len=64,
        num_kv_blocks=256,
        max_num_batched_tokens=64,
    )
    kwargs.update(overrides)
    return LLMEngine(EngineConfig(**kwargs))


@pytest.mark.parametrize("name", MODELS)
def test_prefill_logits_match_hf(name, tiny_model_dirs, hf_models):
    prompt = _prompts(1, seed=1, lo=25, hi=25)[0]
    runner = ModelRunner(
        EngineConfig(
            model=str(tiny_model_dirs[name]),
            device="cpu",
            dtype="float32",
            block_size=4,
            max_model_len=64,
            num_kv_blocks=32,
        )
    )
    bm = BlockManager(runner.num_kv_blocks, runner.block_size)
    seq = Sequence("r", prompt, GREEDY, 0.0)
    bm.allocate(seq, len(prompt))
    ours = runner.compute_logits([ScheduledSeq(seq, len(prompt))])
    with torch.no_grad():
        ref = hf_models[name](torch.tensor([prompt])).logits[0, -1]
    assert ours is not None
    torch.testing.assert_close(ours[0], ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", MODELS)
def test_batched_greedy_matches_hf(name, tiny_model_dirs, hf_models):
    prompts = _prompts(6, seed=2)
    engine = _engine(tiny_model_dirs[name])
    results = engine.generate(prompts, GREEDY)
    for prompt, res in zip(prompts, results, strict=True):
        assert res.output_token_ids == hf_greedy(hf_models[name], prompt, 20)
        assert res.finish_reason == "length"
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


@pytest.mark.parametrize("name", MODELS)
def test_chunked_prefill_matches_hf(name, tiny_model_dirs, hf_models):
    # A 7-token budget splits every prompt across steps and mixes chunks with decodes.
    prompts = _prompts(4, seed=3, lo=15, hi=30)
    engine = _engine(tiny_model_dirs[name], max_num_batched_tokens=7)
    results = engine.generate(prompts, GREEDY)
    for prompt, res in zip(prompts, results, strict=True):
        assert res.output_token_ids == hf_greedy(hf_models[name], prompt, 20)


@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("name", MODELS)
def test_preemption_recompute_matches_hf(name, prefix_caching, tiny_model_dirs, hf_models):
    # 18 blocks x 4 tokens cannot hold 5 sequences of up to 50 tokens: some get
    # preempted, lose their cache, and are recomputed later. With prefix caching
    # on, a resumed sequence re-attaches its own surviving blocks, including
    # blocks of tokens it generated before it was preempted.
    prompts = _prompts(5, seed=4, lo=20, hi=30)
    engine = _engine(tiny_model_dirs[name], num_kv_blocks=18, enable_prefix_caching=prefix_caching)
    results = engine.generate(prompts, GREEDY)
    assert engine.scheduler.num_preemptions > 0
    if prefix_caching:
        assert engine.block_manager.prefix_hit_tokens > 0
    for prompt, res in zip(prompts, results, strict=True):
        assert res.output_token_ids == hf_greedy(hf_models[name], prompt, 20)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


@pytest.mark.parametrize("name", MODELS)
def test_prefix_cache_hits_match_hf(name, tiny_model_dirs, hf_models):
    shared = _prompts(1, seed=5, lo=21, hi=21)[0]
    tails = _prompts(4, seed=6, lo=1, hi=8)
    prompts = [shared + t for t in tails]
    engine = _engine(tiny_model_dirs[name])
    first = engine.generate(prompts[:1], GREEDY)[0]
    rest = engine.generate(prompts[1:], GREEDY)
    assert first.num_cached_tokens == 0
    assert all(r.num_cached_tokens == 20 for r in rest)  # 5 full blocks of the shared prefix
    for prompt, res in zip(prompts, [first, *rest], strict=True):
        assert res.output_token_ids == hf_greedy(hf_models[name], prompt, 20)


def test_block_table_rows_are_reused_correctly(tiny_model_dirs, hf_models):
    """More requests than block-table rows: rows of finished sequences are handed
    to new ones, and a reused row must not leak the previous owner's pages."""
    prompts = _prompts(9, seed=7)
    engine = _engine(tiny_model_dirs["qwen2"], max_num_seqs=2)
    results = engine.generate(prompts, GREEDY)
    expected = [hf_greedy(hf_models["qwen2"], p, GREEDY.max_tokens) for p in prompts]
    assert [r.output_token_ids for r in results] == expected


@pytest.mark.parametrize("name", MODELS)
def test_speculative_decoding_matches_hf(name, tiny_model_dirs, hf_models):
    """N-gram drafts verified in one step: every accepted token, every correction,
    and the rolled-back positions of rejected drafts must leave the output exactly
    HF's. Repetitive prompts make drafts likely; a tight cache adds preemption."""
    rng = random.Random(11)
    prompts = []
    for _ in range(5):
        motif = [rng.randrange(3, 256) for _ in range(rng.randint(2, 5))]
        prompts.append((motif * 8)[: rng.randint(12, 30)])
    engine = _engine(
        tiny_model_dirs[name], speculative_tokens=4, num_kv_blocks=24, max_num_batched_tokens=16
    )
    results = engine.generate(prompts, GREEDY)
    for prompt, res in zip(prompts, results, strict=True):
        assert res.output_token_ids == hf_greedy(hf_models[name], prompt, 20)
    sched = engine.scheduler
    assert 0 < sched.spec_accepted < sched.spec_drafted  # both paths ran
    assert sched.num_preemptions > 0


def test_stale_nan_in_cache_does_not_leak(tiny_model_dirs, hf_models):
    # Poison every cache slot: anything read without being written first would
    # turn the output into NaN (argmax 0). Correct paged attention never reads it.
    prompts = _prompts(4, seed=9)
    engine = _engine(tiny_model_dirs["qwen2"], max_num_batched_tokens=9)
    engine.executor.kv_cache.fill_(float("nan"))
    for prompt, res in zip(prompts, engine.generate(prompts, GREEDY), strict=True):
        assert res.output_token_ids == hf_greedy(hf_models["qwen2"], prompt, 20)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_hf(tiny_model_dirs, hf_models):
    prompts = _prompts(4, seed=7)
    engine = _engine(tiny_model_dirs["qwen2"], device="mps", max_num_batched_tokens=9)
    for prompt, res in zip(prompts, engine.generate(prompts, GREEDY), strict=True):
        assert res.output_token_ids == hf_greedy(hf_models["qwen2"], prompt, 20)


def test_sampling_is_reproducible_with_seed(tiny_model_dirs):
    prompts = _prompts(3, seed=8)
    params = SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, top_k=50, ignore_eos=True)
    a = _engine(tiny_model_dirs["qwen2"], seed=123).generate(prompts, params)
    b = _engine(tiny_model_dirs["qwen2"], seed=123).generate(prompts, params)
    assert [r.output_token_ids for r in a] == [r.output_token_ids for r in b]
