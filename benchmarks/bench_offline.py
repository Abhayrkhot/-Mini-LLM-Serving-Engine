"""Offline throughput: minillm vs Hugging Face generate vs vLLM on identical inputs.

All engines get the same random token-id prompts, greedy decoding and the same
per-request output lengths (EOS ignored), so the comparison is pure engine speed.

With --range-ratio 0 every request is identical in shape, which is static
batching's best case. With --range-ratio > 0 prompt and output lengths vary
uniformly by +/- that fraction, as in real traffic: static batching must pad
each batch to its longest prompt and keep decoding until its longest output
finishes. Throughput counts only the tokens each request asked for.

    python benchmarks/bench_offline.py --engine minillm --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-prompts 256 --input-len 512 --output-len 128 --output results/raw/offline_minillm.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import time
from typing import Any

import torch


def _length(rng: random.Random, mean: int, ratio: float) -> int:
    return max(1, rng.randint(int(mean * (1 - ratio)), int(mean * (1 + ratio))))


# Token ids are drawn from one fixed range that every supported vocabulary covers, so
# every engine gets byte-identical prompts. (Drawing below each engine's own vocabulary
# size made the random stream, and with it the prompt lengths, differ between engines.)
TOKEN_IDS = (100, 31_900)


def make_requests(args: argparse.Namespace) -> list[tuple[list[int], int]]:
    """(prompt token ids, output length) pairs, identical for every engine."""
    rng = random.Random(args.seed)
    lo, hi = TOKEN_IDS
    requests = []
    for _ in range(args.num_prompts):
        n_in = _length(rng, args.input_len, args.range_ratio)
        n_out = _length(rng, args.output_len, args.range_ratio)
        requests.append(([rng.randrange(lo, hi) for _ in range(n_in)], n_out))
    return requests


def _check_vocab(vocab_size: int) -> None:
    if vocab_size < TOKEN_IDS[1]:
        raise SystemExit(f"vocabulary of {vocab_size} is smaller than the prompt token ids")


WARMUP_REQUESTS = (
    32  # full-length prompts, a few tokens each: exercises real prefill and decode shapes
)
WARMUP_TOKENS = 8


def _max_len(args: argparse.Namespace) -> int:
    return int((args.input_len + args.output_len) * (1 + args.range_ratio)) + 2


def run_minillm(args: argparse.Namespace) -> tuple[float, int, dict[str, Any]]:
    from minillm import EngineConfig, LLMEngine, SamplingParams

    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            attention_backend=args.attention_backend,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_gb=args.kv_cache_memory_gb,
            max_model_len=_max_len(args),
            enforce_eager=args.enforce_eager,
            quantization=args.quantization,
            fused_kernels=args.fused_kernels,
            speculative_tokens=args.speculative_tokens,
            async_scheduling=args.async_scheduling,
        )
    )
    _check_vocab(engine.executor.vocab_size)
    requests = make_requests(args)
    prompts = [p for p, _ in requests]
    params = [SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True) for _, n in requests]
    warm = SamplingParams(max_tokens=WARMUP_TOKENS, temperature=0.0, ignore_eos=True)
    engine.generate(prompts[:WARMUP_REQUESTS], warm)
    engine.block_manager.reset_prefix_cache()  # the timed run must not hit warm-up blocks
    _sync()
    start = time.perf_counter()
    results = engine.generate(prompts, params)
    _sync()
    elapsed = time.perf_counter() - start
    out = sum(len(r.output_token_ids) for r in results)
    runner = engine.executor
    extra = {
        "preemptions": engine.scheduler.num_preemptions,
        "kv_blocks": engine.block_manager.num_blocks,
        "block_size": runner.block_size,
        "attention_backend": getattr(runner, "backend", None),
        "cuda_graphs": getattr(runner, "graphs", None) is not None,
        "dtype": str(getattr(runner, "dtype", "")),
        "device": str(getattr(runner, "device", "")),
    }
    return elapsed, out, extra


def run_hf(args: argparse.Namespace) -> tuple[float, int, dict[str, Any]]:
    """Static batching with transformers' generate (the usual naive baseline)."""
    from transformers import AutoModelForCausalLM, GenerationConfig

    from minillm.executor.model_runner import resolve_device, resolve_dtype

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        # cuDNN attention, one of the kernels PyTorch's SDPA picks from, builds a plan the
        # first time it sees a shape: about 4.6 s for a 64 x 512-token batch of Qwen2.5-0.5B
        # on an H200, against 1.4 s to generate it. Static batching meets a new shape with
        # every batch width, so leave cuDNN out; the other SDPA kernels are as fast once warm.
        torch.backends.cuda.enable_cudnn_sdp(False)
    import transformers

    # transformers 5 renamed torch_dtype to dtype; older releases ignore the new name.
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.model, **{key: dtype}).to(device).eval()
    _check_vocab(model.config.vocab_size)
    requests = make_requests(args)
    bs = args.hf_batch_size

    def generate(batch: list[tuple[list[int], int]]) -> int:
        width = max(len(p) for p, _ in batch)
        # Left-pad to the longest prompt; the attention mask hides the padding.
        ids = torch.tensor([[0] * (width - len(p)) + p for p, _ in batch], device=device)
        mask = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p, _ in batch], device=device
        )
        n = max(n for _, n in batch)  # the whole batch decodes until its longest request is done
        with torch.inference_mode():
            model.generate(
                ids,
                attention_mask=mask,
                generation_config=GenerationConfig(
                    max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0
                ),
            )
        return sum(n for _, n in batch)  # only the tokens that were asked for count

    generate([(p, WARMUP_TOKENS) for p, _ in requests[: min(bs, WARMUP_REQUESTS)]])
    _sync()
    start = time.perf_counter()
    out, batch_s = 0, []
    for i in range(0, len(requests), bs):
        t = time.perf_counter()
        out += generate(requests[i : i + bs])
        _sync()
        batch_s.append(round(time.perf_counter() - t, 3))
    meta = {"batch_size": bs, "dtype": str(dtype), "batch_seconds": batch_s}
    if device.type == "cuda":
        meta["cudnn_attention"] = torch.backends.cuda.cudnn_sdp_enabled()
    return time.perf_counter() - start, out, meta


def run_vllm(args: argparse.Namespace) -> tuple[float, int, dict[str, Any]]:
    from vllm import LLM
    from vllm import SamplingParams as VSamplingParams

    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=_max_len(args),
        enforce_eager=args.vllm_enforce_eager,
    )
    _check_vocab(llm.get_tokenizer().vocab_size)
    requests = make_requests(args)
    prompts = [{"prompt_token_ids": p} for p, _ in requests]
    params = [VSamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True) for _, n in requests]
    warm = VSamplingParams(max_tokens=WARMUP_TOKENS, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:WARMUP_REQUESTS], warm, use_tqdm=False)
    llm.reset_prefix_cache()  # the timed run must not hit warm-up blocks
    start = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    return (
        elapsed,
        sum(len(o.outputs[0].token_ids) for o in outs),
        {"enforce_eager": args.vllm_enforce_eager},
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--engine", choices=["minillm", "hf", "vllm"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument(
        "--range-ratio", type=float, default=0.0, help="vary lengths by +/- this fraction"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--attention-backend", default="auto")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--kv-cache-memory-gb", type=float, default=4.0)
    p.add_argument("--hf-batch-size", type=int, default=32)
    p.add_argument("--enforce-eager", action="store_true", help="minillm: no CUDA graphs")
    p.add_argument(
        "--no-fused-kernels",
        dest="fused_kernels",
        action="store_false",
        help="minillm: PyTorch ops instead of FlashInfer's fused norm/rotary/SiLU kernels",
    )
    p.add_argument("--speculative-tokens", type=int, default=0, help="minillm: n-gram drafts")
    p.add_argument("--quantization", choices=["w4a16"], default=None, help="minillm: 4-bit")
    p.add_argument(
        "--no-async-scheduling",
        dest="async_scheduling",
        action="store_false",
        help="minillm: wait for each step's tokens before scheduling the next",
    )
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None, help="engine name to record (default: --engine)")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    logging.basicConfig(level="WARNING", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("minillm").setLevel("INFO")  # KV sizing, graph capture, periodic throughput

    runner = {"minillm": run_minillm, "hf": run_hf, "vllm": run_vllm}[args.engine]
    elapsed, out_tokens, extra = runner(args)
    in_tokens = sum(len(p) for p, _ in make_requests(args))
    result = {
        "engine": args.label or args.engine,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "range_ratio": args.range_ratio,
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "output_throughput": out_tokens / elapsed,
        "total_token_throughput": (in_tokens + out_tokens) / elapsed,
        "request_throughput": args.num_prompts / elapsed,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor(),
        "torch": torch.__version__,
        **extra,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
