"""Speculative decoding on output that repeats its context.

N-gram drafts pay off when the output copies spans of the prompt: editing code,
quoting or summarising a document, filling a template. Random-token prompts
never repeat, so this benchmark asks an instruct model to repeat a passage
verbatim; the passages are chunks of this repository's own source and docs, so
no dataset is needed. It runs each draft length at several concurrency levels
and reports output throughput, time per output token, the draft acceptance
rate, and how many outputs match the run without drafts.

    python benchmarks/bench_spec.py --model Qwen/Qwen2.5-7B-Instruct --draft 0 4 --concurrency 1 8 32
    python benchmarks/bench_spec.py --engine vllm --model Qwen/Qwen2.5-7B-Instruct --draft 0 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from typing import Any

ROOT = os.path.join(os.path.dirname(__file__), "..")


def passages(tokenizer: Any, n: int, length: int) -> list[str]:
    files = [
        os.path.join(ROOT, "README.md"),
        *sorted(glob.glob(os.path.join(ROOT, "minillm/**/*.py"), recursive=True)),
    ]
    texts = []
    for path in files:
        with open(path) as f:
            texts.append(f.read())
    text = "\n\n".join(texts)
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = max(length, (len(ids) - length) // n)
    return [tokenizer.decode(ids[i : i + length]) for i in range(0, step * n, step)][:n]


def prompts(tokenizer: Any, args: argparse.Namespace) -> list[list[int]]:
    out = []
    for passage in passages(tokenizer, args.num_prompts, args.passage_tokens):
        messages = [
            {
                "role": "user",
                "content": "Repeat the following text exactly, with nothing before or after it.\n\n"
                + passage,
            }
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out.append(tokenizer.encode(text, add_special_tokens=False))
    return out


def run_minillm(args: argparse.Namespace, ps: list[list[int]]) -> list[dict[str, Any]]:
    from minillm import EngineConfig, LLMEngine, SamplingParams

    rows = []
    params = SamplingParams(max_tokens=args.passage_tokens + 32, temperature=0.0)
    reference: dict[int, list[list[int]]] = {}
    for draft in args.draft:
        engine = LLMEngine(
            EngineConfig(
                model=args.model,
                max_num_seqs=max(args.concurrency),
                max_num_batched_tokens=8192,
                speculative_tokens=draft,
                **({"dtype": args.dtype} if args.dtype else {}),
                **({"attention_backend": args.attention_backend} if args.attention_backend else {}),
            )
        )
        engine.generate(ps[:4], params)  # warm-up
        for c in args.concurrency:
            engine.scheduler.max_num_seqs = c
            engine.block_manager.reset_prefix_cache()
            drafted0, accepted0 = engine.scheduler.spec_drafted, engine.scheduler.spec_accepted
            start = time.perf_counter()
            results = engine.generate(ps, params)
            wall = time.perf_counter() - start
            outputs = [r.output_token_ids for r in results]
            tokens = sum(len(o) for o in outputs)
            tpot = [
                (r.e2e_latency - r.time_to_first_token) / (len(r.output_token_ids) - 1)
                for r in results
                if len(r.output_token_ids) > 1
            ]
            drafted = engine.scheduler.spec_drafted - drafted0
            accepted = engine.scheduler.spec_accepted - accepted0
            if draft == args.draft[0]:
                reference[c] = outputs
            same = sum(a == b for a, b in zip(outputs, reference[c], strict=True))
            rows.append(
                {
                    "engine": "minillm",
                    "draft": draft,
                    "concurrency": c,
                    "output_tokens": tokens,
                    "output_throughput": tokens / wall,
                    "mean_tpot_ms": 1e3 * statistics.fmean(tpot),
                    "acceptance": accepted / drafted if drafted else None,
                    "tokens_per_step_gain": accepted / max(1, tokens - accepted),
                    "same_as_no_draft": same / len(outputs),
                }
            )
            print(json.dumps(rows[-1]))
        del engine
    return rows


def run_vllm(args: argparse.Namespace, ps: list[list[int]]) -> list[dict[str, Any]]:
    import gc

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    rows = []
    params = SamplingParams(max_tokens=args.passage_tokens + 32, temperature=0.0)
    reference: dict[int, list[list[int]]] = {}
    for draft in args.draft:
        for c in args.concurrency:
            kwargs: dict[str, Any] = {"max_num_seqs": c, "max_num_batched_tokens": 8192}
            if draft:
                kwargs["speculative_config"] = {
                    "method": "ngram",
                    "num_speculative_tokens": draft,
                    "prompt_lookup_max": 3,
                    "prompt_lookup_min": 1,
                }
            llm = LLM(args.model, **kwargs)
            llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps[:4]], params)
            start = time.perf_counter()
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps], params)
            wall = time.perf_counter() - start
            outputs = [list(o.outputs[0].token_ids) for o in outs]
            tokens = sum(len(o) for o in outputs)
            if draft == args.draft[0]:
                reference[c] = outputs
            same = sum(a == b for a, b in zip(outputs, reference[c], strict=True))
            rows.append(
                {
                    "engine": "vllm",
                    "draft": draft,
                    "concurrency": c,
                    "output_tokens": tokens,
                    "output_throughput": tokens / wall,
                    "same_as_no_draft": same / len(outputs),
                }
            )
            print(json.dumps(rows[-1]))
            del llm
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--engine", choices=["minillm", "vllm"], default="minillm")
    p.add_argument("--draft", type=int, nargs="+", default=[0, 4], help="draft lengths; 0 = off")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--passage-tokens", type=int, default=256)
    p.add_argument("--dtype", default=None, help="minillm only, e.g. float32")
    p.add_argument("--attention-backend", default=None, help="minillm only, e.g. torch")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ps = prompts(tokenizer, args)
    rows = run_minillm(args, ps) if args.engine == "minillm" else run_vllm(args, ps)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "args": vars(args), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
