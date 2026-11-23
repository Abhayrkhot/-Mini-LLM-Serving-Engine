"""Online serving benchmark for any OpenAI-compatible /v1/completions server.

Sends streaming requests with Poisson arrivals (or all at once) and reports
throughput, time to first token (TTFT), time per output token (TPOT),
inter-token latency (ITL), end-to-end latency and goodput under latency SLOs.
Prompts are sent as token ids so minillm and vLLM see byte-identical input.

    python benchmarks/bench_serving.py --base-url http://localhost:8000 \
        --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 500 --request-rate 8 \
        --input-len 512 --output-len 128 --output results/raw/minillm_rate8.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class RequestRecord:
    prompt_len: int
    output_len: int = 0
    success: bool = False
    error: str = ""
    ttft: float = 0.0
    e2e: float = 0.0
    chunk_gaps: list[float] = field(default_factory=list)


def build_prompts(
    args: argparse.Namespace, vocab_size: int, seed: int | None = None
) -> list[list[int]]:
    """Random token-id prompts. Every run must use its own seed: engines with prefix
    caching would otherwise serve repeated prompts from cache and look faster."""
    rng = random.Random(args.seed if seed is None else seed)
    lo, hi = 100, vocab_size - 100  # stay clear of special tokens at both ends
    shared = [rng.randrange(lo, hi) for _ in range(args.prefix_len)]
    prompts = []
    for _ in range(args.num_prompts):
        n = args.input_len
        if args.range_ratio > 0:
            n = rng.randint(int(n * (1 - args.range_ratio)), int(n * (1 + args.range_ratio)))
        prompts.append(shared + [rng.randrange(lo, hi) for _ in range(max(1, n))])
    return prompts


async def one_request(
    client: httpx.AsyncClient, args: argparse.Namespace, prompt: list[int]
) -> RequestRecord:
    rec = RequestRecord(prompt_len=len(prompt))
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    last = start
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as r:
            if r.status_code != 200:
                rec.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    rec.output_len = data["usage"]["completion_tokens"]
                if not data.get("choices") or not data["choices"][0].get("text"):
                    continue
                now = time.perf_counter()
                if rec.ttft == 0.0:
                    rec.ttft = now - start
                else:
                    rec.chunk_gaps.append(now - last)
                last = now
        rec.e2e = time.perf_counter() - start
        rec.success = rec.ttft > 0
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        rec.error = repr(exc)
    return rec


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(
    records: list[RequestRecord], duration: float, args: argparse.Namespace
) -> dict[str, Any]:
    ok = [r for r in records if r.success]
    ttft = [r.ttft for r in ok]
    tpot = [(r.e2e - r.ttft) / (r.output_len - 1) for r in ok if r.output_len > 1]
    itl = [g for r in ok for g in r.chunk_gaps]
    e2e = [r.e2e for r in ok]
    out_tokens = sum(r.output_len for r in ok)
    in_tokens = sum(r.prompt_len for r in ok)

    def good(r: RequestRecord) -> bool:
        t = (r.e2e - r.ttft) / (r.output_len - 1) if r.output_len > 1 else 0.0
        return r.ttft * 1000 <= args.slo_ttft_ms and t * 1000 <= args.slo_tpot_ms

    def stats(name: str, xs: list[float]) -> dict[str, float]:
        ms = [x * 1000 for x in xs]
        return {
            f"mean_{name}_ms": statistics.fmean(ms) if ms else float("nan"),
            f"median_{name}_ms": pct(ms, 50),
            f"p90_{name}_ms": pct(ms, 90),
            f"p99_{name}_ms": pct(ms, 99),
        }

    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "prefix_len": args.prefix_len,
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "goodput": sum(1 for r in ok if good(r)) / duration,
        **stats("ttft", ttft),
        **stats("tpot", tpot),
        **stats("itl", itl),
        **stats("e2e", e2e),
        "errors": sorted({r.error for r in records if r.error})[:5],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    # Expire idle connections before uvicorn does (5 s), avoiding the keep-alive
    # race where a pooled connection is reused just as the server closes it.
    limits = httpx.Limits(
        max_connections=None, max_keepalive_connections=None, keepalive_expiry=2.0
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=None, limits=limits) as client:
        models = (await client.get("/v1/models")).json()
        if args.model is None:
            args.model = models["data"][0]["id"]
        vocab_size = args.vocab_size
        prompts = build_prompts(args, vocab_size)
        rng = random.Random(args.seed + 1)
        sem = asyncio.Semaphore(args.max_concurrency or len(prompts))

        if args.warmup:  # separate prompts, so the measured ones are not already cached
            warm = build_prompts(args, vocab_size, seed=args.seed + 1_000_003)[: args.warmup]
            await asyncio.gather(*(one_request(client, args, p) for p in warm))

        async def limited(p: list[int]) -> RequestRecord:
            async with sem:
                return await one_request(client, args, p)

        tasks = []
        start = time.perf_counter()
        for p in prompts:
            tasks.append(asyncio.create_task(limited(p)))
            if args.request_rate != float("inf"):
                await asyncio.sleep(rng.expovariate(args.request_rate))
        records = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
    result = summarize(records, duration, args)
    result["requests"] = (
        [asdict(r) | {"chunk_gaps": None} for r in records] if args.save_requests else None
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="defaults to the server's first model")
    p.add_argument("--label", default="", help="free-form tag stored in the result")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument(
        "--request-rate", type=float, default=float("inf"), help="req/s; inf = all at once"
    )
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument(
        "--range-ratio", type=float, default=0.0, help="vary input lengths by +/- this fraction"
    )
    p.add_argument(
        "--prefix-len", type=int, default=0, help="shared prefix tokens (prefix-cache workload)"
    )
    p.add_argument("--vocab-size", type=int, default=32_000, help="token ids are drawn below this")
    p.add_argument("--slo-ttft-ms", type=float, default=500.0)
    p.add_argument("--slo-tpot-ms", type=float, default=50.0)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-requests", action="store_true")
    p.add_argument("--output", default=None, help="write the JSON result here")
    args = p.parse_args()

    result = asyncio.run(run(args))
    summary = {k: v for k, v in result.items() if k != "requests"}
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k:<{width}}  {v:.2f}" if isinstance(v, float) else f"{k:<{width}}  {v}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
