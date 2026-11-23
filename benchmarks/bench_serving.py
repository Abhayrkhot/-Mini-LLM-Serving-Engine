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
import math
import multiprocessing as mp
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
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
    sent_at: float = 0.0  # wall clock, comparable across worker processes
    done_at: float = 0.0
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
    rec.sent_at = time.time()
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
    rec.done_at = time.time()
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


def _client() -> httpx.AsyncClient:
    # Expire idle connections before uvicorn does (5 s), avoiding the keep-alive
    # race where a pooled connection is reused just as the server closes it.
    limits = httpx.Limits(
        max_connections=None, max_keepalive_connections=None, keepalive_expiry=2.0
    )
    return httpx.AsyncClient(base_url=_BASE_URL, timeout=None, limits=limits)


_BASE_URL = ""


async def _drive(
    args: argparse.Namespace, prompts: list[list[int]], rate: float, seed: int, start_at: float
) -> list[RequestRecord]:
    """Send ``prompts`` with Poisson arrivals at ``rate`` from ``start_at`` (wall clock)."""
    rng = random.Random(seed)
    per_worker = math.ceil((args.max_concurrency or len(prompts) or 1) / max(args.workers, 1))
    sem = asyncio.Semaphore(per_worker)
    async with _client() as client:

        async def limited(p: list[int]) -> RequestRecord:
            async with sem:
                return await one_request(client, args, p)

        await asyncio.sleep(max(0.0, start_at - time.time()))
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(limited(p)))
            if rate != float("inf"):
                await asyncio.sleep(rng.expovariate(rate))
        return list(await asyncio.gather(*tasks))


def _worker(
    args: argparse.Namespace, prompts: list[list[int]], rate: float, seed: int, start_at: float
) -> list[RequestRecord]:
    global _BASE_URL
    _BASE_URL = args.base_url
    return asyncio.run(_drive(args, prompts, rate, seed, start_at))


async def _prepare(args: argparse.Namespace) -> list[list[int]]:
    async with _client() as client:
        models = (await client.get("/v1/models")).json()
        if args.model is None:
            args.model = models["data"][0]["id"]
        if args.warmup:  # separate prompts, so the measured ones are not already cached
            warm = build_prompts(args, args.vocab_size, seed=args.seed + 1_000_003)[: args.warmup]
            await asyncio.gather(*(one_request(client, args, p) for p in warm))
    return build_prompts(args, args.vocab_size)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Warm up, then send the load from ``args.workers`` processes.

    One Python process tops out near 10k streamed tokens/s (it parses every SSE
    event), which would cap fast servers. Each worker sends a disjoint slice of
    the prompts as its own Poisson stream at rate/workers; together they form a
    Poisson stream at the full rate. All workers start at the same wall-clock
    instant, and duration runs from the first send to the last response.
    """
    global _BASE_URL
    _BASE_URL = args.base_url
    prompts = asyncio.run(_prepare(args))
    workers = max(1, min(args.workers, len(prompts)))
    rate = args.request_rate / workers
    slices = [prompts[i::workers] for i in range(workers)]
    start_at = time.time() + (2.0 if workers > 1 else 0.0)  # time for workers to spawn
    if workers == 1:
        records = asyncio.run(_drive(args, slices[0], rate, args.seed + 1, start_at))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [
                pool.submit(_worker, args, sl, rate, args.seed + 1 + i, start_at)
                for i, sl in enumerate(slices)
            ]
            records = [r for f in futures for r in f.result()]
    sent = [r.sent_at for r in records if r.sent_at]
    done = [r.done_at for r in records if r.done_at]
    duration = max(done) - min(sent) if sent and done else float("nan")
    result = summarize(records, duration, args)
    result["workers"] = workers
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
    p.add_argument(
        "--workers", type=int, default=1, help="client processes (one tops out near 10k tok/s)"
    )
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

    result = run(args)
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
