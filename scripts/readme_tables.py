"""Print the README's H200 result tables from the JSON files in results/.

    python scripts/readme_tables.py            # markdown for the Benchmarks section

Each results/h200-<model> directory holds one benchmark run (scripts/explorer/bench.sh);
results/h200-profile holds benchmarks/profile_step.py output per model.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import re
import sys
from typing import Any

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = [("Qwen2.5-0.5B", "h200-qwen2.5-0.5b"), ("Qwen2.5-7B", "h200-qwen2.5-7b")]


def load(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def tput(directory: str, engine: str, ratio: str) -> float | None:
    r = load(os.path.join(ROOT, directory, f"offline_{engine}_r{ratio}.json"))
    return None if r is None else float(r["output_throughput"])


def fmt(x: float | None, suffix: str = "") -> str:
    return "–" if x is None else f"{x:,.0f}{suffix}"


def offline_table() -> str:
    rows = [
        "| model | workload | minillm | minillm, unfused kernels | HF `generate` | vLLM 0.34 "
        "| minillm vs HF | minillm vs vLLM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in MODELS:
        for ratio, label in (("0", "uniform lengths"), ("0.9", "lengths vary ±90%")):
            m, unfused = tput(d, "minillm", ratio), tput(d, "minillm-unfused", ratio)
            hf, vllm = tput(d, "hf", ratio), tput(d, "vllm", ratio)
            vs_hf = f"{m / hf:.1f}×" if m and hf else "–"
            vs_vllm = f"{100 * m / vllm:.0f}%" if m and vllm else "–"
            rows.append(
                f"| {name} | {label} | {fmt(m)} | {fmt(unfused)} | {fmt(hf)} | {fmt(vllm)} "
                f"| {vs_hf} | {vs_vllm} |"
            )
    return "\n".join(rows)


def profile(d: str, variant: str = "") -> dict[str, Any] | None:
    return load(os.path.join(ROOT, "h200-profile", f"{d.removeprefix('h200-')}{variant}.json"))


def profile_table() -> str:
    rows = [
        "| model | batch | GPU, unfused | GPU | engine CPU | step, sync | step, async |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in MODELS:
        r, u = profile(d), profile(d, "-unfused")
        unfused = {row["batch"]: row["gpu_ms"] for row in (u or {}).get("rows", [])}
        for row in (r or {}).get("rows", []):
            before = unfused.get(row["batch"])
            rows.append(
                f"| {name} | {row['batch']} | {'–' if before is None else f'{before:.2f}'} "
                f"| {row['gpu_ms']:.2f} | {row['async_cpu_ms']:.2f} "
                f"| {row['sync_step_ms']:.2f} | {row['async_step_ms']:.2f} |"
            )
    return "\n".join(rows)


ATTENTION = re.compile(r"flashinfer::(BatchPrefill|BatchDecode|\w*MergeStates)")


def kernel_summary() -> str:
    """GPU time of one batch-1 decode step: GEMMs, attention and everything else
    (norms, residual adds, rotary, activation, casts, KV-cache writes, sampling)."""
    rows = [
        "| model | kernels | launches | GPU time | GEMMs | attention | everything else |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (name, d), (variant, label) in itertools.product(
        MODELS, (("-unfused", "unfused"), ("", "fused"))
    ):
        r = profile(d, variant)
        if not r or "kernels" not in r:
            continue
        ks = r["kernels"]["rows"]
        total = sum(k["us"] for k in ks)
        gemm = sum(k["us"] for k in ks if "nvjet" in k["kernel"] or "cublas" in k["kernel"].lower())
        attn = sum(k["us"] for k in ks if ATTENTION.search(k["kernel"]))
        other = total - gemm - attn
        rows.append(
            f"| {name} | {label} | {sum(k['launches'] for k in ks):,.0f} | {total / 1e3:.2f} ms "
            f"| {gemm / 1e3:.2f} ms | {attn / 1e3:.2f} ms "
            f"| {other / 1e3:.2f} ms ({100 * other / total:.0f}%) |"
        )
    return "\n".join(rows)


def serving_table(directory: str) -> str:
    runs: dict[tuple[float, str], dict[str, Any]] = {}
    for path in glob.glob(os.path.join(ROOT, directory, "serve_*_rate*.json")):
        r = load(path)
        assert r is not None
        runs[(float(r["request_rate"]), r["label"])] = r
    rows = [
        "| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) "
        "| TPOT p50 / p99 (ms) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (rate, label), r in sorted(runs.items()):
        rows.append(
            f"| {'burst' if rate == float('inf') else f'{rate:g}'} | {label} "
            f"| {r['request_throughput']:.1f} | {r['output_throughput']:,.0f} "
            f"| {r['median_ttft_ms']:.0f} / {r['p99_ttft_ms']:.0f} "
            f"| {r['median_tpot_ms']:.1f} / {r['p99_tpot_ms']:.1f} |"
        )
    return "\n".join(rows)


def main() -> None:
    out = [offline_table(), profile_table(), kernel_summary()]
    for name, d in MODELS:
        if glob.glob(os.path.join(ROOT, d, "serve_*_rate*.json")):
            out += [f"**{name}, online** (500 requests per point)", serving_table(d)]
    sys.stdout.write("\n\n".join(out) + "\n")


if __name__ == "__main__":
    main()
