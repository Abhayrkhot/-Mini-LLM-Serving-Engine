"""Turn a results directory (bench_offline / bench_serving JSON) into markdown tables.

python scripts/report.py results/raw/20261122-NVIDIA_A100 > REPORT.md
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any


def load(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows.append(json.load(f) | {"_file": os.path.basename(path)})
    return rows


def offline_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    out = [
        "## Offline throughput",
        "",
        "| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ratio in sorted({r.get("range_ratio", 0.0) for r in rows}):
        group = {r["engine"]: r for r in rows if r.get("range_ratio", 0.0) == ratio}
        workload = "uniform lengths" if ratio == 0 else f"lengths ±{ratio:.0%}"
        for engine in ("minillm", "hf", "vllm"):
            r = group.get(engine)
            if r is None:
                continue
            tput = r["output_throughput"]
            vs_hf = f"{tput / group['hf']['output_throughput']:.2f}x" if "hf" in group else "-"
            vs_vllm = (
                f"{100 * tput / group['vllm']['output_throughput']:.0f}%"
                if "vllm" in group
                else "-"
            )
            out.append(
                f"| {workload} | {engine} | {tput:,.0f} | {r['total_token_throughput']:,.0f} "
                f"| {vs_hf} | {vs_vllm} |"
            )
    return "\n".join(out)


def serving_table(rows: list[dict[str, Any]]) -> str:
    rows = [r for r in rows if r.get("prefix_len", 0) == 0]
    if not rows:
        return ""
    out = [
        "## Online serving",
        "",
        "| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) "
        "| TPOT p50 / p99 (ms) | goodput (req/s) | failed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r["request_rate"], r["label"])):
        out.append(
            f"| {r['request_rate']:g} | {r['label']} | {r['request_throughput']:.2f} "
            f"| {r['output_throughput']:,.0f} "
            f"| {r['median_ttft_ms']:.0f} / {r['p99_ttft_ms']:.0f} "
            f"| {r['median_tpot_ms']:.1f} / {r['p99_tpot_ms']:.1f} "
            f"| {r['goodput']:.2f} | {r['failed']} |"
        )
    slo = rows[0]
    out += [
        "",
        f"Goodput counts requests meeting TTFT <= {slo['slo_ttft_ms']:.0f} ms "
        f"and TPOT <= {slo['slo_tpot_ms']:.0f} ms.",
    ]
    return "\n".join(out)


def prefix_table(rows: list[dict[str, Any]]) -> str:
    rows = [r for r in rows if r.get("prefix_len", 0) > 0]
    if not rows:
        return ""
    out = [
        "## Shared-prefix workload",
        "",
        "| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {r['label']} | {r['prefix_len']} | {r['median_ttft_ms']:.0f} / "
            f"{r['p99_ttft_ms']:.0f} | {r['output_throughput']:,.0f} |"
        )
    return "\n".join(out)


def main() -> None:
    root = sys.argv[1]
    offline = load(f"{root}/offline_*.json")
    serving = load(f"{root}/serve_*.json")
    header = [f"# Results: {os.path.basename(root.rstrip('/'))}", ""]
    if offline:
        r = offline[0]
        header.append(
            f"Model `{r['model']}` on {r.get('gpu', '?')}, {r['num_prompts']} prompts, "
            f"~{r['input_len']} in / ~{r['output_len']} out tokens."
        )
    sections = [offline_table(offline), serving_table(serving), prefix_table(serving)]
    print("\n\n".join(["\n".join(header), *[s for s in sections if s]]))


if __name__ == "__main__":
    main()
