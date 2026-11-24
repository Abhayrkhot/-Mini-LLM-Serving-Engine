"""Print the README's H200 result tables from the JSON files in results/.

    python scripts/readme_tables.py            # markdown for the Benchmarks section

results/h200-final-<model> holds the final run per model (scripts/explorer/bench.sh,
then the step profile, 4-bit accuracy and memory). The fused-vs-unfused kernel
breakdown comes from results/h200-profile, and results/h200-spec holds
benchmarks/bench_spec.py output.
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
# (name, final run, name of the model's runs in results/h200-profile)
MODELS = [
    ("Qwen2.5-0.5B", "h200-final-qwen2.5-0.5b", "qwen2.5-0.5b"),
    ("Qwen2.5-7B", "h200-final-qwen2.5-7b", "qwen2.5-7b"),
]
ENGINE_ORDER = ["minillm", "minillm-sync", "minillm-w4a16", "vllm"]
ENGINE_LABEL = {
    "minillm": "minillm",
    "minillm-sync": "minillm, sync scheduling",
    "minillm-w4a16": "minillm, 4-bit",
    "vllm": "vLLM",
}


def load(*parts: str) -> dict[str, Any] | None:
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def tput(directory: str, engine: str, ratio: str) -> float | None:
    r = load(directory, f"offline_{engine}_r{ratio}.json")
    return None if r is None else float(r["output_throughput"])


def fmt(x: float | None, digits: int = 0) -> str:
    return "–" if x is None else f"{x:,.{digits}f}"


def ratio(a: float | None, b: float | None, pct: bool = True) -> str:
    if not a or not b:
        return "–"
    return f"{100 * a / b:.0f}%" if pct else f"{a / b:.1f}×"


def offline_table() -> str:
    rows = [
        "| model | workload | minillm | minillm, 4-bit | HF `generate` | vLLM 0.34 "
        "| minillm vs HF | minillm vs vLLM | 4-bit vs 16-bit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, final, _ in MODELS:
        for r, label in (("0", "uniform lengths"), ("0.9", "lengths vary ±90%")):
            m, w4 = tput(final, "minillm", r), tput(final, "minillm-w4a16", r)
            hf, vllm = tput(final, "hf", r), tput(final, "vllm", r)
            rows.append(
                f"| {name} | {label} | {fmt(m)} | {fmt(w4)} | {fmt(hf)} | {fmt(vllm)} "
                f"| {ratio(m, hf, pct=False)} | {ratio(m, vllm)} | {ratio(w4, m)} |"
            )
    return "\n".join(rows)


def profile_table() -> str:
    rows = [
        "| model | batch | GPU | GPU, 4-bit | engine CPU | step, sync | step, async "
        "| step, async, 4-bit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, final, _ in MODELS:
        r, q = load(final, "profile.json"), load(final, "profile-w4a16.json")
        w4 = {row["batch"]: row for row in (q or {}).get("rows", [])}
        for row in (r or {}).get("rows", []):
            b = w4.get(row["batch"], {})
            rows.append(
                f"| {name} | {row['batch']} | {row['gpu_ms']:.2f} | {fmt(b.get('gpu_ms'), 2)} "
                f"| {row['async_cpu_ms']:.2f} | {row['sync_step_ms']:.2f} "
                f"| {row['async_step_ms']:.2f} | {fmt(b.get('async_step_ms'), 2)} |"
            )
    return "\n".join(rows)


ATTENTION = re.compile(r"flashinfer::(BatchPrefill|BatchDecode|\w*MergeStates)")
# cuBLAS/nvjet for 16-bit weights; the W4A16 kernels (minillm/quant/w4a16.py) for 4-bit.
GEMM = re.compile(r"nvjet|cublas|gemm|^_(small|gemv|mm|dequant)_kernel", re.IGNORECASE)


def kernel_rows(name: str, label: str, r: dict[str, Any] | None) -> list[str]:
    if not r or "kernels" not in r:
        return []
    ks = r["kernels"]["rows"]
    total = sum(k["us"] for k in ks)
    gemm = sum(k["us"] for k in ks if GEMM.search(k["kernel"]))
    attn = sum(k["us"] for k in ks if ATTENTION.search(k["kernel"]))
    other = total - gemm - attn
    return [
        f"| {name} | {label} | {sum(k['launches'] for k in ks):,.0f} | {total / 1e3:.2f} ms "
        f"| {gemm / 1e3:.2f} ms | {attn / 1e3:.2f} ms "
        f"| {other / 1e3:.2f} ms ({100 * other / total:.0f}%) |"
    ]


def kernel_table_fusion() -> str:
    """Batch-1 decode step before and after kernel fusion (results/h200-profile)."""
    rows = [
        "| model | kernels | launches | GPU time | GEMMs | attention | everything else |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (name, _, profile), (variant, label) in itertools.product(
        MODELS, (("-unfused", "unfused"), ("", "fused"))
    ):
        rows += kernel_rows(name, label, load("h200-profile", f"{profile}{variant}.json"))
    return "\n".join(rows)


def kernel_table_weights() -> str:
    """Batch-1 decode step with 16-bit and 4-bit weights (final run)."""
    rows = [
        "| model | weights | launches | GPU time | GEMMs | attention | everything else |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (name, final, _), (variant, label) in itertools.product(
        MODELS, (("", "16-bit"), ("-w4a16", "4-bit"))
    ):
        rows += kernel_rows(name, label, load(final, f"profile{variant}.json"))
    return "\n".join(rows)


def serving_runs(directory: str) -> dict[tuple[float, int], dict[str, Any]]:
    runs = {}
    for path in glob.glob(os.path.join(ROOT, directory, "serve_*_rate*.json")):
        with open(path) as f:
            r = json.load(f)
        if r["label"] in ENGINE_ORDER:
            runs[(float(r["request_rate"]), ENGINE_ORDER.index(r["label"]))] = r
    return runs


def serving_table(directory: str) -> str:
    rows = [
        "| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) "
        "| TPOT p50 / p99 (ms) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (rate, _), r in sorted(serving_runs(directory).items()):
        rows.append(
            f"| {'burst' if rate == float('inf') else f'{rate:g}'} | {ENGINE_LABEL[r['label']]} "
            f"| {r['request_throughput']:.1f} | {r['output_throughput']:,.0f} "
            f"| {r['median_ttft_ms']:.0f} / {r['p99_ttft_ms']:.0f} "
            f"| {r['median_tpot_ms']:.1f} / {r['p99_tpot_ms']:.1f} |"
        )
    return "\n".join(rows)


def prefix_table() -> str:
    rows = ["| model | engine | TTFT p50 / p99 (ms) |", "|---|---|---:|"]
    for (name, final, _), engine in itertools.product(MODELS, ENGINE_ORDER):
        r = load(final, f"serve_{engine}_prefix.json")
        if r:
            rows.append(
                f"| {name} | {ENGINE_LABEL[engine]} "
                f"| {r['median_ttft_ms']:.1f} / {r['p99_ttft_ms']:.1f} |"
            )
    return "\n".join(rows)


def spec_table() -> str:
    rows = [
        "| model | engine | concurrency | no drafts (tok/s) | 4-token drafts (tok/s) | speedup "
        "| drafts accepted | outputs identical |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(glob.glob(os.path.join(ROOT, "h200-spec", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        name = r["model"].split("/")[-1].removesuffix("-Instruct")
        by = {(row["draft"], row["concurrency"]): row for row in r["rows"]}
        for c in sorted({c for _, c in by}):
            base, spec = by.get((0, c)), by.get((4, c))
            if not base or not spec:
                continue
            acc = spec.get("acceptance")
            rows.append(
                f"| {name} | {'vLLM' if r['args']['engine'] == 'vllm' else 'minillm'} | {c} "
                f"| {base['output_throughput']:,.0f} | {spec['output_throughput']:,.0f} "
                f"| {spec['output_throughput'] / base['output_throughput']:.2f}× "
                f"| {'–' if acc is None else f'{100 * acc:.0f}%'} "
                f"| {100 * spec['same_as_no_draft']:.0f}% |"
            )
    return "\n".join(rows)


def quant_table() -> str:
    rows = [
        "| model | weights, 16-bit → 4-bit | KV-cache tokens the saving holds "
        "| WikiText-2 perplexity, 16-bit → 4-bit | next-token agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, final, _ in MODELS:
        e, mem = load(final, "eval_quant.json"), load(final, "memory.json")
        if not e:
            continue
        weights = kv = "–"
        if mem:
            w16, w4 = mem["weights_gib"]["minillm"], mem["weights_gib"]["minillm-w4a16"]
            weights = f"{w16:.2f} → {w4:.2f} GiB"
            kv = f"{(w16 - w4) * 2**30 / mem['kv_bytes_per_token'] / 1e3:,.0f}k"
        rows.append(
            f"| {name} | {weights} | {kv} | {e['bf16_perplexity']:.2f} → "
            f"{e['w4a16_perplexity']:.2f} (+{100 * e['perplexity_increase']:.0f}%) "
            f"| {100 * e['top1_agreement']:.1f}% |"
        )
    return "\n".join(rows)


def main() -> None:
    out = [
        "**Offline**",
        offline_table(),
        "**Step profile**",
        profile_table(),
        "**Kernels, fusion**",
        kernel_table_fusion(),
        "**Kernels, weights**",
        kernel_table_weights(),
    ]
    for name, final, _ in MODELS:
        if serving_runs(final):
            out += [f"**{name}, online** (500 requests per point)", serving_table(final)]
    out += [
        "**Prefix**",
        prefix_table(),
        "**Speculation**",
        spec_table(),
        "**4-bit**",
        quant_table(),
    ]
    sys.stdout.write("\n\n".join(out) + "\n")


if __name__ == "__main__":
    main()
