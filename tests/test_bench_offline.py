"""The offline benchmark gives every engine byte-identical prompts.

It once drew token ids below each engine's own vocabulary size. vLLM reports the
tokenizer's size (151,643 for Qwen2.5) and minillm the model config's (151,936),
so the random stream diverged after the first prompt, and with varied lengths
vLLM ran a different workload.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from types import ModuleType

PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "bench_offline.py")


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_offline", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def args(**kw: object) -> argparse.Namespace:
    base = dict(seed=0, num_prompts=50, input_len=512, output_len=128, range_ratio=0.9)
    base.update(kw)
    return argparse.Namespace(**base)


def test_requests_do_not_depend_on_the_engine() -> None:
    bo = load()
    a = bo.make_requests(args())
    assert a == bo.make_requests(args())
    lo, hi = bo.TOKEN_IDS
    assert all(lo <= t < hi for p, _ in a for t in p)
    assert hi <= 32_000  # the smallest supported vocabulary (Llama 2, Mistral)
    lengths = {len(p) for p, _ in a}
    assert min(lengths) < 200 and max(lengths) > 800  # +/-90% around 512


def test_uniform_lengths() -> None:
    bo = load()
    reqs = bo.make_requests(args(range_ratio=0.0))
    assert {(len(p), n) for p, n in reqs} == {(512, 128)}
