"""The load generator sends as many requests at once as it is asked to.

It once divided the limit by the number of worker processes twice: with
``--workers 4`` and no ``--max-concurrency``, a "burst" of 500 requests ran as
a closed loop of 128 in flight.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "bench_serving.py")


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_serving", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_without_a_limit_every_request_is_in_flight_at_once() -> None:
    bs = load()
    assert bs.worker_concurrency(None, workers=4, worker_prompts=125) == 125
    assert bs.worker_concurrency(None, workers=1, worker_prompts=500) == 500


def test_a_limit_is_a_total_over_workers() -> None:
    bs = load()
    assert bs.worker_concurrency(64, workers=4, worker_prompts=125) == 16
    assert bs.worker_concurrency(5, workers=4, worker_prompts=125) == 2
    assert bs.worker_concurrency(8, workers=1, worker_prompts=125) == 8


def test_peak_concurrency() -> None:
    bs = load()
    rec = bs.RequestRecord
    records = [
        rec(prompt_len=1, sent_at=10.0, done_at=13.0),
        rec(prompt_len=1, sent_at=11.0, done_at=12.0),
        rec(prompt_len=1, sent_at=11.5, done_at=14.0),
        rec(prompt_len=1, sent_at=13.5, done_at=15.0),  # starts after two have finished
        rec(prompt_len=1),  # never sent (times stay 0.0)
    ]
    assert bs.peak_concurrency(records) == 3
