"""Prometheus metrics for the engine and server.

The engine never touches Prometheus directly. It accumulates an ``EngineStats``
snapshot of what happened since the last one (steps, tokens, latencies,
finished requests), and ``EngineMetrics.record`` applies a snapshot to a
registry. That lets the engine run in its own process: it ships snapshots to
the API process, which owns the registry served at /metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (
    0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)  # fmt: skip
_ITL_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
    0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
)  # fmt: skip
_TOKEN_BUCKETS = tuple(float(2**i) for i in range(15))


class FinishedRequest(NamedTuple):
    finish_reason: str
    prompt_tokens: int
    e2e_latency: float | None
    queue_time: float | None


@dataclass
class EngineStats:
    """Everything the engine did since the previous snapshot. Picklable."""

    step_durations: list[float] = field(default_factory=list)
    step_tokens: list[int] = field(default_factory=list)
    generation_tokens: int = 0
    preemptions: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    spec_draft_tokens: int = 0
    spec_accepted_tokens: int = 0
    time_to_first_token: list[float] = field(default_factory=list)
    inter_token_latency: list[float] = field(default_factory=list)
    finished: list[FinishedRequest] = field(default_factory=list)
    # Point-in-time values, taken when the snapshot is cut.
    num_running: int = 0
    num_waiting: int = 0
    kv_cache_usage: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not (self.step_durations or self.finished)


class EngineMetrics:
    """All series share the ``minillm_`` prefix. Pass a registry to isolate instances in tests."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.time_to_first_token = Histogram(
            "minillm_time_to_first_token_seconds",
            "Arrival to first generated token, queueing included.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.inter_token_latency = Histogram(
            "minillm_inter_token_latency_seconds",
            "Gap between consecutive generated tokens of one request.",
            buckets=_ITL_BUCKETS,
            registry=r,
        )
        self.e2e_latency = Histogram(
            "minillm_e2e_request_latency_seconds",
            "Arrival to finish.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.queue_time = Histogram(
            "minillm_request_queue_time_seconds",
            "Arrival to first scheduled.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.step_duration = Histogram(
            "minillm_step_duration_seconds",
            "Wall time of one engine step (model forward + sampling).",
            buckets=_ITL_BUCKETS,
            registry=r,
        )
        self.step_tokens = Histogram(
            "minillm_step_tokens",
            "Tokens computed per engine step.",
            buckets=_TOKEN_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "minillm_prompt_tokens", "Prompt tokens of finished requests.", registry=r
        )
        self.generation_tokens = Counter(
            "minillm_generation_tokens", "Generated tokens.", registry=r
        )
        self.requests_finished = Counter(
            "minillm_requests_finished", "Finished requests.", ["finish_reason"], registry=r
        )
        self.requests_rejected = Counter(
            "minillm_requests_rejected",
            "Requests turned away by the server.",
            ["reason"],
            registry=r,
        )
        self.preemptions = Counter(
            "minillm_preemptions", "Sequences preempted for lack of KV blocks.", registry=r
        )
        self.prefix_cache_queries = Counter(
            "minillm_prefix_cache_queries",
            "Prompt tokens looked up in the prefix cache.",
            registry=r,
        )
        self.prefix_cache_hits = Counter(
            "minillm_prefix_cache_hits", "Prompt tokens served from the prefix cache.", registry=r
        )
        self.spec_draft_tokens = Counter(
            "minillm_spec_draft_tokens", "Speculative tokens drafted.", registry=r
        )
        self.spec_accepted_tokens = Counter(
            "minillm_spec_accepted_tokens",
            "Speculative tokens the model agreed with.",
            registry=r,
        )
        self.num_running = Gauge("minillm_num_requests_running", "Running sequences.", registry=r)
        self.num_waiting = Gauge("minillm_num_requests_waiting", "Queued sequences.", registry=r)
        self.kv_cache_usage = Gauge(
            "minillm_kv_cache_usage_ratio", "Fraction of KV blocks in use.", registry=r
        )

    def record(self, stats: EngineStats) -> None:
        for d in stats.step_durations:
            self.step_duration.observe(d)
        for n in stats.step_tokens:
            self.step_tokens.observe(n)
        for t in stats.time_to_first_token:
            self.time_to_first_token.observe(t)
        for t in stats.inter_token_latency:
            self.inter_token_latency.observe(t)
        self.generation_tokens.inc(stats.generation_tokens)
        self.preemptions.inc(stats.preemptions)
        self.prefix_cache_queries.inc(stats.prefix_cache_queries)
        self.prefix_cache_hits.inc(stats.prefix_cache_hits)
        self.spec_draft_tokens.inc(stats.spec_draft_tokens)
        self.spec_accepted_tokens.inc(stats.spec_accepted_tokens)
        for f in stats.finished:
            self.requests_finished.labels(f.finish_reason).inc()
            self.prompt_tokens.inc(f.prompt_tokens)
            if f.e2e_latency is not None:
                self.e2e_latency.observe(f.e2e_latency)
            if f.queue_time is not None:
                self.queue_time.observe(f.queue_time)
        self.num_running.set(stats.num_running)
        self.num_waiting.set(stats.num_waiting)
        self.kv_cache_usage.set(stats.kv_cache_usage)
