"""Prometheus metrics for the engine and server."""

from __future__ import annotations

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
        self.num_running = Gauge("minillm_num_requests_running", "Running sequences.", registry=r)
        self.num_waiting = Gauge("minillm_num_requests_waiting", "Queued sequences.", registry=r)
        self.kv_cache_usage = Gauge(
            "minillm_kv_cache_usage_ratio", "Fraction of KV blocks in use.", registry=r
        )
