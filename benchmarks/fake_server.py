"""Serve the full HTTP stack over the fake executor: no GPU, no weights.

Step latency follows a simple cost model (fixed launch cost + per-token compute
+ per-context-token memory reads), so queueing, batching, chunked prefill and
preemption behave realistically enough to exercise the server and the load
generator end to end. Useful for CI and for comparing scheduler settings.

    python benchmarks/fake_server.py --port 8001 &
    python benchmarks/bench_serving.py --base-url http://localhost:8001 --vocab-size 259
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from minillm import AsyncLLMEngine, EngineCoreClient, EngineSpec
from minillm.async_engine import EngineClient
from minillm.entrypoints.openai_server import build_app
from minillm.executor.fake import ByteTokenizer, build_fake_engine, linear_cost
from minillm.metrics import EngineMetrics


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--num-kv-blocks", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--no-chunked-prefill", dest="chunked", action="store_false")
    p.add_argument("--no-prefix-caching", dest="prefix", action="store_false")
    p.add_argument("--base-ms", type=float, default=4.0, help="fixed cost per step")
    p.add_argument("--per-token-us", type=float, default=20.0, help="cost per computed token")
    p.add_argument("--no-engine-process", dest="engine_process", action="store_false")
    p.add_argument("--no-async-scheduling", dest="async_scheduling", action="store_false")
    args = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tokenizer = ByteTokenizer()
    executor = dict(
        num_kv_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        cost_model=linear_cost(base_s=args.base_ms / 1000, per_token_s=args.per_token_us / 1e6),
    )
    config = dict(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.chunked,
        enable_prefix_caching=args.prefix,
        async_scheduling=args.async_scheduling,
    )
    metrics = EngineMetrics()
    engine: EngineClient
    if args.engine_process:
        client = EngineCoreClient(
            EngineSpec(build_fake_engine, dict(config=config, executor=executor)), metrics=metrics
        )
        client.launch()
        engine = client
    else:
        llm = build_fake_engine(config=config, executor=executor)
        llm.metrics = metrics
        engine = AsyncLLMEngine(llm)
    app = build_app(engine, tokenizer, model_name="fake", metrics=metrics)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
