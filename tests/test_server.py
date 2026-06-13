import asyncio
import json
import time
import os
import sys

import pytest

# Add the parent directory to sys.path so that `server` can be imported when
# running tests from the project root.  Without this, Python will not
# find the module because ``03-mini-llm-serving-engine`` is not installed as a
# package.  This pattern is common in simple repositories.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import InferenceEngine


@pytest.mark.asyncio
async def test_infer_stream_returns_tokens_and_summary():
    """InferenceEngine should stream the requested number of tokens
    followed by a JSON summary line."""
    engine = InferenceEngine(max_active=2, decode_delay=0.01)
    # Collect streamed output
    chunks = []
    async for chunk in engine.infer("hello", 3):
        chunks.append(chunk)
    # We expect 3 token lines + 1 summary line
    assert len(chunks) == 4
    # First three are tokens
    assert chunks[0].startswith("hello_token_")
    assert chunks[1].startswith("hello_token_")
    assert chunks[2].startswith("hello_token_")
    # Last chunk should be JSON summary
    summary = json.loads(chunks[3])
    assert summary["num_tokens"] == 3
    # Duration should be at least the number of tokens times the decode delay
    assert summary["duration"] >= 3 * engine.decode_delay - 0.001


@pytest.mark.asyncio
async def test_concurrency_limit_enforced():
    """When concurrency limit is 1, two requests should run sequentially."""
    # Use a relatively large delay so the timing is measurable
    delay = 0.05
    engine = InferenceEngine(max_active=1, decode_delay=delay)

    async def consume(prompt: str, tokens: int) -> float:
        # Consume all chunks; record finish time
        async for _ in engine.infer(prompt, tokens):
            pass
        return time.monotonic()

    t0 = time.monotonic()
    # Launch two requests concurrently
    finish_times = await asyncio.gather(consume("A", 2), consume("B", 2))
    elapsed = max(finish_times) - t0
    # Each request takes roughly tokens * delay seconds. With a
    # concurrency limit of 1, the requests run sequentially, so total
    # time should be around 2 * tokens * delay. Allow small slack.
    assert elapsed >= 2 * 2 * delay - 0.02