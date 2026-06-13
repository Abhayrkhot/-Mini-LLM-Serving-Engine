"""
Mini LLM serving engine simulation.

This module provides an `InferenceEngine` class that implements a
concurrency‑limited decoder.  Each request is processed
asynchronously: tokens are emitted one at a time with an artificial
delay to simulate model inference.  A semaphore enforces that no more
than `max_active` decode tasks run concurrently.  When all tokens
have been generated, a summary JSON string is yielded with per‑request
timing information.

Example usage:

```python
import asyncio
from server import InferenceEngine

async def main():
    engine = InferenceEngine(max_active=2, decode_delay=0.01)
    async for chunk in engine.infer("hello", 3):
        print(chunk)

asyncio.run(main())
```

Running this script directly will start a simple demonstration
consuming multiple requests concurrently.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, List


@dataclass
class RequestRecord:
    """Record of a single inference request for metrics purposes."""
    request_id: str
    prompt: str
    num_tokens: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class InferenceEngine:
    """Concurrency‑limited inference engine.

    Parameters
    ----------
    max_active : int
        Maximum number of concurrent decode tasks.
    decode_delay : float
        Artificial delay between tokens in seconds.
    """

    def __init__(self, max_active: int = 2, decode_delay: float = 0.05) -> None:
        if max_active <= 0:
            raise ValueError("max_active must be positive")
        if decode_delay <= 0:
            raise ValueError("decode_delay must be positive")
        self._sem = asyncio.Semaphore(max_active)
        self.decode_delay = decode_delay
        self._metrics: List[RequestRecord] = []

    @property
    def metrics(self) -> List[RequestRecord]:
        """Return a list of completed request records."""
        return list(self._metrics)

    async def infer(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        """Asynchronously yield tokens for a single request.

        This coroutine acquires the semaphore before decoding to enforce
        the concurrency limit.  It releases the semaphore when done.

        Parameters
        ----------
        prompt : str
            Input prompt used as a prefix for generated tokens.
        max_tokens : int
            Number of tokens to generate.  Must be positive.

        Yields
        ------
        str
            Generated token strings followed by a final JSON summary.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        # Generate a unique ID
        req_id = uuid.uuid4().hex
        async with self._sem:
            start = time.monotonic()
            # Emit tokens one by one
            for i in range(max_tokens):
                await asyncio.sleep(self.decode_delay)
                yield f"{prompt}_token_{i}"
            end = time.monotonic()
            record = RequestRecord(
                request_id=req_id,
                prompt=prompt,
                num_tokens=max_tokens,
                start_time=start,
                end_time=end,
            )
            self._metrics.append(record)
            # Emit final JSON summary
            yield json.dumps({**asdict(record), "duration": record.duration})


async def _demo() -> None:
    """Demonstration CLI.

    Generates a few concurrent requests and prints the streamed output.
    """
    engine = InferenceEngine(max_active=2, decode_delay=0.05)

    async def consume(prompt: str, max_tokens: int) -> None:
        async for chunk in engine.infer(prompt, max_tokens):
            print(f"[{prompt}] {chunk}")

    # Define some sample requests
    tasks = [
        asyncio.create_task(consume("A", 3)),
        asyncio.create_task(consume("B", 2)),
        asyncio.create_task(consume("C", 4)),
    ]
    await asyncio.gather(*tasks)
    print("\nMetrics:")
    for rec in engine.metrics:
        print(asdict(rec))


if __name__ == "__main__":
    asyncio.run(_demo())