# Results: h200-qwen2.5-7b

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 5,757 | 28,784 | 2.58x | 73% |
| uniform lengths | hf | 2,235 | 11,176 | 1.00x | 28% |
| uniform lengths | vllm | 7,886 | 39,429 | 3.53x | 100% |
| lengths ±90% | minillm | 4,368 | 22,833 | 8.31x | 60% |
| lengths ±90% | hf | 526 | 2,748 | 1.00x | 7% |
| lengths ±90% | vllm | 7,282 | 37,549 | 13.85x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | minillm | 3.87 | 496 | 25 / 43 | 7.2 / 8.4 | 3.87 | 0 |
| 4 | vllm | 3.88 | 497 | 22 / 30 | 4.9 / 5.3 | 3.88 | 0 |
| 8 | minillm | 8.03 | 1,027 | 26 / 54 | 8.0 / 9.0 | 8.03 | 0 |
| 8 | vllm | 8.06 | 1,032 | 22 / 41 | 5.2 / 5.6 | 8.06 | 0 |
| 16 | minillm | 14.24 | 1,822 | 30 / 66 | 9.8 / 11.2 | 14.24 | 0 |
| 16 | vllm | 14.39 | 1,841 | 24 / 42 | 5.6 / 6.2 | 14.39 | 0 |
| 32 | minillm | 28.37 | 3,631 | 58 / 141 | 22.9 / 33.2 | 28.37 | 0 |
| 32 | vllm | 30.01 | 3,841 | 30 / 62 | 7.1 / 8.4 | 30.01 | 0 |
| 64 | minillm | 34.31 | 4,392 | 461 / 3382 | 51.5 / 58.9 | 0.82 | 0 |
| 64 | vllm | 51.35 | 6,572 | 68 / 205 | 15.3 / 19.9 | 51.35 | 0 |
| inf | minillm | 37.60 | 4,812 | 3907 / 10289 | 46.0 / 48.3 | 1.20 | 0 |
| inf | vllm | 54.53 | 6,980 | 2955 / 7028 | 28.9 / 31.6 | 2.40 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm | 1024 | 21 / 35 | 2,001 |
| vllm | 1024 | 18 / 22 | 2,022 |
