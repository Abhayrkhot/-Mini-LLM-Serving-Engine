# Results: h200-qwen2.5-0.5b-eager

Model `Qwen/Qwen2.5-0.5B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 13,866 | 69,329 | - | - |
| lengths ±90% | minillm | 9,206 | 47,135 | - | - |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | minillm | 7.61 | 974 | 22 / 129 | 11.4 / 15.1 | 7.61 | 0 |
| 16 | minillm | 15.46 | 1,979 | 23 / 37 | 12.0 / 12.8 | 15.46 | 0 |
| 32 | minillm | 26.52 | 3,394 | 27 / 40 | 13.7 / 14.6 | 26.52 | 0 |
| 64 | minillm | 45.06 | 5,768 | 48 / 82 | 22.5 / 26.6 | 45.06 | 0 |
| 128 | minillm | 58.74 | 7,519 | 89 / 878 | 26.3 / 29.5 | 35.25 | 0 |
| inf | minillm | 63.61 | 8,142 | 1297 / 5337 | 24.6 / 31.7 | 0.13 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm | 1024 | 23 / 34 | 1,966 |
