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
| 4 | minillm | 3.50 | 449 | 25 / 47 | 7.1 / 8.0 | 3.50 | 0 |
| 4 | minillm-1proc | 3.50 | 449 | 25 / 50 | 7.2 / 8.2 | 3.50 | 0 |
| 4 | vllm | 3.51 | 449 | 22 / 38 | 5.0 / 5.3 | 3.51 | 0 |
| 8 | minillm | 6.96 | 891 | 25 / 46 | 7.7 / 8.7 | 6.96 | 0 |
| 8 | minillm-1proc | 6.96 | 891 | 25 / 51 | 8.0 / 9.1 | 6.96 | 0 |
| 8 | vllm | 6.99 | 894 | 22 / 37 | 5.1 / 5.7 | 6.99 | 0 |
| 16 | minillm | 14.54 | 1,861 | 29 / 64 | 9.5 / 11.5 | 14.54 | 0 |
| 16 | minillm-1proc | 14.54 | 1,861 | 29 / 63 | 10.2 / 12.2 | 14.54 | 0 |
| 16 | vllm | 14.64 | 1,874 | 24 / 58 | 5.6 / 6.5 | 14.64 | 0 |
| 32 | minillm | 28.23 | 3,614 | 48 / 359 | 18.6 / 23.0 | 28.23 | 0 |
| 32 | minillm-1proc | 27.99 | 3,583 | 71 / 156 | 28.6 / 31.9 | 27.99 | 0 |
| 32 | vllm | 28.70 | 3,674 | 27 / 64 | 7.0 / 8.3 | 28.70 | 0 |
| 64 | minillm | 39.14 | 5,010 | 237 / 411 | 23.0 / 25.2 | 39.14 | 0 |
| 64 | minillm-1proc | 34.93 | 4,471 | 249 / 730 | 25.5 / 29.8 | 33.25 | 0 |
| 64 | vllm | 54.27 | 6,946 | 84 / 191 | 16.4 / 17.5 | 54.27 | 0 |
| inf | minillm | 40.19 | 5,144 | 464 / 1809 | 20.9 / 23.7 | 27.25 | 0 |
| inf | minillm-1proc | 36.08 | 4,619 | 475 / 1827 | 23.5 / 26.9 | 23.82 | 0 |
| inf | vllm | 55.67 | 7,125 | 536 / 1414 | 13.5 / 17.9 | 24.83 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-1proc | 1024 | 22 / 42 | 1,992 |
| minillm | 1024 | 21 / 35 | 1,992 |
| vllm | 1024 | 17 / 21 | 2,007 |
