# Results: h200-qwen2.5-0.5b

Model `Qwen/Qwen2.5-0.5B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 31,253 | 156,266 | 10.05x | 80% |
| uniform lengths | minillm-sync | 23,691 | 118,455 | 7.62x | 61% |
| uniform lengths | hf | 3,108 | 15,542 | 1.00x | 8% |
| uniform lengths | vllm | 39,000 | 195,001 | 12.55x | 100% |
| lengths ±90% | minillm | 18,362 | 94,015 | 13.73x | 82% |
| lengths ±90% | minillm-sync | 14,576 | 74,629 | 10.90x | 65% |
| lengths ±90% | hf | 1,338 | 6,849 | 1.00x | 6% |
| lengths ±90% | vllm | 22,477 | 115,907 | 16.80x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | minillm | 7.02 | 898 | 14 / 33 | 2.5 / 2.9 | 7.02 | 0 |
| 8 | minillm-sync | 7.01 | 898 | 14 / 29 | 2.9 / 3.5 | 7.01 | 0 |
| 8 | vllm | 7.03 | 899 | 12 / 34 | 1.3 / 1.6 | 7.03 | 0 |
| 16 | minillm | 13.96 | 1,787 | 14 / 32 | 2.7 / 3.1 | 13.96 | 0 |
| 16 | minillm-sync | 13.95 | 1,785 | 14 / 24 | 3.2 / 3.8 | 13.95 | 0 |
| 16 | vllm | 14.01 | 1,793 | 12 / 23 | 1.4 / 1.8 | 14.01 | 0 |
| 32 | minillm | 29.22 | 3,740 | 16 / 36 | 3.2 / 3.9 | 29.22 | 0 |
| 32 | minillm-sync | 29.15 | 3,731 | 15 / 25 | 4.0 / 5.1 | 29.15 | 0 |
| 32 | vllm | 29.47 | 3,772 | 13 / 37 | 1.6 / 2.3 | 29.47 | 0 |
| 64 | minillm | 57.14 | 7,314 | 25 / 142 | 4.3 / 5.5 | 57.14 | 0 |
| 64 | minillm-sync | 56.82 | 7,273 | 18 / 92 | 5.9 / 8.2 | 56.82 | 0 |
| 64 | vllm | 58.17 | 7,445 | 16 / 42 | 2.3 / 3.4 | 58.17 | 0 |
| 128 | minillm | 111.28 | 14,244 | 37 / 57 | 7.2 / 7.6 | 111.28 | 0 |
| 128 | minillm-sync | 95.30 | 12,199 | 33 / 55 | 9.6 / 10.2 | 95.30 | 0 |
| 128 | vllm | 114.44 | 14,649 | 33 / 62 | 6.2 / 6.9 | 114.44 | 0 |
| inf | minillm | 184.66 | 23,636 | 136 / 309 | 4.1 / 4.7 | 184.66 | 0 |
| inf | minillm-sync | 142.99 | 18,302 | 75 / 278 | 6.2 / 6.7 | 142.99 | 0 |
| inf | vllm | 181.19 | 23,192 | 105 / 186 | 4.6 / 5.0 | 181.19 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-sync | 1024 | 14 / 35 | 2,024 |
| minillm | 1024 | 15 / 37 | 2,026 |
| vllm | 1024 | 11 / 16 | 2,034 |
