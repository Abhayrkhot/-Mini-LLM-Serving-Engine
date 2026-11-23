# Results: h200-qwen2.5-0.5b

Model `Qwen/Qwen2.5-0.5B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 23,001 | 115,006 | 8.93x | 68% |
| uniform lengths | hf | 2,577 | 12,883 | 1.00x | 8% |
| uniform lengths | vllm | 33,699 | 168,493 | 13.08x | 100% |
| lengths ±90% | minillm | 13,407 | 68,644 | 11.46x | 69% |
| lengths ±90% | hf | 1,170 | 5,991 | 1.00x | 6% |
| lengths ±90% | vllm | 19,341 | 99,734 | 16.53x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | minillm | 7.74 | 991 | 15 / 27 | 3.2 / 3.9 | 7.74 | 0 |
| 8 | vllm | 7.73 | 989 | 14 / 55 | 1.4 / 1.7 | 7.73 | 0 |
| 16 | minillm | 16.02 | 2,050 | 16 / 28 | 3.8 / 4.4 | 16.02 | 0 |
| 16 | vllm | 16.15 | 2,067 | 15 / 30 | 1.5 / 1.8 | 16.15 | 0 |
| 32 | minillm | 28.30 | 3,622 | 17 / 28 | 4.7 / 5.5 | 28.30 | 0 |
| 32 | vllm | 28.81 | 3,688 | 17 / 36 | 1.8 / 2.7 | 28.81 | 0 |
| 64 | minillm | 53.86 | 6,895 | 34 / 310 | 11.5 / 14.7 | 53.86 | 0 |
| 64 | vllm | 57.94 | 7,417 | 25 / 52 | 3.4 / 5.8 | 57.94 | 0 |
| 128 | minillm | 66.08 | 8,458 | 70 / 361 | 23.3 / 27.2 | 66.08 | 0 |
| 128 | vllm | 88.33 | 11,306 | 38 / 88 | 5.3 / 6.3 | 88.33 | 0 |
| inf | minillm | 76.60 | 9,804 | 801 / 4380 | 22.6 / 28.1 | 22.52 | 0 |
| inf | vllm | 76.39 | 9,779 | 889 / 6184 | 13.8 / 20.9 | 27.20 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm | 1024 | 16 / 25 | 2,038 |
| vllm | 1024 | 14 / 25 | 2,052 |
