# Results: h200-qwen2.5-0.5b

Model `Qwen/Qwen2.5-0.5B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 46,897 | 234,483 | 15.09x | 113% |
| uniform lengths | minillm-unfused | 31,525 | 157,625 | 10.14x | 76% |
| uniform lengths | hf | 3,108 | 15,542 | 1.00x | 8% |
| uniform lengths | vllm | 41,418 | 207,089 | 13.32x | 100% |
| lengths ±90% | minillm | 27,000 | 138,238 | 20.18x | 110% |
| lengths ±90% | minillm-unfused | 18,475 | 94,592 | 13.81x | 75% |
| lengths ±90% | hf | 1,338 | 6,849 | 1.00x | 5% |
| lengths ±90% | vllm | 24,487 | 126,269 | 18.30x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | minillm | 7.03 | 900 | 7 / 25 | 1.2 / 1.4 | 7.03 | 0 |
| 8 | minillm-unfused | 7.02 | 898 | 14 / 33 | 2.5 / 2.9 | 7.02 | 0 |
| 8 | vllm | 7.03 | 900 | 11 / 21 | 1.3 / 1.6 | 7.03 | 0 |
| 16 | minillm | 14.01 | 1,794 | 7 / 15 | 1.2 / 1.4 | 14.01 | 0 |
| 16 | minillm-unfused | 13.96 | 1,787 | 14 / 32 | 2.7 / 3.1 | 13.96 | 0 |
| 16 | vllm | 14.01 | 1,794 | 13 / 27 | 1.4 / 1.9 | 14.01 | 0 |
| 32 | minillm | 29.50 | 3,776 | 8 / 17 | 1.3 / 1.6 | 29.50 | 0 |
| 32 | minillm-unfused | 29.23 | 3,741 | 15 / 35 | 3.2 / 4.0 | 29.23 | 0 |
| 32 | vllm | 29.49 | 3,775 | 12 / 27 | 1.6 / 2.2 | 29.49 | 0 |
| 64 | minillm | 58.28 | 7,460 | 11 / 31 | 1.6 / 2.0 | 58.28 | 0 |
| 64 | minillm-unfused | 57.07 | 7,305 | 25 / 98 | 4.4 / 5.5 | 57.07 | 0 |
| 64 | vllm | 58.23 | 7,454 | 16 / 32 | 2.4 / 3.2 | 58.23 | 0 |
| 128 | minillm | 116.64 | 14,930 | 18 / 103 | 3.4 / 4.7 | 116.64 | 0 |
| 128 | minillm-unfused | 111.97 | 14,332 | 38 / 61 | 6.7 / 7.4 | 111.97 | 0 |
| 128 | vllm | 115.24 | 14,751 | 35 / 69 | 5.9 / 7.1 | 115.24 | 0 |
| inf | minillm | 282.12 | 36,111 | 82 / 152 | 2.7 / 3.5 | 282.12 | 0 |
| inf | minillm-unfused | 191.18 | 24,471 | 134 / 237 | 4.0 / 4.8 | 191.18 | 0 |
| inf | vllm | 187.09 | 23,948 | 150 / 203 | 4.2 / 4.7 | 187.09 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-unfused | 1024 | 14 / 30 | 2,026 |
| minillm | 1024 | 7 / 13 | 2,035 |
| vllm | 1024 | 11 / 18 | 2,035 |
