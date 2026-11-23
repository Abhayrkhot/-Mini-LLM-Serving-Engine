# Results: h200-qwen2.5-7b

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 6,058 | 30,290 | 2.64x | 77% |
| uniform lengths | minillm-sync | 5,755 | 28,775 | 2.51x | 73% |
| uniform lengths | hf | 2,293 | 11,466 | 1.00x | 29% |
| uniform lengths | vllm | 7,904 | 39,521 | 3.45x | 100% |
| lengths ±90% | minillm | 4,630 | 24,201 | 8.67x | 64% |
| lengths ±90% | minillm-sync | 4,370 | 22,844 | 8.18x | 60% |
| lengths ±90% | hf | 534 | 2,792 | 1.00x | 7% |
| lengths ±90% | vllm | 7,242 | 37,346 | 13.56x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | minillm | 3.50 | 449 | 29 / 54 | 6.6 / 7.3 | 3.50 | 0 |
| 4 | minillm-sync | 3.50 | 448 | 25 / 51 | 7.1 / 8.2 | 3.50 | 0 |
| 4 | vllm | 3.51 | 449 | 23 / 47 | 5.0 / 5.3 | 3.51 | 0 |
| 8 | minillm | 6.97 | 892 | 31 / 51 | 7.2 / 8.1 | 6.97 | 0 |
| 8 | minillm-sync | 6.96 | 891 | 26 / 49 | 7.8 / 8.8 | 6.96 | 0 |
| 8 | vllm | 6.98 | 894 | 23 / 52 | 5.2 / 5.7 | 6.98 | 0 |
| 16 | minillm | 14.56 | 1,863 | 35 / 63 | 8.6 / 10.2 | 14.56 | 0 |
| 16 | minillm-sync | 14.54 | 1,861 | 29 / 65 | 9.5 / 11.5 | 14.54 | 0 |
| 16 | vllm | 14.64 | 1,873 | 25 / 56 | 5.8 / 6.6 | 14.64 | 0 |
| 32 | minillm | 28.32 | 3,625 | 52 / 154 | 13.3 / 17.3 | 28.32 | 0 |
| 32 | minillm-sync | 28.24 | 3,614 | 51 / 154 | 19.1 / 22.8 | 28.24 | 0 |
| 32 | vllm | 28.67 | 3,670 | 30 / 65 | 7.2 / 8.6 | 28.67 | 0 |
| 64 | minillm | 41.43 | 5,303 | 247 / 501 | 21.5 / 23.3 | 40.35 | 0 |
| 64 | minillm-sync | 38.86 | 4,974 | 249 / 433 | 23.3 / 25.4 | 38.86 | 0 |
| 64 | vllm | 53.59 | 6,860 | 106 / 291 | 16.5 / 17.7 | 53.59 | 0 |
| inf | minillm | 42.83 | 5,482 | 696 / 1774 | 17.6 / 21.8 | 6.94 | 0 |
| inf | minillm-sync | 39.59 | 5,068 | 465 / 1790 | 21.5 / 25.0 | 26.68 | 0 |
| inf | vllm | 54.68 | 6,999 | 590 / 1433 | 13.3 / 17.8 | 24.82 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-sync | 1024 | 21 / 37 | 1,992 |
| minillm | 1024 | 26 / 49 | 1,994 |
| vllm | 1024 | 18 / 36 | 2,005 |
