# Results: h200-final-qwen2.5-7b

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 7,881 | 39,405 | 2.31x | 100% |
| uniform lengths | minillm-w4a16 | 6,892 | 34,459 | 2.02x | 87% |
| uniform lengths | hf | 3,417 | 17,085 | 1.00x | 43% |
| uniform lengths | vllm | 7,881 | 39,406 | 2.31x | 100% |
| lengths ±90% | minillm | 6,922 | 36,141 | 12.36x | 100% |
| lengths ±90% | minillm-w4a16 | 6,019 | 31,428 | 10.75x | 87% |
| lengths ±90% | hf | 560 | 2,925 | 1.00x | 8% |
| lengths ±90% | vllm | 6,951 | 36,292 | 12.41x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | minillm | 3.51 | 449 | 20 / 38 | 4.8 / 5.1 | 3.51 | 0 |
| 4 | minillm-sync | 3.51 | 449 | 18 / 36 | 5.3 / 5.7 | 3.51 | 0 |
| 4 | minillm-w4a16 | 3.51 | 450 | 23 / 43 | 3.7 / 4.3 | 3.51 | 0 |
| 4 | vllm | 3.51 | 449 | 22 / 75 | 5.0 / 5.3 | 3.51 | 0 |
| 8 | minillm | 6.99 | 894 | 21 / 35 | 5.0 / 5.5 | 6.99 | 0 |
| 8 | minillm-sync | 6.98 | 894 | 18 / 34 | 5.5 / 6.2 | 6.98 | 0 |
| 8 | minillm-w4a16 | 7.00 | 896 | 24 / 43 | 4.1 / 5.5 | 7.00 | 0 |
| 8 | vllm | 6.98 | 894 | 23 / 68 | 5.2 / 5.8 | 6.98 | 0 |
| 16 | minillm | 14.65 | 1,875 | 23 / 43 | 5.5 / 6.4 | 14.65 | 0 |
| 16 | minillm-sync | 14.63 | 1,872 | 20 / 42 | 6.3 / 7.4 | 14.63 | 0 |
| 16 | minillm-w4a16 | 14.71 | 1,882 | 28 / 54 | 6.1 / 8.0 | 14.71 | 0 |
| 16 | vllm | 14.64 | 1,874 | 24 / 90 | 5.7 / 6.6 | 14.64 | 0 |
| 32 | minillm | 28.72 | 3,676 | 26 / 68 | 6.9 / 8.6 | 28.72 | 0 |
| 32 | minillm-sync | 28.65 | 3,667 | 23 / 72 | 8.4 / 10.8 | 28.65 | 0 |
| 32 | minillm-w4a16 | 28.91 | 3,701 | 38 / 86 | 11.7 / 15.3 | 28.91 | 0 |
| 32 | vllm | 28.68 | 3,671 | 28 / 67 | 7.1 / 8.5 | 28.68 | 0 |
| 64 | minillm | 55.95 | 7,161 | 102 / 282 | 28.9 / 32.3 | 55.95 | 0 |
| 64 | minillm-sync | 54.76 | 7,010 | 151 / 435 | 30.7 / 33.1 | 54.76 | 0 |
| 64 | minillm-w4a16 | 50.94 | 6,520 | 198 / 906 | 34.8 / 36.4 | 27.91 | 0 |
| 64 | vllm | 55.91 | 7,157 | 108 / 219 | 27.8 / 31.5 | 55.91 | 0 |
| inf | minillm | 60.20 | 7,705 | 2966 / 6946 | 30.2 / 30.4 | 3.37 | 0 |
| inf | minillm-sync | 55.38 | 7,088 | 3502 / 7570 | 31.6 / 32.0 | 0.00 | 0 |
| inf | minillm-w4a16 | 51.37 | 6,576 | 3338 / 7886 | 35.3 / 35.8 | 1.44 | 0 |
| inf | vllm | 59.09 | 7,563 | 3038 / 7075 | 29.2 / 30.4 | 2.25 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-sync | 1024 | 13 / 24 | 2,004 |
| minillm-w4a16 | 1024 | 21 / 41 | 2,015 |
| minillm | 1024 | 16 / 27 | 2,007 |
| vllm | 1024 | 18 / 37 | 2,005 |
