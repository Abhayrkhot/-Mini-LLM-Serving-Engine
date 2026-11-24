# Results: h200-qwen2.5-7b

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 7,520 | 37,599 | 3.28x | 95% |
| uniform lengths | minillm-unfused | 6,021 | 30,103 | 2.63x | 76% |
| uniform lengths | hf | 2,293 | 11,466 | 1.00x | 29% |
| uniform lengths | vllm | 7,885 | 39,423 | 3.44x | 100% |
| lengths ±90% | minillm | 5,567 | 29,100 | 10.42x | 77% |
| lengths ±90% | minillm-unfused | 4,609 | 24,091 | 8.63x | 64% |
| lengths ±90% | hf | 534 | 2,792 | 1.00x | 7% |
| lengths ±90% | vllm | 7,239 | 37,331 | 13.55x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | minillm | 3.51 | 449 | 20 / 75 | 4.8 / 5.2 | 3.51 | 0 |
| 4 | minillm-unfused | 3.50 | 449 | 29 / 49 | 6.6 / 7.3 | 3.50 | 0 |
| 4 | vllm | 3.51 | 449 | 22 / 37 | 5.0 / 5.3 | 3.51 | 0 |
| 8 | minillm | 6.99 | 894 | 21 / 34 | 5.0 / 5.5 | 6.99 | 0 |
| 8 | minillm-unfused | 6.97 | 892 | 31 / 49 | 7.2 / 8.1 | 6.97 | 0 |
| 8 | vllm | 6.98 | 894 | 23 / 36 | 5.2 / 5.6 | 6.98 | 0 |
| 16 | minillm | 14.65 | 1,875 | 23 / 43 | 5.6 / 6.6 | 14.65 | 0 |
| 16 | minillm-unfused | 14.55 | 1,863 | 35 / 64 | 8.6 / 10.3 | 14.55 | 0 |
| 16 | vllm | 14.64 | 1,874 | 24 / 58 | 5.7 / 6.6 | 14.64 | 0 |
| 32 | minillm | 28.72 | 3,676 | 29 / 76 | 7.5 / 10.4 | 28.72 | 0 |
| 32 | minillm-unfused | 28.31 | 3,624 | 53 / 148 | 13.7 / 18.2 | 28.31 | 0 |
| 32 | vllm | 28.68 | 3,671 | 29 / 65 | 7.2 / 8.5 | 28.68 | 0 |
| 64 | minillm | 49.03 | 6,276 | 127 / 270 | 18.6 / 20.1 | 49.03 | 0 |
| 64 | minillm-unfused | 41.37 | 5,296 | 245 / 449 | 21.6 / 23.4 | 41.37 | 0 |
| 64 | vllm | 53.95 | 6,906 | 91 / 200 | 16.6 / 17.7 | 53.95 | 0 |
| inf | minillm | 54.83 | 7,019 | 726 / 1446 | 13.1 / 18.0 | 15.13 | 0 |
| inf | minillm-unfused | 42.99 | 5,503 | 696 / 1777 | 17.6 / 21.8 | 8.25 | 0 |
| inf | vllm | 54.98 | 7,038 | 676 / 1452 | 13.2 / 17.1 | 17.48 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-unfused | 1024 | 26 / 40 | 1,994 |
| minillm | 1024 | 17 / 29 | 2,007 |
| vllm | 1024 | 18 / 22 | 2,005 |
