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
| 8 | minillm | 7.01 | 898 | 14 / 31 | 2.9 / 3.4 | 7.01 | 0 |
| 8 | minillm-1proc | 7.01 | 898 | 14 / 36 | 3.1 / 3.9 | 7.01 | 0 |
| 8 | vllm | 7.03 | 900 | 12 / 33 | 1.3 / 1.6 | 7.03 | 0 |
| 16 | minillm | 13.95 | 1,785 | 14 / 23 | 3.2 / 3.8 | 13.95 | 0 |
| 16 | minillm-1proc | 13.95 | 1,785 | 14 / 53 | 3.4 / 4.3 | 13.95 | 0 |
| 16 | vllm | 14.01 | 1,794 | 12 / 41 | 1.4 / 1.7 | 14.01 | 0 |
| 32 | minillm | 29.16 | 3,733 | 15 / 28 | 4.0 / 5.2 | 29.16 | 0 |
| 32 | minillm-1proc | 29.14 | 3,730 | 16 / 28 | 4.9 / 5.9 | 29.14 | 0 |
| 32 | vllm | 29.48 | 3,773 | 13 / 35 | 1.5 / 2.1 | 29.48 | 0 |
| 64 | minillm | 56.82 | 7,273 | 17 / 73 | 5.8 / 7.4 | 56.82 | 0 |
| 64 | minillm-1proc | 56.62 | 7,247 | 22 / 41 | 9.4 / 10.1 | 56.62 | 0 |
| 64 | vllm | 58.25 | 7,456 | 14 / 36 | 2.0 / 2.7 | 58.25 | 0 |
| 128 | minillm | 95.19 | 12,184 | 31 / 53 | 9.4 / 10.1 | 95.19 | 0 |
| 128 | minillm-1proc | 79.86 | 10,223 | 37 / 57 | 11.6 / 11.8 | 79.86 | 0 |
| 128 | vllm | 115.93 | 14,839 | 29 / 57 | 4.7 / 6.3 | 115.93 | 0 |
| inf | minillm | 145.03 | 18,564 | 103 / 290 | 5.8 / 6.8 | 145.03 | 0 |
| inf | minillm-1proc | 103.28 | 13,220 | 90 / 305 | 8.8 / 9.5 | 103.28 | 0 |
| inf | vllm | 229.48 | 29,373 | 128 / 191 | 3.2 / 3.5 | 229.48 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-1proc | 1024 | 14 / 263 | 2,024 |
| minillm | 1024 | 14 / 24 | 2,023 |
| vllm | 1024 | 11 / 15 | 2,035 |
