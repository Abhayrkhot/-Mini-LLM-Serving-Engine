# Results: h200-final-qwen2.5-0.5b

Model `Qwen/Qwen2.5-0.5B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 56,996 | 284,980 | 9.69x | 139% |
| uniform lengths | minillm-w4a16 | 47,160 | 235,798 | 8.01x | 115% |
| uniform lengths | hf | 5,884 | 29,421 | 1.00x | 14% |
| uniform lengths | vllm | 40,860 | 204,302 | 6.94x | 100% |
| lengths ±90% | minillm | 31,082 | 162,287 | 15.54x | 136% |
| lengths ±90% | minillm-w4a16 | 22,961 | 119,888 | 11.48x | 101% |
| lengths ±90% | hf | 2,001 | 10,446 | 1.00x | 9% |
| lengths ±90% | vllm | 22,833 | 119,220 | 11.41x | 100% |

## Online serving

| req/s offered | engine | req/s done | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | goodput (req/s) | failed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | minillm | 7.03 | 900 | 8 / 32 | 1.2 / 1.5 | 7.03 | 0 |
| 8 | minillm-sync | 7.03 | 899 | 8 / 24 | 1.5 / 1.8 | 7.03 | 0 |
| 8 | minillm-w4a16 | 7.02 | 899 | 14 / 34 | 1.9 / 2.3 | 7.02 | 0 |
| 8 | vllm | 7.03 | 900 | 11 / 39 | 1.4 / 1.7 | 7.03 | 0 |
| 16 | minillm | 14.01 | 1,794 | 9 / 17 | 1.3 / 1.5 | 14.01 | 0 |
| 16 | minillm-sync | 14.00 | 1,792 | 8 / 14 | 1.6 / 1.9 | 14.00 | 0 |
| 16 | minillm-w4a16 | 13.98 | 1,790 | 14 / 32 | 2.1 / 2.6 | 13.98 | 0 |
| 16 | vllm | 14.01 | 1,793 | 12 / 150 | 1.4 / 1.8 | 14.01 | 0 |
| 32 | minillm | 29.50 | 3,777 | 9 / 23 | 1.4 / 1.8 | 29.50 | 0 |
| 32 | minillm-sync | 29.40 | 3,764 | 9 / 19 | 2.0 / 2.5 | 29.40 | 0 |
| 32 | minillm-w4a16 | 29.35 | 3,757 | 16 / 38 | 2.6 / 3.6 | 29.35 | 0 |
| 32 | vllm | 29.48 | 3,773 | 13 / 27 | 1.8 / 2.4 | 29.48 | 0 |
| 64 | minillm | 58.25 | 7,456 | 13 / 25 | 1.7 / 2.3 | 58.25 | 0 |
| 64 | minillm-sync | 57.84 | 7,404 | 10 / 34 | 2.8 / 3.6 | 57.84 | 0 |
| 64 | minillm-w4a16 | 57.57 | 7,369 | 24 / 45 | 3.8 / 4.9 | 57.57 | 0 |
| 64 | vllm | 58.17 | 7,446 | 16 / 32 | 2.4 / 3.1 | 58.17 | 0 |
| 128 | minillm | 116.48 | 14,910 | 22 / 130 | 3.5 / 5.3 | 116.48 | 0 |
| 128 | minillm-sync | 115.38 | 14,768 | 16 / 118 | 5.8 / 7.0 | 115.38 | 0 |
| 128 | minillm-w4a16 | 113.78 | 14,564 | 33 / 116 | 7.7 / 9.6 | 113.78 | 0 |
| 128 | vllm | 115.42 | 14,774 | 33 / 69 | 5.8 / 6.8 | 115.42 | 0 |
| inf | minillm | 252.07 | 32,265 | 397 / 1371 | 7.1 / 8.4 | 129.06 | 0 |
| inf | minillm-sync | 229.23 | 29,342 | 456 / 1499 | 7.6 / 8.6 | 117.37 | 0 |
| inf | minillm-w4a16 | 230.34 | 29,483 | 418 / 1444 | 7.9 / 9.1 | 117.93 | 0 |
| inf | vllm | 190.83 | 24,426 | 431 / 2096 | 9.9 / 10.2 | 97.70 | 0 |

Goodput counts requests meeting TTFT <= 500 ms and TPOT <= 50 ms.

## Shared-prefix workload

| engine | shared prefix | TTFT p50 / p99 (ms) | output tok/s |
|---|---:|---:|---:|
| minillm-sync | 1024 | 8 / 14 | 2,033 |
| minillm-w4a16 | 1024 | 14 / 30 | 2,029 |
| minillm | 1024 | 9 / 17 | 2,034 |
| vllm | 1024 | 10 / 39 | 2,034 |
