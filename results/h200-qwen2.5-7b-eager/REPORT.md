# Results: h200-qwen2.5-7b-eager

Model `Qwen/Qwen2.5-7B-Instruct` on NVIDIA H200, 500 prompts, ~512 in / ~128 out tokens.

## Offline throughput

| workload | engine | output tok/s | total tok/s | vs HF | vs vLLM |
|---|---|---:|---:|---:|---:|
| uniform lengths | minillm | 5,466 | 27,332 | - | - |
| lengths ±90% | minillm | 4,081 | 21,333 | - | - |
