# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 3,100 lines of Python:

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- FlashInfer paged-attention kernels and CUDA graphs for decode
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

**On one H200, minillm reaches 64–82% of vLLM 0.34's offline throughput and 2.6–13.7× Hugging Face `generate`. Under a burst of streaming requests it serves 78% of vLLM's throughput on Qwen2.5-7B and about as much as vLLM on 0.5B**, with greedy output checked against Hugging Face on the same hardware ([results](#results-on-an-h200)).

## Status

| | |
|---|---|
| **Verified on CPU / MPS** | 117 tests pass on a laptop CPU. Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption and prefix-cache hits. On Qwen2.5-0.5B-Instruct, 288 of 288 generated tokens are HF's argmax on CPU and on Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Verified on an H200** | 136 tests pass, including FlashInfer kernels against the reference attention (head dims 64 and 128, decodes, prompts, chunks on cached context), the engine with and without CUDA graphs under preemption, async scheduling (same tokens as synchronous, and no device sync while launching a step), and Qwen2.5-0.5B: exact against HF in fp32, and in bf16 at least 97% of tokens are HF's argmax with only near-ties differing. |
| **Written, not run** | The `flash_attn` backend. No prebuilt flash-attn wheel loads against the torch 2.14 the benchmarks used, so it stays unverified. |
| **Not built yet** | Speculative decoding, quantized weights, multi-GPU (see the roadmap). |

## Architecture

```text
         API process                                      engine-core process
 HTTP ──► FastAPI (asyncio)                            ┌───────────────────────────┐
          /v1/completions          add / abort         │ LLMEngine.step()          │
          /v1/chat/completions  ──────── pipe ───────► │  Scheduler                │
          tokenize, detokenize                         │   ├ token budget          │
          stop strings, SSE        tokens + stats      │   ├ chunked prefill       │
          admission control     ◄─────── pipe ──────── │   └ preemption            │
          /metrics, /health         (one msg/step)     │  BlockManager             │
                                                       │   ├ paged KV blocks       │
                                                       │   └ prefix cache (LRU)    │
                                                       │  ModelRunner              │
                                                       │   ├ CausalLM (own impl)   │
                                                       │   ├ FlashInfer / torch    │
                                                       │   └ CUDA graphs, sampler  │
                                                       └───────────────────────────┘
```

**Two processes** ([engine_core.py](minillm/engine_core.py)). The API server does real per-token Python work: detokenizing, matching stop strings, formatting server-sent events. If the engine ran on a thread in the same process, that work and the scheduler would take turns holding the GIL, and the GPU would wait while the event loop formats JSON. So, as in vLLM, the engine runs in a child process.

- *Protocol:* the two sides exchange small pickled messages over pipes. Each engine step sends one message with every token it produced, plus an `EngineStats` snapshot that the API process applies to its Prometheus registry.
- *Timing:* request arrival is stamped in the API process with a system-wide clock, so time to first token includes the trip to the engine.
- *Lifecycle:* the child is started with `spawn` and exits when its pipe closes. If it crashes or is killed, the API process notices within a second and fails requests instead of hanging.
- *Opting out:* `--no-engine-process` runs the engine on a thread instead.

**One kind of step.** The engine never distinguishes "prefill" from "decode". Each sequence has `num_tokens` (prompt plus generated tokens), and the first `num_computed_tokens` of them already have their K/V in the cache. A step computes some of the remaining tokens. When a sequence catches up, the model samples its next token. All of these are the same operation with different counts, so they can share one batch:

- a fresh prompt
- a 300-token chunk of a long prompt
- a single decode token
- recomputing a sequence after preemption

**Scheduling** ([scheduler.py](minillm/scheduler.py)). Each step has a token budget, and running sequences get it first, oldest first.

- *Chunked prefill:* a long prompt is split across steps, so it never stalls the requests that are already decoding.
- *Preemption:* when the KV cache runs out, the newest running sequence is preempted. Its blocks are freed and it goes back to the front of the queue. Its finished blocks usually survive in the prefix cache, so recomputing it is often cheap.
- *Admission:* new requests are admitted only on steps with no preemption, and only while a small reserve of free blocks remains.

**Memory** ([block_manager.py](minillm/block_manager.py)). The KV cache is a pool of fixed-size blocks.

- *Block tables:* a sequence's table maps its token positions to physical blocks, so memory grows one block at a time.
- *Prefix-cache keys:* a full block is registered under a hash chained from every block before it, so a hit on block *i* means the whole prefix up to *i* matches. Each hit is also checked against the block's stored token ids.
- *Eviction:* blocks are reference counted. A freed block keeps its hash and can be revived until memory is needed. Eviction is LRU, and blocks with no cached content go first.

**Attention** ([attention.py](minillm/layers/attention.py)). Every step's tokens are packed into one flat batch with no padding. Three backends:

- `torch`: gathers each sequence's blocks and runs SDPA with an explicit mask. It runs on CPU, MPS and CUDA, and the tests check it against dense attention.
- `flashinfer` (the default on CUDA): FlashInfer's paged prefill and decode kernels read blocks in place through the block table. FlashInfer plans each step once on the CPU, and every layer reuses that plan.
- `flash_attn`: flash-attn's varlen and kv-cache kernels with block tables (block size 256). Written, but not yet run.

**CUDA graphs** ([cuda_graphs.py](minillm/executor/cuda_graphs.py)). In eager mode, a decode step launches about 30 kernels per layer from Python, and for a small model that launch overhead outweighs the GPU work. At startup minillm captures the decode forward pass as CUDA graphs at a ladder of batch sizes (1, 2, 4, 8, 16, 24, 32, then every 16). Each decode step:

1. pads its batch up to the nearest captured size; padded rows write into a scratch KV block the block manager never hands out
2. copies inputs into static buffers and refills FlashInfer's fixed index buffers
3. replays the graph

Steps that contain prompt chunks run eagerly. `--enforce-eager` turns graphs off.

**Async scheduling** ([engine.py](minillm/engine.py), [scheduler.py](minillm/scheduler.py)). Every step has CPU work (scheduling, building input tensors, FlashInfer planning, applying the sampled tokens) and GPU work. Run one after the other, the GPU idles during the CPU part. So, as in vLLM, the engine launches step n+1 before it collects step n's tokens:

1. `schedule()` commits step n+1 up front, planning as if step n's tokens had already arrived. Each sequence counts its token still in flight as *pending*.
2. The model runner launches step n+1 and returns without waiting for the GPU. Any input that is one of step n's sampled tokens is copied on the GPU, straight from step n's output tensor.
3. Only then does the engine wait for step n's tokens (they come back through a pinned buffer), append them and check stop conditions.

- *Wasted work:* length limits are known in advance, so a sequence that its pending token will end is not scheduled again. EOS and stop tokens are not known in advance: such a sequence runs one extra step, and that token is discarded.
- *No hidden syncs:* nothing on the launch path may wait for the GPU. FlashInfer gets pinned host buffers, and CUDA-graph wrappers get their page indices already on the device (from host memory they copy them synchronously). The sampler takes its greedy and top-k/top-p decisions from the parameters on the CPU, not by reading device tensors. A GPU test queues half a second of GPU work and checks that a step still launches immediately.
- *Same output:* a randomized test runs 150 workloads with chunked prefill, preemption, EOS, stop tokens and length limits, and checks that async and synchronous scheduling generate identical tokens. On the H200, a small random Qwen2 model generates exactly the same bf16 tokens in both modes, sampled ones included, with CUDA graphs on and off. `--no-async-scheduling` turns it off.

## Quickstart

```bash
pip install -e '.[dev]'            # on CUDA, add the FlashInfer kernels: pip install -e '.[cuda]'
minillm-serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64, "stream": true}'
```

Any OpenAI client works against `http://localhost:8000/v1`. Offline use:

```python
from minillm import EngineConfig, LLMEngine, SamplingParams

engine = LLMEngine(EngineConfig(model="Qwen/Qwen2.5-0.5B-Instruct"))
for r in engine.generate(
    ["The capital of France is"], SamplingParams(max_tokens=16, temperature=0)
):
    print(r.text, r.finish_reason, f"ttft={r.time_to_first_token:.3f}s")
```

### Server behaviour

| | |
|---|---|
| Endpoints | `/v1/completions` (text or token-id prompts), `/v1/chat/completions` (uses the model's chat template), `/v1/models`, `/health`, `/metrics` |
| Sampling | temperature, top-p, top-k, `stop` strings (never included in output, held back while streaming), `stop_token_ids`, `ignore_eos`, `stream_options.include_usage`, cached-token counts in `usage` |
| Backpressure | over `--max-pending-requests` in flight → `429` with `Retry-After` |
| Validation | prompt + `max_tokens` over the context limit → `400` with the OpenAI error shape. Unsupported options (`n > 1`, `logprobs`, `seed`) also get a `400` instead of being silently ignored. |
| Cancellation | a client that disconnects, streaming or not, aborts its request and frees its KV blocks |
| Failure | if the engine process crashes or is killed (say, by the OOM killer), in-flight and new requests fail fast and `/health` returns `503` |
| Shutdown | stops accepting connections, drains in-flight requests (30 s), then stops the engine |

### Metrics

Prometheus series (all prefixed `minillm_`):

- latency histograms:
  - `time_to_first_token_seconds`
  - `inter_token_latency_seconds`
  - `e2e_request_latency_seconds`
  - `request_queue_time_seconds`
  - `step_duration_seconds`
- throughput:
  - `step_tokens`
  - `prompt_tokens_total`
  - `generation_tokens_total`
- requests:
  - `requests_finished_total{finish_reason}`
  - `requests_rejected_total{reason}`
- cache and memory:
  - `preemptions_total`
  - `prefix_cache_queries_total`
  - `prefix_cache_hits_total`
  - `kv_cache_usage_ratio`
- queue depth:
  - `num_requests_running`
  - `num_requests_waiting`

TTFT counts from arrival, so queueing time is included.

## Testing

```bash
pytest -q                                   # 117 tests, ~20 s on a laptop CPU, no downloads
MINILLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m slow   # real-checkpoint parity
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh  # on a GPU
```

| Test file | What it proves |
|---|---|
| `test_model_parity.py` | The engine's greedy output equals Hugging Face's on randomly initialised Qwen2, Qwen3 and Llama 3 checkpoints. This holds under batching, chunked prefill (7-token budget), preemption (18-block cache) and prefix-cache hits, and also when the KV pool is pre-filled with NaN, so stale memory can never leak into another request. |
| `test_attention.py` | Paged attention equals dense causal attention, for blocks in shuffled order, grouped-query heads, decodes, full prompts and chunks on cached context. |
| `test_block_manager.py` | A Hypothesis state machine tracks the physical contents of every block. After any sequence of admit, decode and free operations, every computed token of every live sequence is still in place and reference counts match the tables. A deliberately broken allocator fails this test (mutation-checked). |
| `test_scheduler.py` | Token budget, chunk sizes, decodes never starved by long prompts, `max_num_seqs`, preemption preserving outputs, resume order, stop conditions, abort freeing blocks, prefix-cache accounting |
| `test_server.py` | Real uvicorn over HTTP, with every test run twice (engine in its own process, and on a thread): SSE format, usage, chat template, stop strings, 400/429 responses, 32 concurrent clients, aborts on streaming and non-streaming disconnect |
| `test_engine_core.py` | The engine process: streamed output equals offline output, metrics cross the process boundary, closing a stream aborts the request in the engine, invalid requests are rejected without killing it, and a crash, a SIGKILL or a failed startup is reported instead of hanging. |
| `test_async_scheduling.py` | Async scheduling generates exactly what synchronous scheduling does: a Hypothesis test runs 150 random workloads mixing chunked prefill, preemption, EOS, stop tokens, length limits and `max_num_seqs`. An EOS stop costs exactly one discarded step and a length limit none; aborts and preemption while a step is in flight leave other requests untouched. A deliberately broken scheduler fails it (mutation-checked). |
| `test_async_engine.py` | Streamed output equals offline output. Closing a stream aborts its request. An executor crash fails in-flight and new requests. |
| `test_gpu.py` | CUDA only. Each GPU kernel matches the torch reference on random paged batches. The engine in bf16, with and without CUDA graphs and under preemption, picks the fp32 model's argmax as often as the torch backend does in bf16. Async and synchronous scheduling generate identical tokens, sampled ones included. With half a second of GPU work queued, launching a step still returns immediately, so nothing on the launch path waits for the GPU. |
| `test_real_model.py` | On a real checkpoint, every generated token is HF's argmax: exactly with the torch backend in fp32, and in bf16 for the GPU backends (at least 97%, with only near-ties differing). Teacher forcing is used so that one float near-tie can't send the comparison down a different continuation. |

The scheduler, server and async tests run on `FakeExecutor`, a deterministic stand-in for the GPU. Its next token depends only on the sequence's own tokens, so outputs must be identical however requests are batched, chunked or preempted. That's the same property the real engine needs.

## Benchmarks

Both harnesses send token-id prompts, so every engine sees byte-identical input:

- `benchmarks/bench_offline.py`: minillm, HF `generate` and vLLM on identical prompts and per-request output lengths. Each engine first warms up on 32 full-length prompts; the prefix cache is cleared before timing.
- `benchmarks/bench_serving.py`: a Poisson-arrival streaming load generator for any OpenAI-compatible server. It reports TTFT, TPOT, ITL and end-to-end latency (p50/p90/p99), throughput, and goodput under SLOs.
  - Every run draws fresh prompts, because an engine with prefix caching would otherwise serve repeated prompts from cache.
  - `--workers N` spreads the load across processes. One Python client tops out near 10k streamed tokens/s and inflates burst TTFT; the H200 runs use 4 workers.

`benchmarks/fake_server.py` serves the full HTTP stack over the fake executor with a latency cost model. CI uses it to load-test the server with no GPU.

### Results on an H200

One NVIDIA H200 per job, running the vLLM 0.34 container (torch 2.14, CUDA 13, FlashInfer 0.7.3), bf16.

- **minillm:** FlashInfer attention, decode CUDA graphs and async scheduling (the defaults). `minillm-sync` is the same engine with `--no-async-scheduling`.
- **vLLM:** defaults (CUDA graphs, prefix caching, chunked prefill).
- **Shared limits:** both engines get 256 concurrent sequences and an 8,192-token step budget.
- **HF `generate`:** static batches of 64.
- **Workload:** random token-id prompts of about 512 tokens with 128 output tokens (EOS ignored), 500 requests per measurement.

Engines in a table ran back to back on the same GPU, with two exceptions, both caused by busy nodes. HF's `generate` loop, which is CPU-bound, ran 1.4–3.9× slower in the first jobs than in a separate HF-only job and in earlier runs, so HF's numbers come from that job. The first 0.5B job also shared its node with another long job, and vLLM's server, also CPU-heavy, had roughly twice its usual per-token latency there at mid load, so the 0.5B online sweep comes from a second job. These are single runs on a shared cluster, so treat differences under about 10% as noise: vLLM's 0.5B offline figure has measured 34k, 39k and 41k tok/s in three jobs. Raw JSON and per-run reports are in [`results/`](results).

**Offline throughput** (output tokens/s; HF counts only the tokens each request asked for):

| model | workload | minillm | minillm, sync scheduling | HF `generate` | vLLM 0.34 | minillm vs HF | minillm vs vLLM |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | uniform lengths | 31,253 | 23,691 | 3,108 | 39,000 | 10.1× | 80% |
| Qwen2.5-0.5B | lengths vary ±90% | 18,362 | 14,576 | 1,338 | 22,477 | 13.7× | 82% |
| Qwen2.5-7B | uniform lengths | 6,058 | 5,755 | 2,293 | 7,904 | 2.6× | 77% |
| Qwen2.5-7B | lengths vary ±90% | 4,630 | 4,370 | 534 | 7,242 | 8.7× | 64% |

- **minillm reaches 64–82% of vLLM and 2.6–13.7× HF.**
- **Async scheduling adds 26–32% on 0.5B and 5–6% on 7B.** On a small model the CPU's share of a step is large, and it now runs while the GPU works; 7B is limited by the GPU itself.
- **Variable lengths break static batching.** HF pads every batch to its longest prompt and decodes until its longest request ends, so it loses 57% of its throughput on 0.5B and 77% on 7B. The continuous-batching engines lose 41–42% on 0.5B; on 7B minillm loses 24% and vLLM 8%.
- **7B with variable lengths is the weakest case (64%).** Steps that mix prompt chunks with decodes run eagerly in minillm, while vLLM captures graphs for those steps too.
- **CUDA graphs add 46–66% on 0.5B and 5–7% on 7B** (measured before async scheduling; [`results/*-eager`](results)).

**Where a decode step's time goes** (`benchmarks/profile_step.py`; B requests decoding in lockstep after 512-token prompts, ms per step). *GPU* is the device time of the step's work alone, measured while a spin kernel keeps the GPU ahead of the CPU, so launch gaps don't count. *Engine CPU* is everything else the engine does per step: scheduling, building inputs, launching, applying tokens.

| model | batch | GPU | engine CPU | step, sync | step, async |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1 | 2.16 | 0.39 | 2.37 | 2.17 |
| Qwen2.5-0.5B | 4 | 2.45 | 0.43 | 2.69 | 2.45 |
| Qwen2.5-0.5B | 16 | 2.61 | 0.69 | 2.96 | 2.62 |
| Qwen2.5-0.5B | 64 | 2.97 | 1.11 | 3.82 | 2.97 |
| Qwen2.5-0.5B | 256 | 4.06 | 2.95 | 6.32 | 3.91 |
| Qwen2.5-7B | 1 | 5.89 | 0.37 | 6.09 | 5.90 |
| Qwen2.5-7B | 4 | 6.30 | 0.42 | 6.52 | 6.30 |
| Qwen2.5-7B | 16 | 6.83 | 0.57 | 7.17 | 6.84 |
| Qwen2.5-7B | 64 | 7.78 | 1.08 | 8.38 | 7.70 |
| Qwen2.5-7B | 128 | 9.11 | 1.71 | 10.07 | 8.92 |

- **Async scheduling turns a step's cost from CPU + GPU into about max(CPU, GPU):** 8% faster at batch 1, and 38% at batch 256 on 0.5B, where the CPU work (3 ms) nearly matches the GPU's (4 ms).
- **At small batches the GPU work dominates, and most of it is not the model's math.** Kernel breakdown of one batch-1 decode step:

| model | kernel launches | GPU time | GEMMs | attention | other kernels |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1,153 | 2.19 ms | 0.61 ms | 0.21 ms | 1.37 ms (62%) |
| Qwen2.5-7B | 1,369 | 5.92 ms | 3.88 ms | 0.33 ms | 1.72 ms (29%) |

  On 0.5B, 62% of the GPU time goes to about a thousand small elementwise, copy and reduction kernels: RMSNorm, residual adds, rotary embeddings, SiLU and dtype casts, each written as several PyTorch ops. vLLM fuses these, which is most of why its per-token latency is lower at light load. Fusing them is the first roadmap item.

**Online serving** (Poisson arrivals, streaming, fresh prompts at every rate, load from 4 client processes).

**Qwen2.5-0.5B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | minillm | 7.0 | 898 | 14 / 33 | 2.5 / 2.9 |
| 8 | minillm-sync | 7.0 | 898 | 14 / 29 | 2.9 / 3.5 |
| 8 | vllm | 7.0 | 899 | 12 / 34 | 1.3 / 1.6 |
| 16 | minillm | 14.0 | 1,787 | 14 / 32 | 2.7 / 3.1 |
| 16 | minillm-sync | 13.9 | 1,785 | 14 / 24 | 3.2 / 3.8 |
| 16 | vllm | 14.0 | 1,793 | 12 / 23 | 1.4 / 1.8 |
| 32 | minillm | 29.2 | 3,740 | 16 / 36 | 3.2 / 3.9 |
| 32 | minillm-sync | 29.1 | 3,731 | 15 / 25 | 4.0 / 5.1 |
| 32 | vllm | 29.5 | 3,772 | 13 / 37 | 1.6 / 2.3 |
| 64 | minillm | 57.1 | 7,314 | 25 / 142 | 4.3 / 5.5 |
| 64 | minillm-sync | 56.8 | 7,273 | 18 / 92 | 5.9 / 8.2 |
| 64 | vllm | 58.2 | 7,445 | 16 / 42 | 2.3 / 3.4 |
| 128 | minillm | 111.3 | 14,244 | 37 / 57 | 7.2 / 7.6 |
| 128 | minillm-sync | 95.3 | 12,199 | 33 / 55 | 9.6 / 10.2 |
| 128 | vllm | 114.4 | 14,649 | 33 / 62 | 6.2 / 6.9 |
| burst | minillm | 184.7 | 23,636 | 136 / 309 | 4.1 / 4.7 |
| burst | minillm-sync | 143.0 | 18,302 | 75 / 278 | 6.2 / 6.7 |
| burst | vllm | 181.2 | 23,192 | 105 / 186 | 4.6 / 5.0 |

**Qwen2.5-7B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | minillm | 3.5 | 449 | 29 / 54 | 6.6 / 7.3 |
| 4 | minillm-sync | 3.5 | 448 | 25 / 51 | 7.1 / 8.2 |
| 4 | vllm | 3.5 | 449 | 23 / 47 | 5.0 / 5.3 |
| 8 | minillm | 7.0 | 892 | 31 / 51 | 7.2 / 8.1 |
| 8 | minillm-sync | 7.0 | 891 | 26 / 49 | 7.8 / 8.8 |
| 8 | vllm | 7.0 | 894 | 23 / 52 | 5.2 / 5.7 |
| 16 | minillm | 14.6 | 1,863 | 35 / 63 | 8.6 / 10.2 |
| 16 | minillm-sync | 14.5 | 1,861 | 29 / 65 | 9.5 / 11.5 |
| 16 | vllm | 14.6 | 1,873 | 25 / 56 | 5.8 / 6.6 |
| 32 | minillm | 28.3 | 3,625 | 52 / 154 | 13.3 / 17.3 |
| 32 | minillm-sync | 28.2 | 3,614 | 51 / 154 | 19.1 / 22.8 |
| 32 | vllm | 28.7 | 3,670 | 30 / 65 | 7.2 / 8.6 |
| 64 | minillm | 41.4 | 5,303 | 247 / 501 | 21.5 / 23.3 |
| 64 | minillm-sync | 38.9 | 4,974 | 249 / 433 | 23.3 / 25.4 |
| 64 | vllm | 53.6 | 6,860 | 106 / 291 | 16.5 / 17.7 |
| burst | minillm | 42.8 | 5,482 | 696 / 1774 | 17.6 / 21.8 |
| burst | minillm-sync | 39.6 | 5,068 | 465 / 1790 | 21.5 / 25.0 |
| burst | vllm | 54.7 | 6,999 | 590 / 1433 | 13.3 / 17.8 |


- **Async scheduling lowers per-token latency at every load.** On 0.5B it falls from 2.9 to 2.5 ms at 8 req/s, 5.9 to 4.3 ms at 64 req/s and 9.6 to 7.2 ms at 128 req/s, and burst throughput rises 29% (18.3k to 23.6k tok/s). On 7B latency falls from 7.1 to 6.6 ms at 4 req/s and 19.1 to 13.3 ms at 32 req/s, and burst throughput rises 8%.
- **It costs time to first token.** A request that arrives while a step runs now waits for the step already launched behind it too.
  - At light load that is about one decode step: on 7B, median TTFT goes from 25 to 29 ms at 4 req/s; on 0.5B it is unchanged.
  - Under a burst it is larger: on 7B, median TTFT rose from 465 to 696 ms. Four repeat runs in two more jobs reproduced it (622–653 vs 456–481 ms).
  - The engine alone shows no such cost: on the fake executor, fed 500 requests at once without HTTP, async scheduling is faster at every percentile. Collecting prompt-heavy steps before scheduling more work did not recover it either (tested on the H200). The cause is still open.
  - `--no-async-scheduling` restores synchronous scheduling for TTFT-critical bursty traffic.
- **Against vLLM, under load:** on 0.5B, minillm's burst throughput is level with vLLM's in the same job (23.6k vs 23.2k tok/s; vLLM measured 29.4k in an earlier job, so call it 80–100%). On 7B it is 78% (5.5k vs 7.0k tok/s).
- **At light load, vLLM is still faster per token:** 1.3 vs 2.5 ms on 0.5B, 5.0 vs 6.6 ms on 7B. The profile above shows why: small unfused kernels on the GPU, not CPU overhead.
- **A single client process skews the results.** With one client process, vLLM's 0.5B burst measured 9.8k tok/s; with four it measures 29.4k. One process tops out near 10k streamed tokens/s and inflates burst TTFT.
- **CUDA graphs matter online too:** without them, minillm's 0.5B per-token latency at light load is 11.4 ms instead of about 3 (before async scheduling).
- **Shared 1,024-token system prompt at 16 req/s:** with prefix caching, median TTFT is 15 ms (minillm) vs 11 ms (vLLM) on 0.5B, and 26 vs 18 ms on 7B.

### Reproducing

`scripts/explorer/` runs everything on a Slurm cluster from the vLLM container image (vLLM 0.34, torch 2.14, CUDA 13, FlashInfer 0.7.3), so minillm, vLLM and HF share one torch build. Set `MINILLM_SIF` and `HF_HOME` if your paths differ from the defaults.

```bash
srun -p short -c 4 --mem=16G -t 20 bash scripts/explorer/setup.sh          # once: test-only packages
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh
ENGINES="minillm minillm-sync vllm hf" SERVE_ENGINES="minillm minillm-sync vllm" \
  sbatch -p gpu --gres=gpu:h200:1 -c 8 --mem=64G -t 3:00:00 \
  scripts/explorer/bench.sh Qwen/Qwen2.5-7B-Instruct h200-qwen2.5-7b 4 8 16 32 64 inf
PROFILE_ARGS="--kernels 1" srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 \
  bash scripts/explorer/profile.sh Qwen/Qwen2.5-7B-Instruct h200-profile
python scripts/readme_tables.py                                             # tables below
```

Each run writes JSON per measurement and a `REPORT.md` (via `scripts/report.py`) to `results/<tag>/`.

### Dev-machine numbers (Apple M4, MPS)

On a MacBook the portable `torch` backend runs Qwen2.5-0.5B at 197 tok/s on uniform lengths (HF `generate`: 264) and 142 tok/s when lengths vary ±90% (HF: 139). That backend gathers each sequence's context out of the paged cache every layer and step, which is the cost FlashInfer's paged kernels remove on CUDA. SDPA's native GQA gave +17% there; shape bucketing and dropping the NaN guard made no measurable difference, so neither was kept.

## Layout

```text
minillm/
  scheduler.py, block_manager.py, sequence.py, sampling.py   scheduling and memory (no torch)
  engine.py, async_engine.py, metrics.py, tokenizer.py       engine loop, streaming, metrics
  executor/model_runner.py   batch tensors, KV-cache sizing and allocation, sampling
  executor/cuda_graphs.py    decode CUDA graphs (FlashInfer)
  executor/fake.py           deterministic executor and byte tokenizer for tests and CI
  models/llama.py, loader.py model implementation, safetensors loading
  layers/attention.py, sampler.py
  entrypoints/openai_server.py, protocol.py
benchmarks/  bench_offline.py, bench_serving.py, fake_server.py, profile_step.py
scripts/     report.py, readme_tables.py
scripts/explorer/  Slurm + Apptainer jobs: setup, GPU tests, benchmarks, profiling
results/     H200 benchmark results (JSON + REPORT.md per run)
tests/
```

## Roadmap

1. **Fuse the small kernels.** At batch 1, a 0.5B decode step spends 62% of its GPU time (1.4 of 2.2 ms) in about a thousand small elementwise, copy and reduction kernels: RMSNorm, residual adds, rotary embeddings, SiLU and dtype casts, each written as several PyTorch ops. FlashInfer already ships fused kernels for all of them (`fused_add_rmsnorm`, `apply_rope_with_cos_sin_cache_inplace`, `silu_and_mul`), as used by SGLang. This, not CPU overhead, is what separates minillm from vLLM at light load.
2. **CUDA graphs for mixed steps.** Steps that contain prompt chunks still run eagerly, which is part of why the variable-length workload trails vLLM by more than the uniform one. vLLM captures everything except attention ("piecewise" graphs).
3. **Speculative decoding** (n-gram / prompt lookup first) and **4-bit weights**, using the W4A16 Triton kernel from Quantized-inference-lab.
4. **Container image** built and smoke-tested in CI on a GPU runner.

## License

Apache License 2.0. See [LICENSE](LICENSE).
