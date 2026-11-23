# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 3,100 lines of Python:

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- FlashInfer paged-attention kernels and CUDA graphs for decode
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

**On one H200, minillm reaches 60–73% of vLLM 0.34's offline throughput and 2.6–11.5× Hugging Face `generate`, and 63–72% of vLLM's serving throughput under a burst of streaming requests**, with greedy output checked against Hugging Face on the same hardware ([results](#results-on-an-h200)).

## Status

| | |
|---|---|
| **Verified on CPU / MPS** | 110 tests pass on a laptop CPU. Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption and prefix-cache hits. On Qwen2.5-0.5B-Instruct, 288 of 288 generated tokens are HF's argmax on CPU and on Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Verified on an H200** | 125 tests pass, including FlashInfer kernels against the reference attention (head dims 64 and 128, decodes, prompts, chunks on cached context), the engine with and without CUDA graphs under preemption, and Qwen2.5-0.5B: exact against HF in fp32, and in bf16 at least 97% of tokens are HF's argmax with only near-ties differing. |
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
pytest -q                                   # 110 tests, ~20 s on a laptop CPU, no downloads
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
| `test_async_engine.py` | Streamed output equals offline output. Closing a stream aborts its request. An executor crash fails in-flight and new requests. |
| `test_gpu.py` | CUDA only. Each GPU kernel matches the torch reference on random paged batches. The engine in bf16, with and without CUDA graphs and under preemption, picks the fp32 model's argmax as often as the torch backend does in bf16. |
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

- **minillm:** FlashInfer attention and decode CUDA graphs.
- **vLLM:** defaults (CUDA graphs, prefix caching, chunked prefill).
- **Shared limits:** both engines get 256 concurrent sequences and an 8,192-token step budget.
- **HF `generate`:** static batches of 64.
- **Workload:** random token-id prompts of about 512 tokens with 128 output tokens (EOS ignored), 500 requests per measurement.

All engines in a job ran back to back on the same GPU. These are single runs on a shared cluster, so treat differences under about 10% as noise: vLLM's 0.5B offline figure, for example, measured 34k and 41k tok/s in two runs. Raw JSON and per-run reports are in [`results/`](results).

**Offline throughput** (output tokens/s; HF counts only the tokens each request asked for):

| model | workload | minillm | minillm, no CUDA graphs | HF `generate` | vLLM 0.34 | minillm vs HF | minillm vs vLLM |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | uniform lengths | 23,001 | 13,866 | 2,577 | 33,699 | 8.9× | 68% |
| Qwen2.5-0.5B | lengths vary ±90% | 13,407 | 9,206 | 1,170 | 19,341 | 11.5× | 69% |
| Qwen2.5-7B | uniform lengths | 5,757 | 5,466 | 2,235 | 7,886 | 2.6× | 73% |
| Qwen2.5-7B | lengths vary ±90% | 4,368 | 4,081 | 526 | 7,282 | 8.3× | 60% |

- **minillm reaches 60–73% of vLLM and 2.6–11.5× HF.**
- **Variable lengths break static batching.** HF pads every batch to its longest prompt and decodes until its longest request ends, so it loses 55% of its throughput on 0.5B and 76% on 7B (2.2k down to 0.5k tok/s). The continuous-batching engines lose 42–43% on 0.5B; on 7B minillm loses 24% and vLLM 8%.
- **CUDA graphs add 46–66% on 0.5B and 5–7% on 7B.** A small model's decode step is mostly Python launching kernels, and graphs remove that; 7B is limited by the GPU itself.
- **7B with variable lengths is the weakest case (60%).** Steps that mix prompt chunks with decodes run eagerly in minillm, while vLLM captures graphs for those steps too.

**Online serving** (Poisson arrivals, streaming, fresh prompts at every rate, load from 4 client processes). `minillm` runs its engine in its own process (the default); `minillm-1proc` is the same engine on a thread inside the API process (`--no-engine-process`). All three servers ran back to back in one job per model.

**Qwen2.5-0.5B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | minillm | 7.0 | 898 | 14 / 31 | 2.9 / 3.4 |
| 8 | minillm-1proc | 7.0 | 898 | 14 / 36 | 3.1 / 3.9 |
| 8 | vllm | 7.0 | 900 | 12 / 33 | 1.3 / 1.6 |
| 16 | minillm | 13.9 | 1,785 | 14 / 23 | 3.2 / 3.8 |
| 16 | minillm-1proc | 13.9 | 1,785 | 14 / 53 | 3.4 / 4.3 |
| 16 | vllm | 14.0 | 1,794 | 12 / 41 | 1.4 / 1.7 |
| 32 | minillm | 29.2 | 3,733 | 15 / 28 | 4.0 / 5.2 |
| 32 | minillm-1proc | 29.1 | 3,730 | 16 / 28 | 4.9 / 5.9 |
| 32 | vllm | 29.5 | 3,773 | 13 / 35 | 1.5 / 2.1 |
| 64 | minillm | 56.8 | 7,273 | 17 / 73 | 5.8 / 7.4 |
| 64 | minillm-1proc | 56.6 | 7,247 | 22 / 41 | 9.4 / 10.1 |
| 64 | vllm | 58.3 | 7,456 | 14 / 36 | 2.0 / 2.7 |
| 128 | minillm | 95.2 | 12,184 | 31 / 53 | 9.4 / 10.1 |
| 128 | minillm-1proc | 79.9 | 10,223 | 37 / 57 | 11.6 / 11.8 |
| 128 | vllm | 115.9 | 14,839 | 29 / 57 | 4.7 / 6.3 |
| burst | minillm | 145.0 | 18,564 | 103 / 290 | 5.8 / 6.8 |
| burst | minillm-1proc | 103.3 | 13,220 | 90 / 305 | 8.8 / 9.5 |
| burst | vllm | 229.5 | 29,373 | 128 / 191 | 3.2 / 3.5 |

**Qwen2.5-7B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | minillm | 3.5 | 449 | 25 / 47 | 7.1 / 8.0 |
| 4 | minillm-1proc | 3.5 | 449 | 25 / 50 | 7.2 / 8.2 |
| 4 | vllm | 3.5 | 449 | 22 / 38 | 5.0 / 5.3 |
| 8 | minillm | 7.0 | 891 | 25 / 46 | 7.7 / 8.7 |
| 8 | minillm-1proc | 7.0 | 891 | 25 / 51 | 8.0 / 9.1 |
| 8 | vllm | 7.0 | 894 | 22 / 37 | 5.1 / 5.7 |
| 16 | minillm | 14.5 | 1,861 | 29 / 64 | 9.5 / 11.5 |
| 16 | minillm-1proc | 14.5 | 1,861 | 29 / 63 | 10.2 / 12.2 |
| 16 | vllm | 14.6 | 1,874 | 24 / 58 | 5.6 / 6.5 |
| 32 | minillm | 28.2 | 3,614 | 48 / 359 | 18.6 / 23.0 |
| 32 | minillm-1proc | 28.0 | 3,583 | 71 / 156 | 28.6 / 31.9 |
| 32 | vllm | 28.7 | 3,674 | 27 / 64 | 7.0 / 8.3 |
| 64 | minillm | 39.1 | 5,010 | 237 / 411 | 23.0 / 25.2 |
| 64 | minillm-1proc | 34.9 | 4,471 | 249 / 730 | 25.5 / 29.8 |
| 64 | vllm | 54.3 | 6,946 | 84 / 191 | 16.4 / 17.5 |
| burst | minillm | 40.2 | 5,144 | 464 / 1809 | 20.9 / 23.7 |
| burst | minillm-1proc | 36.1 | 4,619 | 475 / 1827 | 23.5 / 26.9 |
| burst | vllm | 55.7 | 7,125 | 536 / 1414 | 13.5 / 17.9 |

- **The separate engine process pays off under load.** On 0.5B it serves a burst 40% faster than the single-process server (18.6k vs 13.2k tok/s) and cuts per-token latency at 64 req/s from 9.4 to 5.8 ms. On 7B the burst gain is 11% (5.1k vs 4.6k tok/s), and latency at 32 req/s drops from 28.6 to 18.6 ms. At light load the two modes are within 0.2 ms: the gain comes from no longer sharing the GIL with the HTTP layer when it is busy.
- **Against vLLM, under load:** minillm reaches 63% of vLLM's burst throughput on 0.5B and 72% on 7B, close to the offline ratios.
- **Light load:** time to first token is within a few milliseconds of vLLM's. Per-token latency is still higher: 2.9 vs 1.3 ms on 0.5B, and 7.1 vs 5.0 ms on 7B. Each step does its CPU work in Python (scheduling, building batch tensors, FlashInfer planning, sampling) while the GPU waits; vLLM overlaps that work with the previous step.
- **A single client process skews the results.** With one client process, vLLM's 0.5B burst measured 9.8k tok/s; with four it measures 29.4k. One process tops out near 10k streamed tokens/s and inflates burst TTFT.
- **CUDA graphs matter online too:** without them, minillm's 0.5B per-token latency at light load is 11.4 ms instead of about 3.
- **Shared 1,024-token system prompt at 16 req/s:** with prefix caching, median TTFT is 14 ms (minillm) vs 10.5 ms (vLLM) on 0.5B, and 21 vs 17 ms on 7B.

### Reproducing

`scripts/explorer/` runs everything on a Slurm cluster from the vLLM container image (vLLM 0.34, torch 2.14, CUDA 13, FlashInfer 0.7.3), so minillm, vLLM and HF share one torch build. Set `MINILLM_SIF` and `HF_HOME` if your paths differ from the defaults.

```bash
srun -p short -c 4 --mem=16G -t 20 bash scripts/explorer/setup.sh          # once: test-only packages
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh
sbatch -p gpu-short --gres=gpu:h200:1 -c 8 --mem=64G -t 2:00:00 \
  scripts/explorer/bench.sh Qwen/Qwen2.5-7B-Instruct h200-qwen2.5-7b 4 8 16 32 64 inf
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
benchmarks/  bench_offline.py, bench_serving.py, fake_server.py
scripts/     report.py, readme_tables.py
scripts/explorer/  Slurm + Apptainer jobs: setup, GPU tests, benchmarks
results/     H200 benchmark results (JSON + REPORT.md per run)
tests/
```

## Roadmap

1. **Overlap CPU scheduling with GPU work.** Build step *n+1*'s batch while step *n* runs. This is the main remaining cost: per-token latency at light load is 2.9 ms against vLLM's 1.3 ms on 0.5B, and the gap persists under load.
2. **CUDA graphs for mixed steps.** Steps that contain prompt chunks still run eagerly, which is part of why the variable-length workload trails vLLM by more than the uniform one. vLLM captures everything except attention ("piecewise" graphs).
3. **Speculative decoding** (n-gram / prompt lookup first) and **4-bit weights**, using the W4A16 Triton kernel from Quantized-inference-lab.
4. **Container image** built and smoke-tested in CI on a GPU runner.
