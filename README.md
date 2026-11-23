# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 3,100 lines of Python:

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- FlashInfer paged-attention kernels and CUDA graphs for decode
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

**On one H200, minillm reaches 60–73% of vLLM 0.34's offline throughput and 2.6–11.5× Hugging Face `generate`**, with greedy output checked against Hugging Face on the same hardware ([results](#results-on-an-h200)).

## Status

| | |
|---|---|
| **Verified on CPU / MPS** | 85 tests pass on a laptop CPU. Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption and prefix-cache hits. On Qwen2.5-0.5B-Instruct, 288 of 288 generated tokens are HF's argmax on CPU and on Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Verified on an H200** | 100 tests pass, including FlashInfer kernels against the reference attention (head dims 64 and 128, decodes, prompts, chunks on cached context), the engine with and without CUDA graphs under preemption, and Qwen2.5-0.5B: exact against HF in fp32, and in bf16 at least 97% of tokens are HF's argmax with only near-ties differing. |
| **Written, not run** | The `flash_attn` backend. No prebuilt flash-attn wheel loads against the torch 2.14 the benchmarks used, so it stays unverified. |
| **Not built yet** | Speculative decoding, quantized weights, multi-GPU (see the roadmap). |

## Architecture

```text
 HTTP clients ──► FastAPI (asyncio)                         engine thread
                  /v1/completions                     ┌───────────────────────────┐
                  /v1/chat/completions   add / abort  │ LLMEngine.step()          │
                  tokenize, detokenize ─────────────► │  Scheduler                │
                  stop strings, SSE      queue        │   ├ token budget          │
                  admission control                   │   ├ chunked prefill       │
                  /metrics, /health  ◄─────────────── │   └ preemption            │
                                      per-step tokens │  BlockManager             │
                                                      │   ├ paged KV blocks       │
                                                      │   └ prefix cache (LRU)    │
                                                      │  ModelRunner              │
                                                      │   ├ CausalLM (own impl)   │
                                                      │   ├ paged attention       │
                                                      │   └ sampler               │
                                                      └───────────────────────────┘
```

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
| Failure | if the engine thread dies, in-flight and new requests fail fast and `/health` returns `503` |
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
pytest -q                                   # 85 tests, ~12 s on a laptop CPU, no downloads
MINILLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m slow   # real-checkpoint parity
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh  # on a GPU
```

| Test file | What it proves |
|---|---|
| `test_model_parity.py` | The engine's greedy output equals Hugging Face's on randomly initialised Qwen2, Qwen3 and Llama 3 checkpoints. This holds under batching, chunked prefill (7-token budget), preemption (18-block cache) and prefix-cache hits, and also when the KV pool is pre-filled with NaN, so stale memory can never leak into another request. |
| `test_attention.py` | Paged attention equals dense causal attention, for blocks in shuffled order, grouped-query heads, decodes, full prompts and chunks on cached context. |
| `test_block_manager.py` | A Hypothesis state machine tracks the physical contents of every block. After any sequence of admit, decode and free operations, every computed token of every live sequence is still in place and reference counts match the tables. A deliberately broken allocator fails this test (mutation-checked). |
| `test_scheduler.py` | Token budget, chunk sizes, decodes never starved by long prompts, `max_num_seqs`, preemption preserving outputs, resume order, stop conditions, abort freeing blocks, prefix-cache accounting |
| `test_server.py` | Real uvicorn over HTTP: SSE format, usage, chat template, stop strings, 400/429 responses, 32 concurrent clients, aborts on streaming and non-streaming disconnect |
| `test_async_engine.py` | Streamed output equals offline output. Closing a stream aborts its request. An executor crash fails in-flight and new requests. |
| `test_gpu.py` | CUDA only. Each GPU kernel matches the torch reference on random paged batches. The engine in bf16, with and without CUDA graphs and under preemption, picks the fp32 model's argmax as often as the torch backend does in bf16. |
| `test_real_model.py` | On a real checkpoint, every generated token is HF's argmax: exactly with the torch backend in fp32, and in bf16 for the GPU backends (at least 97%, with only near-ties differing). Teacher forcing is used so that one float near-tie can't send the comparison down a different continuation. |

The scheduler, server and async tests run on `FakeExecutor`, a deterministic stand-in for the GPU. Its next token depends only on the sequence's own tokens, so outputs must be identical however requests are batched, chunked or preempted. That's the same property the real engine needs.

## Benchmarks

Both harnesses send token-id prompts, so every engine sees byte-identical input:

- `benchmarks/bench_offline.py`: minillm, HF `generate` and vLLM on identical prompts and per-request output lengths. Each engine first warms up on 32 full-length prompts; the prefix cache is cleared before timing.
- `benchmarks/bench_serving.py`: a Poisson-arrival streaming load generator for any OpenAI-compatible server. It reports TTFT, TPOT, ITL and end-to-end latency (p50/p90/p99), throughput, and goodput under SLOs. Every run draws fresh prompts, because an engine with prefix caching would otherwise serve repeated prompts from cache.

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

**Online serving** (Poisson arrivals, streaming, fresh prompts at every rate):

**Qwen2.5-0.5B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | minillm | 7.7 | 991 | 15 / 27 | 3.2 / 3.9 |
| 8 | vllm | 7.7 | 989 | 14 / 55 | 1.4 / 1.7 |
| 16 | minillm | 16.0 | 2,050 | 16 / 28 | 3.8 / 4.4 |
| 16 | vllm | 16.1 | 2,067 | 15 / 30 | 1.5 / 1.8 |
| 32 | minillm | 28.3 | 3,622 | 17 / 28 | 4.7 / 5.5 |
| 32 | vllm | 28.8 | 3,688 | 17 / 36 | 1.8 / 2.7 |
| 64 | minillm | 53.9 | 6,895 | 34 / 310 | 11.5 / 14.7 |
| 64 | vllm | 57.9 | 7,417 | 25 / 52 | 3.4 / 5.8 |
| 128 | minillm | 66.1 | 8,458 | 70 / 361 | 23.3 / 27.2 |
| 128 | vllm | 88.3 | 11,306 | 38 / 88 | 5.3 / 6.3 |
| burst | minillm | 76.6 | 9,804 | 801 / 4380 | 22.6 / 28.1 |
| burst | vllm | 76.4 | 9,779 | 889 / 6184 | 13.8 / 20.9 |

**Qwen2.5-7B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | minillm | 3.9 | 496 | 25 / 43 | 7.2 / 8.4 |
| 4 | vllm | 3.9 | 497 | 22 / 30 | 4.9 / 5.3 |
| 8 | minillm | 8.0 | 1,027 | 26 / 54 | 8.0 / 9.0 |
| 8 | vllm | 8.1 | 1,032 | 22 / 41 | 5.2 / 5.6 |
| 16 | minillm | 14.2 | 1,822 | 30 / 66 | 9.8 / 11.2 |
| 16 | vllm | 14.4 | 1,841 | 24 / 42 | 5.6 / 6.2 |
| 32 | minillm | 28.4 | 3,631 | 58 / 141 | 22.9 / 33.2 |
| 32 | vllm | 30.0 | 3,841 | 30 / 62 | 7.1 / 8.4 |
| 64 | minillm | 34.3 | 4,392 | 461 / 3382 | 51.5 / 58.9 |
| 64 | vllm | 51.3 | 6,572 | 68 / 205 | 15.3 / 19.9 |
| burst | minillm | 37.6 | 4,812 | 3907 / 10289 | 46.0 / 48.3 |
| burst | vllm | 54.5 | 6,980 | 2955 / 7028 | 28.9 / 31.6 |

- **Light load:** time to first token is within a few milliseconds of vLLM's. minillm's per-token latency is higher: 3.2 vs 1.4 ms on 0.5B, and 7.2 vs 4.9 ms on 7B. Each step does more CPU work in Python (scheduling, batch tensors, FlashInfer planning, sampling), and vLLM overlaps that work with the GPU.
- **Saturation:** on 7B, minillm tops out at 34–38 req/s and vLLM at 51–55, about the same ratio as offline. On 0.5B, minillm's per-token latency climbs to 23 ms at 128 req/s while vLLM's stays at 5 ms. At the burst point both engines reach about 9.8k tok/s, which is roughly the ceiling of the single-process load generator, so the 0.5B saturation numbers understate both.
- **CUDA graphs matter online too:** without them, minillm's 0.5B per-token latency at light load is 11.4 ms instead of 3.2.
- **Shared 1,024-token system prompt at 16 req/s:** with prefix caching, median TTFT is 16 ms (minillm) vs 14 ms (vLLM) on 0.5B, and 21 vs 18 ms on 7B.

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

1. **Engine core in its own process.** Under load on Qwen2.5-0.5B, minillm's per-token latency climbs to 23 ms at 128 req/s, against vLLM's 5 ms, even though the engine alone produces 23k tok/s offline. Tokenizing, detokenizing and SSE formatting share one Python process (and the GIL) with the scheduler. vLLM separates them; moving minillm's engine behind a ZeroMQ socket would too.
2. **Overlap CPU scheduling with GPU work.** Build step *n+1*'s batch while step *n* runs. This is the main remaining cost at light load: 3.2 ms per token against vLLM's 1.4 ms on 0.5B.
3. **CUDA graphs for mixed steps.** Steps that contain prompt chunks still run eagerly, which is part of why the variable-length workload trails vLLM by more than the uniform one. vLLM captures everything except attention ("piecewise" graphs).
4. **Speculative decoding** (n-gram / prompt lookup first) and **4-bit weights**, using the W4A16 Triton kernel from Quantized-inference-lab.
5. **Multi-process load generator.** A single Python client tops out near 10k streamed tokens/s, which caps the 0.5B saturation measurements for both engines.
6. **Container image** built and smoke-tested in CI on a GPU runner.
