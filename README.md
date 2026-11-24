# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 4,200 lines of Python:

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- FlashInfer paged attention and fused norm, rotary and activation kernels
- CUDA graphs for decode, and async scheduling that overlaps CPU work with the GPU
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

**On one H200, minillm outperforms vLLM 0.34 on Qwen2.5-0.5B: 110–113% of its offline throughput, lower per-token latency and time to first token at every load, and 1.2–1.5× its burst serving throughput. On Qwen2.5-7B it reaches 95% of vLLM offline (77% with variable lengths) and matches it at light load and under a burst.** It runs 3.3–20× faster than Hugging Face `generate`, with greedy output checked against Hugging Face on the same hardware ([results](#results-on-an-h200)).

## Status

| | |
|---|---|
| **Verified on CPU / MPS** | 117 tests pass on a laptop CPU. Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption and prefix-cache hits. On Qwen2.5-0.5B-Instruct, 288 of 288 generated tokens are HF's argmax on CPU and on Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Verified on an H200** | 140 tests pass, including FlashInfer kernels against the reference attention (head dims 64 and 128, decodes, prompts, chunks on cached context), the engine with and without CUDA graphs under preemption, fused kernels against the PyTorch ops, async scheduling (same tokens as synchronous, and no device sync while launching a step), and Qwen2.5-0.5B: exact against HF in fp32, and in bf16 at least 97% of tokens are HF's argmax with only near-ties differing. |
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

**CUDA graphs** ([cuda_graphs.py](minillm/executor/cuda_graphs.py)). In eager mode, a decode step launches every kernel from Python, a dozen per layer (about 45 without fused kernels), and for a small model that launch overhead outweighs the GPU work. At startup minillm captures the decode forward pass as CUDA graphs at a ladder of batch sizes (1, 2, 4, 8, 16, 24, 32, then every 16). Each decode step:

1. pads its batch up to the nearest captured size; padded rows write into a scratch KV block the block manager never hands out
2. copies inputs into static buffers and refills FlashInfer's fixed index buffers
3. replays the graph

Steps that contain prompt chunks run eagerly. `--enforce-eager` turns graphs off.

**Fused kernels** ([fused.py](minillm/layers/fused.py)). Written as plain PyTorch, RMSNorm is eight small kernels (casts, square, mean, add, rsqrt, two multiplies), rotary embeddings about ten per tensor, and SiLU-and-mul two. At batch 1 they took 62% of a 0.5B decode step's GPU time. On CUDA minillm runs each as one FlashInfer kernel instead, and folds every residual add into the norm that follows it (`residual += x; x = rmsnorm(residual)` in one kernel), the layer structure vLLM and SGLang use. Rotary embeddings rotate q and k in place, straight from the fused QKV projection's output. A 0.5B decode step drops from 1,153 kernel launches to 328 and from 2.2 to 1.1 ms of GPU time. The plain PyTorch path still runs on CPU, MPS and the `torch` backend, where the parity tests check it token for token against Hugging Face; `--no-fused-kernels` selects it on CUDA too.

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

## Container

The [`Dockerfile`](Dockerfile) builds a CUDA serving image with no CUDA toolkit or compiler inside. PyTorch 2.11 is built for CUDA 12.8, and its wheels bring the CUDA libraries. FlashInfer 0.6.17 comes with its precompiled kernels (`flashinfer-jit-cache`), so nothing is compiled at runtime. Both are the newest releases with CUDA 12.8 builds. The driver comes from the host through the NVIDIA Container Toolkit and must support CUDA 12.8 (570 or newer).

```bash
docker build -t minillm .
docker run --gpus all -p 8000:8000 -v hf-cache:/data minillm \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

- Arguments after the image name go to `minillm-serve`.
- The server runs as an unprivileged user (uid 1000).
- Weights download to the `/data` volume (`HF_HOME`). To reuse the host's cache, mount it instead with `-v ~/.cache/huggingface:/data` (it must be writable by uid 1000). Add `-e HF_TOKEN` for gated models.
- A health check polls `/health` on port 8000. It passes once the weights are loaded and the CUDA graphs captured.
- The image is 10.2 GB unpacked (5.7 GB compressed). Most of it is the CUDA libraries that ship with torch (4.3 GB: cuDNN, cuBLAS, NCCL and others), FlashInfer's kernels (1.8 GB) and torch itself (1.6 GB).

**Checked on a GPU** (RTX 3050 Ti Laptop, 4 GB, driver 580) serving Qwen2.5-0.5B-Instruct with `--max-model-len 2048 --max-num-seqs 16 --gpu-memory-utilization 0.85`. The log shows `attention=flashinfer, fused kernels` and decode CUDA graphs captured. `/health` answered about 30 s after start with the 1 GB weight download, and 9 s with the weights cached. Completions, streaming chat and 32 concurrent requests returned sensible text, the container turned healthy, and `docker stop` shut it down cleanly. The image has not been run on an H200 yet; it includes FlashInfer's sm90 kernels for head dims 64 and 128.

**In CI** the `container` workflow builds the image and checks it on CPU, since GitHub's runners have no GPU. It runs on every push to master and on pull requests that change the Dockerfile or `pyproject.toml`:

- torch and FlashInfer import, with torch built for CUDA 12.8
- `minillm-serve --help` runs
- the fake-engine server (`benchmarks/fake_server.py`) answers a completion and a streamed chat from inside the container, as the image's user
- the CPU test suite passes against the installed package

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
| `test_gpu.py` | CUDA only. Each GPU kernel matches the torch reference on random paged batches. The engine in bf16, with and without CUDA graphs and under preemption, picks the fp32 model's argmax as often as the torch backend does in bf16. The fused norm, residual-add, rotary and SiLU kernels match the PyTorch ops within bf16 rounding, rotary included on strided q/k views out to 32k positions. Async and synchronous scheduling generate identical tokens, sampled ones included. With half a second of GPU work queued, launching a step still returns immediately, so nothing on the launch path waits for the GPU. |
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

- **minillm:** FlashInfer attention and fused kernels, decode CUDA graphs and async scheduling (the defaults). `minillm-unfused` is the same engine with `--no-fused-kernels`.
- **vLLM:** defaults (CUDA graphs, prefix caching, chunked prefill).
- **Shared limits:** both engines get 256 concurrent sequences and an 8,192-token step budget.
- **HF `generate`:** static batches of 64.
- **Workload:** random token-id prompts of about 512 tokens with 128 output tokens (EOS ignored), 500 requests per measurement.

minillm, minillm-unfused and vLLM ran back to back on the same GPU, one job per model. HF ran in a separate job: its `generate` loop is CPU-bound, and on a busy node it ran up to 3.9× slower. These are single runs on a shared cluster, so treat differences under about 10% as noise: vLLM's 0.5B offline figure has measured 34k, 39k, 41k and 41k tok/s in four jobs, and its 0.5B burst figure 23k to 29k. Correctness is checked against Hugging Face (see Testing), not against vLLM. Raw JSON and per-run reports are in [`results/`](results).

**Offline throughput** (output tokens/s; HF counts only the tokens each request asked for):

| model | workload | minillm | minillm, unfused kernels | HF `generate` | vLLM 0.34 | minillm vs HF | minillm vs vLLM |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | uniform lengths | 46,897 | 31,525 | 3,108 | 41,418 | 15.1× | 113% |
| Qwen2.5-0.5B | lengths vary ±90% | 27,000 | 18,475 | 1,338 | 24,487 | 20.2× | 110% |
| Qwen2.5-7B | uniform lengths | 7,520 | 6,021 | 2,293 | 7,885 | 3.3× | 95% |
| Qwen2.5-7B | lengths vary ±90% | 5,567 | 4,609 | 534 | 7,239 | 10.4× | 77% |

- **On 0.5B minillm is faster than vLLM: 113% of its throughput, and 110% with variable lengths.** On 7B it reaches 95%, and 77% with variable lengths. It runs 3.3–20× faster than HF.
- **Fused kernels add 46–49% on 0.5B and 21–25% on 7B.** See the profile below.
- **Async scheduling** added 26–32% on 0.5B and 5–6% on 7B when it was introduced (measured against `--no-async-scheduling` before kernel fusion).
- **Variable lengths break static batching.** HF pads every batch to its longest prompt and decodes until its longest request ends, so it loses 57% of its throughput on 0.5B and 77% on 7B. The continuous-batching engines lose 41–42% on 0.5B; on 7B minillm loses 26% and vLLM 8%.
- **7B with variable lengths is the weakest case (77%).** Steps that mix prompt chunks with decodes run eagerly in minillm, while vLLM captures graphs for those steps too.
- **CUDA graphs add 46–66% on 0.5B and 5–7% on 7B** (measured before async scheduling and fused kernels; [`results/*-eager`](results)).

**Where a decode step's time goes** (`benchmarks/profile_step.py`; B requests decoding in lockstep after 512-token prompts, ms per step). *GPU* is the device time of the step's work alone, measured while a spin kernel keeps the GPU ahead of the CPU, so launch gaps don't count. *Engine CPU* is everything else the engine does per step: scheduling, building inputs, launching, applying tokens. Everything except the "GPU, unfused" column is with fused kernels.

| model | batch | GPU, unfused | GPU | engine CPU | step, sync | step, async |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1 | 2.16 | 1.10 | 0.36 | 1.29 | 1.11 |
| Qwen2.5-0.5B | 4 | 2.45 | 1.21 | 0.41 | 1.43 | 1.21 |
| Qwen2.5-0.5B | 16 | 2.61 | 1.26 | 0.55 | 1.60 | 1.28 |
| Qwen2.5-0.5B | 64 | 2.97 | 1.56 | 1.18 | 2.30 | 1.56 |
| Qwen2.5-0.5B | 256 | 4.06 | 2.44 | 2.92 | 4.65 | 2.93 |
| Qwen2.5-7B | 1 | 5.89 | 4.64 | 0.35 | 4.83 | 4.64 |
| Qwen2.5-7B | 4 | 6.30 | 4.74 | 0.44 | 4.97 | 4.74 |
| Qwen2.5-7B | 16 | 6.83 | 4.97 | 0.54 | 5.29 | 4.98 |
| Qwen2.5-7B | 64 | 7.78 | 5.64 | 1.02 | 6.20 | 5.56 |
| Qwen2.5-7B | 128 | 9.11 | 6.63 | 1.64 | 7.55 | 6.70 |

- **Fused kernels halve the GPU time of a small 0.5B batch (2.16 to 1.10 ms at batch 1) and cut 7B's by 21% (5.89 to 4.64 ms).** Kernel breakdown of one batch-1 decode step:

| model | kernels | launches | GPU time | GEMMs | attention | everything else |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | unfused | 1,153 | 2.19 ms | 0.61 ms | 0.21 ms | 1.37 ms (62%) |
| Qwen2.5-0.5B | fused | 328 | 1.08 ms | 0.62 ms | 0.21 ms | 0.25 ms (23%) |
| Qwen2.5-7B | unfused | 1,369 | 5.92 ms | 3.88 ms | 0.33 ms | 1.72 ms (29%) |
| Qwen2.5-7B | fused | 408 | 4.68 ms | 3.95 ms | 0.32 ms | 0.40 ms (9%) |

  Unfused, RMSNorm, residual adds, rotary embeddings, SiLU and dtype casts ran as about a thousand small kernels and took 62% of a 0.5B step. Fused, everything besides the weight GEMMs and attention takes 0.25 ms: the fused kernels themselves, the K/V cache writes, and sampling.
- **Async scheduling makes a step cost about max(CPU, GPU) instead of CPU + GPU.** With fusion the GPU part shrank enough that a large batch of a small model is now CPU-bound: at batch 256 on 0.5B the engine's CPU work (2.9 ms) exceeds the GPU's (2.4 ms), and the step takes the CPU's time. That is the first roadmap item.

**Online serving** (Poisson arrivals, streaming, fresh prompts at every rate, load from 4 client processes).

**Qwen2.5-0.5B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | minillm | 7.0 | 900 | 7 / 25 | 1.2 / 1.4 |
| 8 | minillm-unfused | 7.0 | 898 | 14 / 33 | 2.5 / 2.9 |
| 8 | vllm | 7.0 | 900 | 11 / 21 | 1.3 / 1.6 |
| 16 | minillm | 14.0 | 1,794 | 7 / 15 | 1.2 / 1.4 |
| 16 | minillm-unfused | 14.0 | 1,787 | 14 / 32 | 2.7 / 3.1 |
| 16 | vllm | 14.0 | 1,794 | 13 / 27 | 1.4 / 1.9 |
| 32 | minillm | 29.5 | 3,776 | 8 / 17 | 1.3 / 1.6 |
| 32 | minillm-unfused | 29.2 | 3,741 | 15 / 35 | 3.2 / 4.0 |
| 32 | vllm | 29.5 | 3,775 | 12 / 27 | 1.6 / 2.2 |
| 64 | minillm | 58.3 | 7,460 | 11 / 31 | 1.6 / 2.0 |
| 64 | minillm-unfused | 57.1 | 7,305 | 25 / 98 | 4.4 / 5.5 |
| 64 | vllm | 58.2 | 7,454 | 16 / 32 | 2.4 / 3.2 |
| 128 | minillm | 116.6 | 14,930 | 18 / 103 | 3.4 / 4.7 |
| 128 | minillm-unfused | 112.0 | 14,332 | 38 / 61 | 6.7 / 7.4 |
| 128 | vllm | 115.2 | 14,751 | 35 / 69 | 5.9 / 7.1 |
| burst | minillm | 282.1 | 36,111 | 82 / 152 | 2.7 / 3.5 |
| burst | minillm-unfused | 191.2 | 24,471 | 134 / 237 | 4.0 / 4.8 |
| burst | vllm | 187.1 | 23,948 | 150 / 203 | 4.2 / 4.7 |

**Qwen2.5-7B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | minillm | 3.5 | 449 | 20 / 75 | 4.8 / 5.2 |
| 4 | minillm-unfused | 3.5 | 449 | 29 / 49 | 6.6 / 7.3 |
| 4 | vllm | 3.5 | 449 | 22 / 37 | 5.0 / 5.3 |
| 8 | minillm | 7.0 | 894 | 21 / 34 | 5.0 / 5.5 |
| 8 | minillm-unfused | 7.0 | 892 | 31 / 49 | 7.2 / 8.1 |
| 8 | vllm | 7.0 | 894 | 23 / 36 | 5.2 / 5.6 |
| 16 | minillm | 14.6 | 1,875 | 23 / 43 | 5.6 / 6.6 |
| 16 | minillm-unfused | 14.6 | 1,863 | 35 / 64 | 8.6 / 10.3 |
| 16 | vllm | 14.6 | 1,874 | 24 / 58 | 5.7 / 6.6 |
| 32 | minillm | 28.7 | 3,676 | 29 / 76 | 7.5 / 10.4 |
| 32 | minillm-unfused | 28.3 | 3,624 | 53 / 148 | 13.7 / 18.2 |
| 32 | vllm | 28.7 | 3,671 | 29 / 65 | 7.2 / 8.5 |
| 64 | minillm | 49.0 | 6,276 | 127 / 270 | 18.6 / 20.1 |
| 64 | minillm-unfused | 41.4 | 5,296 | 245 / 449 | 21.6 / 23.4 |
| 64 | vllm | 54.0 | 6,906 | 91 / 200 | 16.6 / 17.7 |
| burst | minillm | 54.8 | 7,019 | 726 / 1446 | 13.1 / 18.0 |
| burst | minillm-unfused | 43.0 | 5,503 | 696 / 1777 | 17.6 / 21.8 |
| burst | vllm | 55.0 | 7,038 | 676 / 1452 | 13.2 / 17.1 |


- **Fused kernels roughly halve per-token latency on 0.5B at every load.** It falls from 2.5 to 1.2 ms at 8 req/s, 4.4 to 1.6 ms at 64 req/s and 6.7 to 3.4 ms at 128 req/s, and burst throughput rises 48%. On 7B, latency falls from 6.6 to 4.8 ms at 4 req/s and 13.7 to 7.5 ms at 32 req/s, and burst throughput rises 28%. Time to first token drops too (14 to 7 ms on 0.5B, 29 to 20 ms on 7B at light load): prompt steps run the fused kernels as well.
- **Against vLLM, 0.5B: minillm is ahead at every load.** Per-token latency is 1.18 vs 1.27 ms at 8 req/s, 1.60 vs 2.36 ms at 64 and 3.43 vs 5.86 ms at 128. Time to first token is 7 vs 11 ms at light load. Burst throughput is 36.1k vs 23.9k tok/s, 1.5× vLLM's figure in this job and 1.2× its best in any job.
- **Against vLLM, 7B: level at light load and under a burst, behind in between.** At 4 req/s per-token latency is 4.83 vs 4.98 ms and time to first token 20 vs 22 ms. Burst throughput is 7.02k vs 7.04k tok/s. At 32 req/s latency is 7.5 vs 7.2 ms, and at 64 req/s minillm serves 91% of vLLM's throughput (6.3k vs 6.9k tok/s) with higher TTFT (127 vs 91 ms).
- **Async scheduling's time-to-first-token cost under bursts remains.** When it was introduced, 7B's median burst TTFT rose from about 470 to 650 ms against synchronous scheduling, reproduced in five runs. Here it is 726 ms vs vLLM's 676. The engine alone does not show it (see the async-scheduling notes above), and its cause is still open. `--no-async-scheduling` restores synchronous scheduling for TTFT-critical bursty traffic.
- **Shared 1,024-token system prompt at 16 req/s:** with prefix caching, median TTFT is 7.5 ms (minillm) vs 10.8 ms (vLLM) on 0.5B, and 16.6 vs 17.9 ms on 7B.
- **A single client process skews the results.** With one client process, vLLM's 0.5B burst measured 9.8k tok/s; with four it measures 23–29k. One process tops out near 10k streamed tokens/s and inflates burst TTFT.
- **CUDA graphs matter online too:** without them, minillm's 0.5B per-token latency at light load was 11.4 ms instead of about 3 (measured before async scheduling and fused kernels).

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
  layers/attention.py, fused.py, sampler.py
  entrypoints/openai_server.py, protocol.py
benchmarks/  bench_offline.py, bench_serving.py, fake_server.py, profile_step.py
scripts/     report.py, readme_tables.py
scripts/explorer/  Slurm + Apptainer jobs: setup, GPU tests, benchmarks, profiling
results/     H200 benchmark results (JSON + REPORT.md per run)
tests/
Dockerfile   CUDA serving image (see Container)
```

## Roadmap

1. **Cut the engine's CPU time per step.** With the kernels fused, large batches of a small model are CPU-bound again: at batch 256 on 0.5B the engine spends 2.9 ms of CPU per step against 2.4 ms of GPU work. Most of it is Python building per-token input lists and FlashInfer page tables from scratch each step; keeping them in preallocated arrays and updating them incrementally, as vLLM does, would remove most of that.
2. **CUDA graphs for mixed steps.** Steps that contain prompt chunks still run eagerly, which is part of why the variable-length workload trails vLLM by more than the uniform one. vLLM captures everything except attention ("piecewise" graphs).
3. **Speculative decoding** (n-gram / prompt lookup first) and **4-bit weights**, using the W4A16 Triton kernel from Quantized-inference-lab.
4. **A GPU smoke test for the container image in CI.** CI builds the image and tests it on CPU; the GPU path has only been checked by hand (see Container).

## License

Apache License 2.0. See [LICENSE](LICENSE).
