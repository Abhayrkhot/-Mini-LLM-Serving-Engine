# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 2,800 lines of Python:

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

## Status

| | |
|---|---|
| **Verified** | 84 tests pass on CPU (10 more skip without a GPU or a downloaded model). Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption and prefix-cache hits. On a real checkpoint (Qwen2.5-0.5B-Instruct), all 288 of 288 generated tokens are HF's argmax, on both CPU and Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Written, not yet run** | The CUDA path: the flash-attn paged kernels and GPU memory profiling. `tests/test_gpu.py` and `scripts/gpu_validate.sh` cover it, but no CUDA machine has run them yet. |
| **Not built yet** | CUDA graphs, speculative decoding, quantized weights, multi-GPU (see the roadmap below). |

There are no GPU performance numbers yet. The only measurements so far are from a MacBook (see [Benchmarks](#benchmarks)).

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

**Attention** ([attention.py](minillm/layers/attention.py)). Every step's tokens are packed into one flat batch with no padding. Two backends:

- `torch`: gathers each sequence's blocks and runs SDPA with an explicit mask. It runs on CPU, MPS and CUDA, and the tests check it against dense attention.
- `flash_attn`: uses flash-attn's varlen and kv-cache kernels, which read the block table directly. CUDA only.

## Quickstart

```bash
pip install -e '.[dev]'            # on CUDA, also: pip install flash-attn --no-build-isolation
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
pytest -q                                   # 84 tests, ~13 s on a laptop CPU, no downloads
MINILLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m slow   # real-checkpoint parity
```

| Test file | What it proves |
|---|---|
| `test_model_parity.py` | The engine's greedy output equals Hugging Face's on randomly initialised Qwen2, Qwen3 and Llama 3 checkpoints. This holds under batching, chunked prefill (7-token budget), preemption (18-block cache) and prefix-cache hits, and also when the KV pool is pre-filled with NaN, so stale memory can never leak into another request. |
| `test_attention.py` | Paged attention equals dense causal attention, for blocks in shuffled order, grouped-query heads, decodes, full prompts and chunks on cached context. |
| `test_block_manager.py` | A Hypothesis state machine tracks the physical contents of every block. After any sequence of admit, decode and free operations, every computed token of every live sequence is still in place and reference counts match the tables. A deliberately broken allocator fails this test (mutation-checked). |
| `test_scheduler.py` | Token budget, chunk sizes, decodes never starved by long prompts, `max_num_seqs`, preemption preserving outputs, resume order, stop conditions, abort freeing blocks, prefix-cache accounting |
| `test_server.py` | Real uvicorn over HTTP: SSE format, usage, chat template, stop strings, 400/429 responses, 32 concurrent clients, aborts on streaming and non-streaming disconnect |
| `test_async_engine.py` | Streamed output equals offline output. Closing a stream aborts its request. An executor crash fails in-flight and new requests. |
| `test_gpu.py` | The flash-attn backend agrees with the torch backend (CUDA only) |
| `test_real_model.py` | On a real checkpoint, every generated token is HF's argmax. Teacher forcing is used so that one float near-tie can't send the comparison down a different continuation. |

The scheduler, server and async tests run on `FakeExecutor`, a deterministic stand-in for the GPU. Its next token depends only on the sequence's own tokens, so outputs must be identical however requests are batched, chunked or preempted. That's the same property the real engine needs.

## Benchmarks

Both harnesses send token-id prompts, so every engine sees byte-identical input:

- `benchmarks/bench_offline.py`: minillm, HF `generate` and vLLM, with identical prompts and per-request output lengths.
- `benchmarks/bench_serving.py`: a Poisson-arrival streaming load generator for any OpenAI-compatible server. It reports TTFT, TPOT, ITL and end-to-end latency (p50/p90/p99), throughput, and goodput under SLOs.

`scripts/gpu_validate.sh` does the full CUDA run and writes `REPORT.md` via `scripts/report.py`:

1. runs the test suite, including the GPU and real-model tests
2. runs the offline comparison on uniform and variable-length workloads
3. sweeps request rates against minillm and then vLLM with the same limits
4. runs a shared-system-prompt workload against both

`benchmarks/fake_server.py` serves the full HTTP stack over the fake executor with a latency cost model. CI uses it to load-test the server with no GPU.

### Dev-machine numbers (Apple M4, MPS, bfloat16)

These come from a MacBook, not a GPU server, so treat them as a sanity check. The gap they show is the reason for the CUDA work above.

Qwen2.5-0.5B-Instruct, 64 requests, about 256 tokens in and 128 out. HF runs static batches of 32. Throughput counts only the tokens each request asked for.

| workload | minillm | HF `generate` |
|---|---:|---:|
| uniform lengths (static batching's best case) | 197 tok/s | 264 tok/s |
| lengths vary ±90% (prompts and outputs) | 142 tok/s | 139 tok/s |

How to read this:

- **Variable lengths:** continuous batching recovers what static batching loses to padding and stragglers, and the two engines tie.
- **Uniform lengths:** HF wins. My torch attention backend gathers each sequence's whole context out of the paged cache on every layer and every step. HF reads a contiguous cache. This is the cost a paged attention kernel removes; on CUDA that kernel is flash-attn's.
- **What didn't help:** switching to SDPA's native GQA instead of copying K/V for every query head gave +17%. Bucketing shapes for MPS kernel reuse and removing the NaN guard made no measurable difference, so neither change was kept.

## Layout

```text
minillm/
  scheduler.py, block_manager.py, sequence.py, sampling.py   scheduling and memory (no torch)
  engine.py, async_engine.py, metrics.py, tokenizer.py       engine loop, streaming, metrics
  executor/model_runner.py   batch tensors, KV-cache sizing and allocation, sampling
  executor/fake.py           deterministic executor and byte tokenizer for tests and CI
  models/llama.py, loader.py model implementation, safetensors loading
  layers/attention.py, sampler.py
  entrypoints/openai_server.py, protocol.py
benchmarks/  bench_offline.py, bench_serving.py, fake_server.py
scripts/     gpu_validate.sh, report.py
tests/
```

## Roadmap

1. **CUDA validation and numbers.** Run `scripts/gpu_validate.sh` on an A100 or H200 and publish the report: offline and online throughput versus HF and vLLM, latency under load, and prefix-cache effect.
2. **CUDA graphs for decode.** Capture decode steps at a set of batch sizes, padding each batch up to the nearest one. At small batch sizes, kernel launch overhead dominates decode time.
3. **Engine core in its own process.** The scheduler and the HTTP layer currently share one Python process, so they compete for the GIL. Moving the engine behind a ZeroMQ socket removes that contention.
4. **Speculative decoding** (n-gram / prompt lookup first) and **4-bit weights**, using the W4A16 Triton kernel from Quantized-inference-lab.
5. **Container image** built and smoke-tested in CI on a GPU runner.
