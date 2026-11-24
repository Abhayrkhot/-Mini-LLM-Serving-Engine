# minillm: a small LLM serving engine

A from-scratch inference server for decoder-only LLMs (Llama 2/3, Mistral, Qwen2/2.5, Qwen3) in about 4,900 lines of Python (plus a vendored 4-bit Triton kernel):

- a paged KV cache with reference-counted blocks
- continuous batching with chunked prefill
- preemption by recompute
- prefix caching
- FlashInfer paged attention and fused norm, rotary and activation kernels
- CUDA graphs for decode, and async scheduling that overlaps CPU work with the GPU
- speculative decoding with n-gram drafts, and 4-bit weights (W4A16)
- an OpenAI-compatible streaming API with Prometheus metrics

It loads Hugging Face safetensors checkpoints directly and implements the model itself. `transformers` is used only for tokenizers and as the reference in tests.

**On one H200, minillm outperforms vLLM 0.34 on Qwen2.5-0.5B: 136–139% of its offline throughput, lower per-token latency at every load, and 1.32× its throughput when 500 requests arrive at once. On Qwen2.5-7B it matches vLLM: 100% of its offline throughput, and level online from light load to a burst.** On text that repeats its context, n-gram speculative decoding makes 7B generate 3.2× faster for a single request. 4-bit weights shrink 7B's weights from 14.2 to 5.2 GiB, raising WikiText-2 perplexity by 6%. minillm runs 2.3–16× faster than Hugging Face `generate`, and its greedy output is checked against Hugging Face's on the same hardware ([results](#results-on-an-h200)).

## Status

| | |
|---|---|
| **Verified on CPU / MPS** | 142 tests pass on a laptop CPU. Greedy output matches Hugging Face token for token on Qwen2, Qwen3 and Llama 3 (with RoPE scaling), through batching, chunked prefill, preemption, prefix-cache hits and speculative decoding. On Qwen2.5-0.5B-Instruct, 288 of 288 generated tokens are HF's argmax on CPU and on Apple MPS. The server is tested over real HTTP: streaming, stop strings, client disconnects, backpressure and engine crashes. |
| **Verified on an H200** | 207 tests pass, including the FlashInfer and flash-attn kernels against the reference attention (head dims 64 and 128, decodes, prompts, chunks on cached context, mixed steps split across FlashInfer's decode and prefill kernels), the engine with and without CUDA graphs under preemption, fused kernels against the PyTorch ops, async scheduling (same tokens as synchronous, and no device sync while launching a step), speculative decoding, the 4-bit kernel and engine, and Qwen2.5-0.5B: exact against HF in fp32, and in bf16 at least 97% of tokens are HF's argmax with only near-ties differing. |
| **Checked by hand only** | The container image on a GPU: run on an RTX 3050 Ti laptop GPU, not yet on an H200. CI builds it and tests it on CPU (see Container). |
| **Not built yet** | Multi-GPU, CUDA graphs for prompt and verification steps, calibrated 4-bit quantization (see the roadmap). |

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
- *Mirrored on the device* ([block_tables.py](minillm/executor/block_tables.py)): each running sequence's table has a row in a device tensor, updated incrementally. A sequence gains a block only every 16 tokens, so a step sends just the few new entries. Prompt chunks get their KV slots and positions from a few tensor ops, and long FlashInfer page lists are gathered on the device. Decode steps still compute their one slot per sequence on the CPU, which is cheaper there than extra kernels. At batch 256 on 0.5B this took the engine's CPU time per step from 2.9 to 1.8 ms.

**Attention** ([attention.py](minillm/layers/attention.py)). Every step's tokens are packed into one flat batch with no padding. Three backends:

- `torch`: gathers each sequence's blocks and runs SDPA with an explicit mask. It runs on CPU, MPS and CUDA, and the tests check it against dense attention.
- `flashinfer` (the default on CUDA): FlashInfer's paged prefill and decode kernels read blocks in place through the block table. FlashInfer plans each step once on the CPU, and every layer reuses that plan.
  - *Mixed steps:* a step can hold decodes (one new token each) and prompt chunks. The prefill kernel tiles over query rows, so a decode row wastes most of a tile. On Qwen2.5-7B, a step with 192 decodes and a 512-token chunk took 28 ms of GPU time that way, more than the two halves alone (8 + 11 ms). So the runner puts single-token rows first, and FlashInfer plans its decode kernel for those and its prefill kernel for the chunks, both writing into one output. That step now takes 17 ms. Together with the CPU savings above, this took 7B with variable-length requests from 77% to 100% of vLLM's offline throughput.
- `flash_attn`: flash-attn's varlen and kv-cache kernels with block tables (block size 256). No prebuilt wheel loads against the torch 2.14 the H200 runs use, so [`build_flash_attn.sh`](scripts/explorer/build_flash_attn.sh) builds flash-attn 2.8.3 from source; with it the backend's kernel and engine tests pass on the H200, and on Qwen2.5-0.5B its greedy tokens are HF's argmax. It runs without CUDA graphs and has not been benchmarked.

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
- *Long steps:* a request arriving while step n runs would wait for step n+1 too, which was planned before it arrived. When those steps are prompt-heavy (over 100 ms on 7B), that is a whole extra step before its first token. So the engine estimates each step's duration from the measured time per token of recent prompt-heavy steps, and does not plan past a step predicted to take over 20 ms. It collects that step and returns; the caller takes in new requests, and the next step includes them. Decode-heavy steps still overlap. With 128 clients streaming 512-token prompts to 7B, mean time to first token fell from about 700 to 560 ms at the same throughput.

**Speculative decoding** ([ngram.py](minillm/ngram.py)). When output repeats its context, as when editing code, quoting a document or filling a template, the next tokens can be guessed. Look up the most recent earlier occurrence of the sequence's last n tokens and propose what followed them. With `--speculative-tokens k`:

1. The scheduler drafts up to k tokens for each decoding greedy request, never past its length limits.
2. One step checks them all: the model's token comes back at every draft position.
3. Drafts are accepted while they equal the model's token. The first disagreement is replaced by the model's token, and if all agree, the token after the last draft comes free.
4. Positions holding rejected drafts stop counting as computed; their K/V is overwritten later.

Outputs are unchanged: sampled requests decode normally, and a randomized test checks greedy outputs against plain decoding across draft lengths, preemption and stop conditions. Each sequence keeps an n-gram index that grows with its tokens, so a lookup is O(1) instead of a scan of the context. A speculative step's token count is known only when it finishes, so speculation turns async scheduling off. Verification steps run eagerly, without CUDA graphs. Both costs show in the [results](#speculative-decoding): drafts pay off on 7B but not on 0.5B.

**4-bit weights** ([quant/](minillm/quant)). `--quantization w4a16` quantizes every decoder projection (q/k/v, o, gate/up, down) to 4-bit codes at load time: one scale and zero point per 128 input columns of each output row (asymmetric round-to-nearest, the GPTQ layout). The embedding, norms and LM head stay 16-bit. The projections run on the W4A16 kernel from [Quantized-inference-lab](https://github.com/Abhayrkhot/Quantized-inference-lab), which dequantizes inside the matmul:
- a Gluon tensor-core kernel for up to 32 rows;
- Triton or dequantize-then-cuBLAS above that, autotuned per shape.

A decode step reads a quarter of the weight bytes, and the freed memory goes to the KV cache. The kernel's error against fp32 equals cuBLAS's in bf16 (0.22%). Round-to-nearest without calibration costs some accuracy; [`eval_quant.py`](benchmarks/eval_quant.py) measures it as WikiText-2 perplexity and next-token agreement with the 16-bit model (results below).

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
pytest -q                                   # 142 tests, ~20 s on a laptop CPU, no downloads
MINILLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m slow   # real-checkpoint parity
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh  # on a GPU
# with flash-attn built (scripts/explorer/build_flash_attn.sh), the flash_attn tests run too:
EXTRA_PYTHONPATH=/scratch/$USER/minillm/overlay-fa srun ... bash scripts/explorer/tests.sh
```

| Test file | What it proves |
|---|---|
| `test_model_parity.py` | The engine's greedy output equals Hugging Face's on randomly initialised Qwen2, Qwen3 and Llama 3 checkpoints. This holds under batching, chunked prefill (7-token budget), preemption (18-block cache) and prefix-cache hits, and also when the KV pool is pre-filled with NaN, so stale memory can never leak into another request. Speculative decoding (4-token drafts on repetitive prompts, with preemption) also gives exactly HF's output. |
| `test_attention.py` | Paged attention equals dense causal attention, for blocks in shuffled order, grouped-query heads, decodes, full prompts and chunks on cached context. |
| `test_block_manager.py` | A Hypothesis state machine tracks the physical contents of every block. After any sequence of admit, decode and free operations, every computed token of every live sequence is still in place and reference counts match the tables. A deliberately broken allocator fails this test (mutation-checked). |
| `test_scheduler.py` | Token budget, chunk sizes, decodes never starved by long prompts, `max_num_seqs`, preemption preserving outputs, resume order, stop conditions, abort freeing blocks, prefix-cache accounting |
| `test_server.py` | Real uvicorn over HTTP, with every test run twice (engine in its own process, and on a thread): SSE format, usage, chat template, stop strings, 400/429 responses, 32 concurrent clients, aborts on streaming and non-streaming disconnect |
| `test_engine_core.py` | The engine process: streamed output equals offline output, metrics cross the process boundary, closing a stream aborts the request in the engine, invalid requests are rejected without killing it, and a crash, a SIGKILL or a failed startup is reported instead of hanging. |
| `test_async_scheduling.py` | Async scheduling generates exactly what synchronous scheduling does: a Hypothesis test runs 150 random workloads mixing chunked prefill, preemption, EOS, stop tokens, length limits and `max_num_seqs`. An EOS stop costs exactly one discarded step and a length limit none; aborts and preemption while a step is in flight leave other requests untouched. A deliberately broken scheduler fails it (mutation-checked). |
| `test_speculative.py` | Speculative decoding never changes what is generated: a Hypothesis test runs random workloads across draft lengths, token budgets, preemption, EOS and length limits against plain decoding. The fake executor models the KV cache slot by slot, so a rejected draft whose position were not rolled back would change the output (mutation-checked). Accepted drafts save steps, a stop token inside accepted drafts ends the request there, and sampled requests never draft. |
| `test_ngram.py` | The incremental n-gram index proposes what a brute-force scan of the context would, as tokens arrive one by one, preferring longer and more recent matches and never matching the suffix with itself. |
| `test_block_tables.py` | The device mirror of the block tables stays equal to the Python tables through growth, row reuse and preemption, and the KV slots and FlashInfer page indices computed on it equal the Python computation. A mirror that reuses a row without rewriting it fails it (mutation-checked). |
| `test_quant.py` | 4-bit round-to-nearest: every weight within half a quantization step, zero exactly representable, constant groups survive, and the GPTQ int32 packing round-trips. |
| `test_bench_serving.py` | The load generator sends as many requests at once as asked (a concurrency limit is a total over its worker processes), and reports the peak concurrency it reached. |
| `test_async_engine.py` | Streamed output equals offline output. Closing a stream aborts its request. An executor crash fails in-flight and new requests. |
| `test_gpu.py` | CUDA only. Each GPU kernel matches the torch reference on random paged batches. The engine in bf16, with and without CUDA graphs and under preemption, picks the fp32 model's argmax as often as the torch backend does in bf16. The fused norm, residual-add, rotary and SiLU kernels match the PyTorch ops within bf16 rounding, rotary included on strided q/k views out to 32k positions. Async and synchronous scheduling generate identical tokens, sampled ones included. With half a second of GPU work queued, launching a step still returns immediately, so nothing on the launch path waits for the GPU. Mixed-step attention split into decode and prefill kernels equals the torch reference, and page lists built on the device equal the host-built ones under preemption. Speculative decoding is as accurate as the reference backend. The W4A16 kernel matches a dequantized matmul across shapes, and the 4-bit engine picks its dequantized model's argmax as often as the torch backend does. |
| `test_real_model.py` | On a real checkpoint, every generated token is HF's argmax: exactly with the torch backend in fp32, and in bf16 for the GPU backends (at least 97%, with only near-ties differing). Teacher forcing is used so that one float near-tie can't send the comparison down a different continuation. |

The scheduler, server and async tests run on `FakeExecutor`, a deterministic stand-in for the GPU. Its next token depends only on the sequence's own tokens, so outputs must be identical however requests are batched, chunked or preempted. That's the same property the real engine needs.

## Benchmarks

Both harnesses send token-id prompts, so every engine sees byte-identical input:

- `benchmarks/bench_offline.py`: minillm, HF `generate` and vLLM on identical prompts and per-request output lengths. Each engine first warms up on 32 full-length prompts; the prefix cache is cleared before timing. HF runs with PyTorch's cuDNN attention turned off, since cuDNN plans anew for every batch shape it hasn't seen.
- `benchmarks/bench_serving.py`: a Poisson-arrival streaming load generator for any OpenAI-compatible server. It reports TTFT, TPOT, ITL and end-to-end latency (p50/p90/p99), throughput, and goodput under SLOs.
  - Every run draws fresh prompts, because an engine with prefix caching would otherwise serve repeated prompts from cache.
  - `--workers N` spreads the load across processes. One Python client tops out near 10k streamed tokens/s and inflates burst TTFT; the H200 runs use 4 workers.
  - `--max-concurrency` caps requests in flight, as a total over the workers. Without it, every request may be in flight at once.
- `benchmarks/bench_spec.py`: speculative decoding on output that repeats its context (minillm or vLLM, offline).
- `benchmarks/eval_quant.py`: the accuracy cost of 4-bit weights, as WikiText-2 perplexity and next-token agreement with the 16-bit model.
- `benchmarks/profile_step.py`: GPU and CPU time per decode step, with a per-kernel breakdown.

`benchmarks/fake_server.py` serves the full HTTP stack over the fake executor with a latency cost model. CI uses it to load-test the server with no GPU.

### Results on an H200

One NVIDIA H200 per job, running the vLLM 0.34 container (torch 2.14, CUDA 13, FlashInfer 0.7.3), bf16.

- **minillm:** FlashInfer attention and fused kernels, decode CUDA graphs and async scheduling (the defaults). Two variants:
  - `minillm, sync scheduling` adds `--no-async-scheduling`;
  - `minillm, 4-bit` adds `--quantization w4a16`.
- **vLLM:** defaults (CUDA graphs, prefix caching, chunked prefill).
- **Shared limits:** both engines get 256 concurrent sequences and an 8,192-token step budget.
- **HF `generate`:** static batches of 64, with PyTorch's cuDNN attention kernel turned off (see below).
- **Workload:** random token-id prompts of about 512 tokens with 128 output tokens (EOS ignored), 500 requests per measurement.

The minillm variants and vLLM ran back to back on the same GPU, one job per model, and HF in separate jobs. These are single runs on a shared cluster, so treat differences under about 10% as noise. For example, vLLM's 0.5B offline figure has measured from 34k to 41k tok/s across five jobs. Correctness is checked against Hugging Face (see Testing), not against vLLM. Raw JSON and per-run reports are in [`results/`](results): `h200-final-*` for the tables below, and earlier runs alongside.

Two fixes to the offline benchmark changed earlier figures:

- **HF `generate` paid for cuDNN's kernel planning.** On the H200, PyTorch's attention uses a cuDNN kernel that builds a plan for each new input shape. In isolation, a 64 × 512-token batch of 0.5B took 6.0 s the first time and 1.4 s after. In the benchmark, uniform 0.5B batches took 2.6 s each in the first published run and 5.4–6.8 s in later ones; with cuDNN attention off, every batch takes 1.36 s. The HF figures below are with it off, 1.05–1.9× the first published ones, so minillm's lead over HF is smaller than first reported (3.4–24×).
- **vLLM once got different prompts when lengths varied.** Token ids were drawn below each engine's own vocabulary size, and vLLM reports a smaller one than minillm (151,643 vs 151,936 for Qwen2.5), so the random stream, and with it the prompt lengths, diverged. Every engine now gets byte-identical prompts. Earlier ±90% figures, including the first H200 run's, carry this error.

**Offline throughput** (output tokens/s; HF counts only the tokens each request asked for):

| model | workload | minillm | minillm, 4-bit | HF `generate` | vLLM 0.34 | minillm vs HF | minillm vs vLLM | 4-bit vs 16-bit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | uniform lengths | 56,996 | 47,160 | 5,884 | 40,860 | 9.7× | 139% | 83% |
| Qwen2.5-0.5B | lengths vary ±90% | 31,082 | 22,961 | 2,001 | 22,833 | 15.5× | 136% | 74% |
| Qwen2.5-7B | uniform lengths | 7,881 | 6,892 | 3,417 | 7,881 | 2.3× | 100% | 87% |
| Qwen2.5-7B | lengths vary ±90% | 6,922 | 6,019 | 560 | 6,951 | 12.4× | 100% | 87% |

- **On 0.5B minillm is faster than vLLM: 139% of its throughput, and 136% with variable lengths.** On 7B the two are level on both workloads. minillm runs 2.3–16× faster than HF.
- **What closed the gap since the first H200 run** (113%, 110%, 95% and 77% of vLLM; the two variable-length figures predate the prompt fix above):
  - building step inputs on the device and other CPU savings (0.5B, large batches);
  - splitting mixed steps across FlashInfer's decode and prefill kernels (7B with variable lengths, 77% to 100%).
- **Earlier measurements of single features** ([`results/h200-qwen2.5-*`](results)):
  - fused kernels added 46–49% on 0.5B and 21–25% on 7B;
  - async scheduling added 26–32% on 0.5B and 5–6% on 7B, measured before kernel fusion;
  - CUDA graphs added 46–66% on 0.5B and 5–7% on 7B, measured before async scheduling and fused kernels ([`results/*-eager`](results)).
- **Variable lengths break static batching.** HF pads every batch to its longest prompt and decodes until its longest request ends, so it loses 66% of its throughput on 0.5B and 84% on 7B. The continuous-batching engines lose 44–45% on 0.5B and 12% on 7B (with 4-bit weights, 51% and 13%).
- **4-bit weights cost 13–26% offline.** Offline batches are large, and the W4A16 kernel only beats 16-bit cuBLAS at small batch sizes (see the profile below).

**Where a decode step's time goes** (`benchmarks/profile_step.py`; B requests decoding in lockstep after 512-token prompts, ms per step). *GPU* is the device time of the step's work alone. It is measured while a spin kernel keeps the GPU ahead of the CPU, so launch gaps don't count. *Engine CPU* is everything else the engine does per step: scheduling, building inputs, launching, applying tokens.

| model | batch | GPU | GPU, 4-bit | engine CPU | step, sync | step, async | step, async, 4-bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1 | 1.11 | 1.74 | 0.39 | 1.31 | 1.11 | 1.74 |
| Qwen2.5-0.5B | 4 | 1.21 | 1.88 | 0.43 | 1.45 | 1.22 | 1.89 |
| Qwen2.5-0.5B | 16 | 1.27 | 2.08 | 0.55 | 1.59 | 1.27 | 2.08 |
| Qwen2.5-0.5B | 64 | 1.56 | 2.45 | 0.96 | 2.20 | 1.55 | 2.43 |
| Qwen2.5-0.5B | 256 | 2.48 | 3.16 | 1.80 | 3.70 | 2.33 | 3.02 |
| Qwen2.5-7B | 1 | 4.63 | 3.63 | 0.38 | 4.84 | 4.64 | 3.64 |
| Qwen2.5-7B | 4 | 4.74 | 3.60 | 0.42 | 4.97 | 4.74 | 3.60 |
| Qwen2.5-7B | 16 | 4.97 | 4.87 | 0.55 | 5.28 | 4.97 | 4.86 |
| Qwen2.5-7B | 64 | 5.64 | 8.20 | 1.01 | 6.13 | 5.54 | 8.11 |
| Qwen2.5-7B | 128 | 6.66 | 11.12 | 1.18 | 7.20 | 6.74 | 10.93 |

- **Async scheduling makes a step cost about max(CPU, GPU) instead of CPU + GPU.** At batch 256 on 0.5B, a step takes 2.3 ms instead of 3.7.
- **The engine is GPU-bound at every batch size.** The largest case is batch 256 on 0.5B, where the CPU work is 1.8 ms against 2.5 ms of GPU work. Before the CPU savings, that CPU work was 2.9 ms, more than the GPU work.
- **Fused kernels halve the GPU time of a small 0.5B batch (2.2 to 1.1 ms) and cut 7B's by 21% (5.9 to 4.7 ms).** Kernel breakdown of one batch-1 decode step ([`results/h200-profile`](results/h200-profile)):

| model | kernels | launches | GPU time | GEMMs | attention | everything else |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | unfused | 1,153 | 2.19 ms | 0.61 ms | 0.21 ms | 1.37 ms (62%) |
| Qwen2.5-0.5B | fused | 328 | 1.08 ms | 0.62 ms | 0.21 ms | 0.25 ms (23%) |
| Qwen2.5-7B | unfused | 1,369 | 5.92 ms | 3.88 ms | 0.33 ms | 1.72 ms (29%) |
| Qwen2.5-7B | fused | 408 | 4.68 ms | 3.95 ms | 0.32 ms | 0.40 ms (9%) |

  Unfused, RMSNorm, residual adds, rotary embeddings, SiLU and dtype casts ran as about a thousand small kernels and took 62% of a 0.5B step. Fused, everything besides the weight GEMMs and attention takes 0.25 ms: the fused kernels themselves, the K/V cache writes, and sampling.
- **4-bit weights speed up 7B decoding at small batches and slow everything else.** At batch 1–4, a 7B step takes 22–24% less GPU time (4.63 to 3.63 ms at batch 1), because the projections read a quarter of the bytes. At batch 16 the two are even. From batch 64 up, 4-bit is slower (8.2 vs 5.6 ms): above 32 rows, the Gluon kernel hands over to slower paths. On 0.5B, 4-bit is slower at every batch size, since its matrices are too small for the weight reads to dominate. Kernel breakdown at batch 1:

| model | weights | launches | GPU time | GEMMs | attention | everything else |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 16-bit | 328 | 1.09 ms | 0.62 ms | 0.21 ms | 0.26 ms (24%) |
| Qwen2.5-0.5B | 4-bit | 328 | 1.57 ms | 1.07 ms | 0.21 ms | 0.28 ms (18%) |
| Qwen2.5-7B | 16-bit | 408 | 4.67 ms | 3.94 ms | 0.32 ms | 0.41 ms (9%) |
| Qwen2.5-7B | 4-bit | 380 | 3.45 ms | 2.71 ms | 0.32 ms | 0.42 ms (12%) |

**Online serving** (Poisson arrivals, streaming, fresh prompts at every rate, load from 4 client processes). A *burst* sends all 500 requests at once.

**Qwen2.5-0.5B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | minillm | 7.0 | 900 | 8 / 32 | 1.2 / 1.5 |
| 8 | minillm, sync scheduling | 7.0 | 899 | 8 / 24 | 1.5 / 1.8 |
| 8 | minillm, 4-bit | 7.0 | 899 | 14 / 34 | 1.9 / 2.3 |
| 8 | vLLM | 7.0 | 900 | 11 / 39 | 1.4 / 1.7 |
| 16 | minillm | 14.0 | 1,794 | 9 / 17 | 1.3 / 1.5 |
| 16 | minillm, sync scheduling | 14.0 | 1,792 | 8 / 14 | 1.6 / 1.9 |
| 16 | minillm, 4-bit | 14.0 | 1,790 | 14 / 32 | 2.1 / 2.6 |
| 16 | vLLM | 14.0 | 1,793 | 12 / 150 | 1.4 / 1.8 |
| 32 | minillm | 29.5 | 3,777 | 9 / 23 | 1.4 / 1.8 |
| 32 | minillm, sync scheduling | 29.4 | 3,764 | 9 / 19 | 2.0 / 2.5 |
| 32 | minillm, 4-bit | 29.4 | 3,757 | 16 / 38 | 2.6 / 3.6 |
| 32 | vLLM | 29.5 | 3,773 | 13 / 27 | 1.8 / 2.4 |
| 64 | minillm | 58.3 | 7,456 | 13 / 25 | 1.7 / 2.3 |
| 64 | minillm, sync scheduling | 57.8 | 7,404 | 10 / 34 | 2.8 / 3.6 |
| 64 | minillm, 4-bit | 57.6 | 7,369 | 24 / 45 | 3.8 / 4.9 |
| 64 | vLLM | 58.2 | 7,446 | 16 / 32 | 2.4 / 3.1 |
| 128 | minillm | 116.5 | 14,910 | 22 / 130 | 3.5 / 5.3 |
| 128 | minillm, sync scheduling | 115.4 | 14,768 | 16 / 118 | 5.8 / 7.0 |
| 128 | minillm, 4-bit | 113.8 | 14,564 | 33 / 116 | 7.7 / 9.6 |
| 128 | vLLM | 115.4 | 14,774 | 33 / 69 | 5.8 / 6.8 |
| burst | minillm | 252.1 | 32,265 | 397 / 1371 | 7.1 / 8.4 |
| burst | minillm, sync scheduling | 229.2 | 29,342 | 456 / 1499 | 7.6 / 8.6 |
| burst | minillm, 4-bit | 230.3 | 29,483 | 418 / 1444 | 7.9 / 9.1 |
| burst | vLLM | 190.8 | 24,426 | 431 / 2096 | 9.9 / 10.2 |

**Qwen2.5-7B, online** (500 requests per point)

| offered req/s | engine | served req/s | output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | minillm | 3.5 | 449 | 20 / 38 | 4.8 / 5.1 |
| 4 | minillm, sync scheduling | 3.5 | 449 | 18 / 36 | 5.3 / 5.7 |
| 4 | minillm, 4-bit | 3.5 | 450 | 23 / 43 | 3.7 / 4.3 |
| 4 | vLLM | 3.5 | 449 | 22 / 75 | 5.0 / 5.3 |
| 8 | minillm | 7.0 | 894 | 21 / 35 | 5.0 / 5.5 |
| 8 | minillm, sync scheduling | 7.0 | 894 | 18 / 34 | 5.5 / 6.2 |
| 8 | minillm, 4-bit | 7.0 | 896 | 24 / 43 | 4.1 / 5.5 |
| 8 | vLLM | 7.0 | 894 | 23 / 68 | 5.2 / 5.8 |
| 16 | minillm | 14.6 | 1,875 | 23 / 43 | 5.5 / 6.4 |
| 16 | minillm, sync scheduling | 14.6 | 1,872 | 20 / 42 | 6.3 / 7.4 |
| 16 | minillm, 4-bit | 14.7 | 1,882 | 28 / 54 | 6.1 / 8.0 |
| 16 | vLLM | 14.6 | 1,874 | 24 / 90 | 5.7 / 6.6 |
| 32 | minillm | 28.7 | 3,676 | 26 / 68 | 6.9 / 8.6 |
| 32 | minillm, sync scheduling | 28.6 | 3,667 | 23 / 72 | 8.4 / 10.8 |
| 32 | minillm, 4-bit | 28.9 | 3,701 | 38 / 86 | 11.7 / 15.3 |
| 32 | vLLM | 28.7 | 3,671 | 28 / 67 | 7.1 / 8.5 |
| 64 | minillm | 55.9 | 7,161 | 102 / 282 | 28.9 / 32.3 |
| 64 | minillm, sync scheduling | 54.8 | 7,010 | 151 / 435 | 30.7 / 33.1 |
| 64 | minillm, 4-bit | 50.9 | 6,520 | 198 / 906 | 34.8 / 36.4 |
| 64 | vLLM | 55.9 | 7,157 | 108 / 219 | 27.8 / 31.5 |
| burst | minillm | 60.2 | 7,705 | 2966 / 6946 | 30.2 / 30.4 |
| burst | minillm, sync scheduling | 55.4 | 7,088 | 3502 / 7570 | 31.6 / 32.0 |
| burst | minillm, 4-bit | 51.4 | 6,576 | 3338 / 7886 | 35.3 / 35.8 |
| burst | vLLM | 59.1 | 7,563 | 3038 / 7075 | 29.2 / 30.4 |

- **Against vLLM, 0.5B: minillm has lower per-token latency at every load.** It is 1.2 vs 1.4 ms at 8 req/s, 1.7 vs 2.4 ms at 64 and 3.5 vs 5.8 ms at 128. Median time to first token is lower at every load too, for example 8 vs 11 ms at 8 req/s. Tail TTFT is lower everywhere except at 128 req/s (p99 130 vs 69 ms). Under a burst minillm serves 32.3k vs 24.4k tok/s (1.32×), with p99 TTFT of 1.37 vs 2.10 s.
- **Against vLLM, 7B: level at every load.** Per-token latency is 4.8 vs 5.0 ms at 4 req/s and 6.9 vs 7.1 ms at 32. At 64 req/s both serve 7.16k tok/s; minillm's median TTFT is lower (102 vs 108 ms) and its p99 higher (282 vs 219 ms). Under a burst it serves 7.71k vs 7.56k tok/s, with median TTFT of 2.97 vs 3.04 s. In the first H200 run, minillm served 91% of vLLM's throughput at 64 req/s; that gap has closed.
- **Async vs synchronous scheduling:**
  - *Per-token latency:* async is lower at every load, for example 1.2 vs 1.5 ms on 0.5B at 8 req/s and 4.8 vs 5.3 ms on 7B at 4 req/s.
  - *Burst:* async serves 10% more on 0.5B and 9% more on 7B, with lower median TTFT (397 vs 456 ms on 0.5B, 2.97 vs 3.50 s on 7B).
  - *Light and moderate loads* (up to 128 req/s on 0.5B, up to 32 on 7B): synchronous scheduling's median TTFT is 0–6 ms lower. A request arriving mid-step waits for the step already planned behind the running one. At 64 req/s on 7B, async's is lower (102 vs 151 ms).
- **Bursts here are true bursts.** Burst figures in earlier versions of this README had at most 128 requests in flight: the load generator divided its concurrency limit by the number of worker processes twice. That made the burst a closed loop of 128 clients, and its TTFT hundreds of milliseconds. With all 500 requests in flight, 7B is GPU-bound. The prompts alone are about 260k tokens, and a per-step trace shows 5.7 s of the 8.2 s burst spent on steps with prompt chunks. Under the closed loop, async scheduling had cost about 180 ms of median TTFT, because it planned a step ahead of each long prompt step. The [long-step rule](#architecture) above cut mean TTFT under that load from about 700 to 560 ms.
- **4-bit weights online:** on 7B at light load they cut per-token latency from 4.8 to 3.7 ms (4 req/s). From 16 req/s up they raise it, and burst throughput falls by 15%. On 0.5B they are slower at every load.
- **Shared 1,024-token system prompt at 16 req/s:** with prefix caching, median TTFT is 8.6 ms (minillm) vs 10.4 ms (vLLM) on 0.5B, and 15.7 vs 18.0 ms on 7B.
- **A single client process skews the results.** One process tops out near 10k streamed tokens/s and inflates burst TTFT; with one, vLLM's 0.5B burst once measured 9.8k tok/s.
- **CUDA graphs matter online too:** without them, minillm's 0.5B per-token latency at light load was 11.4 ms instead of about 3 (measured before async scheduling and fused kernels).

#### Speculative decoding

`benchmarks/bench_spec.py` has 64 prompts that each ask the model to repeat a 256-token passage of this repository's source verbatim. Decoding is greedy, up to 288 output tokens, through the offline API. minillm runs at 1, 8 and 32 requests at a time, and vLLM at 1 and 32. Drafts are up to 4 tokens, from n-grams of up to 3 tokens, for both engines.

| model | engine | concurrency | no drafts (tok/s) | 4-token drafts (tok/s) | speedup | drafts accepted | outputs identical |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | minillm | 1 | 897 | 724 | 0.81× | 65% | 84% |
| Qwen2.5-0.5B | minillm | 8 | 5,309 | 4,570 | 0.86× | 66% | 86% |
| Qwen2.5-0.5B | minillm | 32 | 13,236 | 8,231 | 0.62× | 65% | 91% |
| Qwen2.5-7B | minillm | 1 | 217 | 696 | 3.20× | 82% | 100% |
| Qwen2.5-7B | minillm | 8 | 1,626 | 4,215 | 2.59× | 82% | 97% |
| Qwen2.5-7B | minillm | 32 | 5,455 | 8,822 | 1.62× | 82% | 98% |
| Qwen2.5-7B | vLLM | 1 | 214 | 844 | 3.94× | – | 100% |
| Qwen2.5-7B | vLLM | 32 | 5,406 | 12,069 | 2.23× | – | 100% |

- **On 7B, drafts make generation 3.2× faster for one request and 1.6× at 32.** 82% of drafted tokens are accepted. The gain shrinks with concurrency because larger batches already keep the GPU busy.
- **vLLM gains more (3.9× and 2.2×).** Its verification steps run in CUDA graphs and keep async scheduling. minillm's run eagerly and turn async off.
- **On 0.5B, drafts slow generation down.** A 0.5B decode step takes about 1.1 ms in a CUDA graph. An eager verification step costs more than the tokens it saves.
- **Outputs identical:** in bf16, 84–100% of outputs equal plain decoding token for token. The rest diverge at a near-tie, because a verification step runs FlashInfer's prefill kernel and a plain decode step its decode kernel, which round differently. In fp32 with the `torch` backend (0.5B, [`results/h200-spec/fp32`](results/h200-spec/fp32)), all 64 outputs are identical at every concurrency, and drafts make generation 2.6–3.3× faster. There every step runs eagerly, so a verification step costs about as much as a decode step. The randomized test in `test_speculative.py` checks the algorithm itself for exactness.

#### 4-bit weights

| model | weights, 16-bit → 4-bit | KV-cache tokens the saving holds | WikiText-2 perplexity, 16-bit → 4-bit | next-token agreement |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.92 → 0.43 GiB | 43k | 14.26 → 17.06 (+20%) | 76.8% |
| Qwen2.5-7B | 14.19 → 5.19 GiB | 169k | 7.46 → 7.89 (+6%) | 87.6% |

- **Memory:** 7B's weights shrink from 14.2 to 5.2 GiB. The embedding and LM head stay 16-bit and are 2 GiB of what is left. The 9 GiB saved holds 169k more tokens of KV cache, at 56 KiB per token.
- **Accuracy** ([`eval_quant.py`](benchmarks/eval_quant.py): the WikiText-2 test set, 299k tokens, scored in 2,048-token windows):
  - On 7B, perplexity rises 6%, and the 4-bit model's top token equals the 16-bit model's at 87.6% of positions.
  - On 0.5B the cost is larger: perplexity rises 20%, and the top tokens agree at 76.8% of positions.

  The evaluation runs a Hugging Face model with the same 4-bit weights, dequantized. `test_gpu.py` checks that the engine's 4-bit path matches such a model. This is round-to-nearest without calibration; GPTQ- or AWQ-style calibration is on the roadmap.
- **Speed:** 4-bit pays off only where decoding is bound by weight reads, which here means 7B at small batch sizes (see the profile and online results above). Elsewhere the 16-bit engine is faster, so 4-bit is mainly for fitting a model, or more KV cache, into less memory.

### Reproducing

`scripts/explorer/` runs everything on a Slurm cluster from the vLLM container image (vLLM 0.34, torch 2.14, CUDA 13, FlashInfer 0.7.3), so minillm, vLLM and HF share one torch build. Set `MINILLM_SIF` and `HF_HOME` if your paths differ from the defaults. The commands below are for 7B. For 0.5B, use rates `8 16 32 64 128 inf` and profile batch sizes `1 4 16 64 256`.

```bash
srun -p short -c 4 --mem=16G -t 20 bash scripts/explorer/setup.sh          # once: test-only packages, pyarrow
srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh
M=Qwen/Qwen2.5-7B-Instruct TAG=h200-final-qwen2.5-7b
ENGINES="minillm minillm-w4a16 vllm hf" SERVE_ENGINES="minillm minillm-sync minillm-w4a16 vllm" \
  sbatch -p gpu --gres=gpu:h200:1 -c 8 --mem=64G -t 4:00:00 \
  scripts/explorer/bench.sh $M $TAG 4 8 16 32 64 inf                        # offline + online
gpu() { srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 60 "$@"; }
for q in "" w4a16; do                                                       # step profile, 16- and 4-bit
  PROFILE_NAME=profile${q:+-$q} PROFILE_ARGS="--batch-sizes 1 4 16 64 128 --kernels 1 ${q:+--quantization $q}" \
    gpu bash scripts/explorer/profile.sh $M $TAG
done
for f in "" --no-fused-kernels; do                                          # fused vs unfused kernels
  PROFILE_NAME=qwen2.5-7b${f:+-unfused} PROFILE_ARGS="--batch-sizes 1 4 16 64 128 --kernels 1 $f" \
    gpu bash scripts/explorer/profile.sh $M h200-profile
done
run() { gpu bash -c "source scripts/explorer/common.sh && img python3 $*"; }
run benchmarks/eval_quant.py --model $M --output \$ROOT/results/$TAG/eval_quant.json  # WikiText-2 must be in the HF cache
run benchmarks/bench_spec.py --model $M --draft 0 4 --concurrency 1 8 32 --output \$ROOT/results/h200-spec/minillm_7b.json
run benchmarks/bench_spec.py --engine vllm --model $M --draft 0 4 --concurrency 1 32 \
  --output \$ROOT/results/h200-spec/vllm_7b.json
run benchmarks/bench_spec.py --model Qwen/Qwen2.5-0.5B-Instruct --draft 0 4 --concurrency 1 8 32 \
  --dtype float32 --attention-backend torch --output \$ROOT/results/h200-spec/fp32/minillm_0.5b_fp32.json
```

- **Where results go:** everything is written to `$ROOT/results/<tag>/`, where `$ROOT` defaults to `/scratch/$USER/minillm`. Each benchmark run writes JSON per measurement and a `REPORT.md` (via `scripts/report.py`).
- **Building the tables:** copy the JSON, `REPORT.md` and `nvidia-smi.txt` into this repository's `results/<tag>/` and run `python scripts/readme_tables.py`. The logs stay behind, since they contain cluster paths.
- **`memory.json`:** each `h200-final-*` directory has one, written by hand from the `Loaded ... GiB of weights` lines in `offline_minillm*_r0.log`. The 4-bit table uses it.
- **Speculation passages:** `bench_spec.py` builds them from the checked-out README and source, so editing those changes its workload slightly.

### Dev-machine numbers (Apple M4, MPS)

On a MacBook the portable `torch` backend runs Qwen2.5-0.5B at 197 tok/s on uniform lengths (HF `generate`: 264) and 142 tok/s when lengths vary ±90% (HF: 139). That backend gathers each sequence's context out of the paged cache every layer and step, which is the cost FlashInfer's paged kernels remove on CUDA. SDPA's native GQA gave +17% there; shape bucketing and dropping the NaN guard made no measurable difference, so neither was kept.

## Layout

```text
minillm/
  scheduler.py, block_manager.py, sequence.py, sampling.py   scheduling and memory (no torch)
  ngram.py                   n-gram drafts for speculative decoding
  engine.py, engine_core.py, async_engine.py                 engine loop, engine process, streaming
  metrics.py, tokenizer.py
  executor/model_runner.py   batch tensors, KV-cache sizing and allocation, sampling
  executor/block_tables.py   block tables mirrored on the device
  executor/cuda_graphs.py    decode CUDA graphs (FlashInfer)
  executor/fake.py           deterministic executor and byte tokenizer for tests and CI
  models/llama.py, loader.py model implementation, safetensors loading
  layers/attention.py, fused.py, sampler.py
  quant/                     4-bit quantization at load; w4a16.py and packing.py are the
                             W4A16 kernel vendored from Quantized-inference-lab
  entrypoints/openai_server.py, protocol.py
benchmarks/  bench_offline.py, bench_serving.py, bench_spec.py, eval_quant.py,
             fake_server.py, profile_step.py
scripts/     report.py, readme_tables.py
scripts/explorer/  Slurm + Apptainer jobs: setup, GPU tests, benchmarks, profiling, flash-attn build
results/     H200 benchmark results (JSON + REPORT.md per run)
tests/
Dockerfile   CUDA serving image (see Container)
```

## Roadmap

1. **CUDA graphs for prompt and verification steps.** Steps that contain prompt chunks or check speculative drafts still run eagerly. vLLM captures everything except attention ("piecewise" graphs). That is most of why its speculation gains more than minillm's (844 vs 696 tok/s for one request on 7B) and why drafts don't pay off on 0.5B.
2. **Speculation with async scheduling.** The step after a verification step is planned only once its accepted count is known. Planning it for the full draft length and trimming afterwards would let the two overlap.
3. **Calibrated 4-bit weights and a faster mid-size kernel.** GPTQ- or AWQ-style calibration should cut most of the perplexity cost of round-to-nearest, especially on small models. A W4A16 kernel tuned for 32–256 rows would stop 4-bit losing at medium batch sizes.
4. **Multi-GPU** (tensor parallelism), for models that don't fit on one GPU.
5. **A GPU smoke test for the container image in CI.** CI builds the image and tests it on CPU; the GPU path has only been checked by hand (see Container).

## License

Apache License 2.0. See [LICENSE](LICENSE).
