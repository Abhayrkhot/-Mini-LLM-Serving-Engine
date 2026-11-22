#!/usr/bin/env bash
# Validate and benchmark minillm on a CUDA machine, against HF generate and vLLM.
#
# On a Slurm cluster, grab a GPU first, for example:
#   srun --partition=gpu --gres=gpu:a100:1 --mem=64G --time=03:00:00 --pty bash
# then from the repo root:
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct bash scripts/gpu_validate.sh
#
# Results land in results/raw/<date>-<gpu>/ and a markdown summary in REPORT.md there.
# vLLM is installed into its own virtualenv because it pins its own torch.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-500}
RATES=${RATES:-"2 4 8 16 32 inf"}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT=${OUT:-results/raw/$(date +%Y%m%d)-${GPU}}
PORT=8000
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"

echo "== environments"
if [ ! -d .venv-gpu ]; then
  python -m venv .venv-gpu
  .venv-gpu/bin/pip install -q -e '.[dev,bench]'
  .venv-gpu/bin/pip install -q flash-attn --no-build-isolation
fi
if [ ! -d .venv-vllm ]; then
  python -m venv .venv-vllm
  .venv-vllm/bin/pip install -q vllm httpx
fi
M=.venv-gpu/bin/python
V=.venv-vllm/bin/python

echo "== tests (CPU suite + CUDA/flash-attn checks + real-model parity)"
$M -m pytest -q | tee "$OUT/pytest.txt"
MINILLM_TEST_MODEL=$MODEL MINILLM_TEST_DEVICE=cuda $M -m pytest -q -m slow | tee -a "$OUT/pytest.txt"

echo "== offline throughput"
for ratio in 0 0.9; do
  common=(--model "$MODEL" --num-prompts "$NUM_PROMPTS" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --range-ratio "$ratio")
  $M benchmarks/bench_offline.py --engine minillm "${common[@]}" --output "$OUT/offline_minillm_r${ratio}.json"
  $M benchmarks/bench_offline.py --engine hf "${common[@]}" --hf-batch-size 64 --output "$OUT/offline_hf_r${ratio}.json"
  $V benchmarks/bench_offline.py --engine vllm "${common[@]}" --output "$OUT/offline_vllm_r${ratio}.json"
done

wait_ready() {
  for _ in $(seq 600); do curl -sf "localhost:$PORT/health" >/dev/null && return 0; sleep 1; done
  echo "server did not come up" >&2; return 1
}

sweep() {  # $1 = label, $2 = python with httpx
  for rate in $RATES; do
    $2 benchmarks/bench_serving.py --base-url "http://localhost:$PORT" --model "$MODEL" \
      --label "$1" --num-prompts "$NUM_PROMPTS" --request-rate "$rate" \
      --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --range-ratio 0.5 --vocab-size 32000 \
      --output "$OUT/serve_${1}_rate${rate}.json"
  done
  # Shared 1024-token system prompt: exercises the prefix cache.
  $2 benchmarks/bench_serving.py --base-url "http://localhost:$PORT" --model "$MODEL" \
    --label "$1" --num-prompts "$NUM_PROMPTS" --request-rate 16 --prefix-len 1024 \
    --input-len 128 --output-len "$OUTPUT_LEN" --vocab-size 32000 \
    --output "$OUT/serve_${1}_prefix.json"
}

echo "== online serving: minillm"
.venv-gpu/bin/minillm-serve --model "$MODEL" --port $PORT --max-num-seqs 256 \
  --max-num-batched-tokens 8192 --log-level warning > "$OUT/minillm_server.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
wait_ready && sweep minillm $M
kill $SERVER; wait $SERVER 2>/dev/null || true

echo "== online serving: vLLM (same limits)"
.venv-vllm/bin/vllm serve "$MODEL" --port $PORT --max-num-seqs 256 \
  --max-num-batched-tokens 8192 --disable-log-requests > "$OUT/vllm_server.log" 2>&1 &
SERVER=$!
wait_ready && sweep vllm $V
kill $SERVER; wait $SERVER 2>/dev/null || true

$M scripts/report.py "$OUT" | tee "$OUT/REPORT.md"
echo "Results in $OUT"
