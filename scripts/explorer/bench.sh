#!/bin/bash
# bench.sh <model> <tag> [rates...]: offline and online benchmarks of minillm, HF and vLLM.
#   sbatch -p gpu --gres=gpu:h200:1 -c 8 --mem=64G -t 4:00:00 scripts/explorer/bench.sh Qwen/Qwen2.5-7B-Instruct 7b
# Env: NUM_PROMPTS (500), ENGINES (offline: "minillm vllm hf"), SERVE_ENGINES ("minillm vllm"),
#      SKIP_OFFLINE / SKIP_ONLINE (1 to skip), MINILLM_ARGS (extra minillm flags, e.g.
#      --enforce-eager), WORKERS (load-generator processes, default 4).
#      SERVE_ENGINES may include minillm-1proc: the engine on a thread in the API process.
#      Results: $ROOT/results/<tag>/ + REPORT.md.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# sbatch runs a spooled copy of this file; fall back to the submit dir (the repo root).
[ -f "$HERE/common.sh" ] || HERE=${SLURM_SUBMIT_DIR:-.}/scripts/explorer
source "$HERE/common.sh"
MODEL=$1; TAG=$2; shift 2; RATES=${*:-"2 4 8 16 32 inf"}
OUT=$ROOT/results/$TAG; mkdir -p "$OUT"; cd "$REPO"
export PORT=$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))
nvidia-smi > "$OUT/nvidia-smi.txt"
LIMITS=(--max-num-seqs 256 --max-num-batched-tokens 8192)
N=${NUM_PROMPTS:-500}
WORKERS=${WORKERS:-4}  # one client process tops out near 10k streamed tokens/s

if [ "${SKIP_OFFLINE:-0}" != 1 ]; then
  for ratio in 0 0.9; do
    common=(--model "$MODEL" --num-prompts "$N" --input-len 512 --output-len 128 --range-ratio "$ratio" "${LIMITS[@]}")
    for engine in ${ENGINES:-minillm vllm hf}; do
      echo "[$(date +%T)] offline $engine ratio=$ratio"
      extra=(); [ "$engine" = minillm ] && extra=(${MINILLM_ARGS:-})
      img python3 benchmarks/bench_offline.py --engine "$engine" "${common[@]}" "${extra[@]}" --hf-batch-size 64 \
        --output "$OUT/offline_${engine}_r${ratio}.json" > "$OUT/offline_${engine}_r${ratio}.log" 2>&1 \
        || { echo "  FAILED"; tail -20 "$OUT/offline_${engine}_r${ratio}.log"; }
      grep -E '"output_throughput"' "$OUT/offline_${engine}_r${ratio}.json" 2>/dev/null
    done
  done
fi

sweep() {  # sweep <label>; every point gets fresh prompts (its own seed) so no point
  # is served from the prefix cache filled by the previous one
  local seed=0
  for rate in $RATES; do
    seed=$((seed + 1))
    img python3 benchmarks/bench_serving.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
      --label "$1" --num-prompts "$N" --request-rate "$rate" --input-len 512 --output-len 128 --seed "$seed" --workers "$WORKERS" \
      --range-ratio 0.5 --output "$OUT/serve_${1}_rate${rate}.json" > "$OUT/serve_${1}_rate${rate}.log" 2>&1
    grep -E "^(completed|failed|output_throughput|median_ttft_ms|median_tpot_ms) " "$OUT/serve_${1}_rate${rate}.log" | tr '\n' ' '; echo " <- $1 rate $rate"
  done
  img python3 benchmarks/bench_serving.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --label "$1" --num-prompts "$N" --request-rate 16 --prefix-len 1024 --input-len 128 --seed 100 --workers "$WORKERS" \
    --output-len 128 --output "$OUT/serve_${1}_prefix.json" > "$OUT/serve_${1}_prefix.log" 2>&1
  grep -E "^(median_ttft_ms|p99_ttft_ms) " "$OUT/serve_${1}_prefix.log" | tr '\n' ' '; echo " <- $1 prefix"
}

if [ "${SKIP_ONLINE:-0}" != 1 ]; then
  for engine in ${SERVE_ENGINES:-minillm vllm}; do
    if [ "$engine" = minillm ] || [ "$engine" = minillm-1proc ]; then
      mode=(); [ "$engine" = minillm-1proc ] && mode=(--no-engine-process)
      start_server "$OUT/${engine}_server.log" python3 -m minillm.entrypoints.openai_server \
        --model "$MODEL" --host 127.0.0.1 --port "$PORT" "${LIMITS[@]}" "${mode[@]}" ${MINILLM_ARGS:-} \
        --log-level warning && sweep "$engine"
    else
      start_server "$OUT/vllm_server.log" vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --served-model-name "$MODEL" "${LIMITS[@]}" && sweep vllm
    fi
    stop_server
  done
fi
img python3 scripts/report.py "$OUT" > "$OUT/REPORT.md" 2>/dev/null; echo "[$(date +%T)] done: $OUT"
