#!/bin/bash
# profile.sh <model> <tag>: per-step CPU vs GPU time, sync vs async (benchmarks/profile_step.py).
#   srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 30 bash scripts/explorer/profile.sh Qwen/Qwen2.5-0.5B-Instruct h200-qwen2.5-0.5b
# Env: PROFILE_ARGS (extra profile_step.py flags, e.g. --enforce-eager).
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/common.sh" ] || HERE=${SLURM_SUBMIT_DIR:-.}/scripts/explorer
source "$HERE/common.sh"
MODEL=$1; TAG=$2; OUT=$ROOT/results/$TAG; mkdir -p "$OUT"; cd "$REPO"
NAME=${PROFILE_NAME:-profile}
img python3 benchmarks/profile_step.py --model "$MODEL" --output "$OUT/$NAME.json" ${PROFILE_ARGS:-} \
  > "$OUT/$NAME.log" 2>&1
grep -E "^\||ms per decode step|GPU kernels|Error|error" "$OUT/$NAME.log"
