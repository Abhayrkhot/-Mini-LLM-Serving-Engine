#!/bin/bash
# Full test suite on one GPU: CPU tests, CUDA kernel/engine tests, real-checkpoint parity.
#   srun -p gpu-short --gres=gpu:h200:1 -c 8 --mem=48G -t 40 bash scripts/explorer/tests.sh
HERE=$(cd "$(dirname "$0")" && pwd)
# sbatch runs a spooled copy of this file; fall back to the submit dir (the repo root).
[ -f "$HERE/common.sh" ] || HERE=${SLURM_SUBMIT_DIR:-.}/scripts/explorer
source "$HERE/common.sh"
cd "$REPO"
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
LOG=$ROOT/results/tests_${SLURM_JOB_ID:-local}.log
{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  img python3 -c "import torch, flashinfer, transformers; print('torch', torch.__version__, '| flashinfer', flashinfer.__version__, '| transformers', transformers.__version__)"
  echo "=== full suite"
  img python3 -m pytest -q -p no:cacheprovider -p no:warnings -rsf
  echo "=== real checkpoint on CUDA ($MODEL): torch fp32 exact, GPU backends bf16"
  MINILLM_TEST_MODEL=$MODEL MINILLM_TEST_DEVICE=cuda \
    img python3 -m pytest -q -p no:cacheprovider -p no:warnings -rsf -m slow tests/test_real_model.py
} > "$LOG" 2>&1
grep -E "^(NVIDIA|torch )|^===|passed|failed|FAILED|Error" "$LOG"
echo "log: $LOG"
