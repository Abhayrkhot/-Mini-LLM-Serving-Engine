#!/bin/bash
# One-time: install the test-only packages the vLLM image lacks into $ROOT/overlay.
# Run on a CPU node, e.g.: srun -p short -c 4 --mem=16G -t 20 bash scripts/explorer/setup.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# sbatch runs a spooled copy of this file; fall back to the submit dir (the repo root).
[ -f "$HERE/common.sh" ] || HERE=${SLURM_SUBMIT_DIR:-.}/scripts/explorer
source "$HERE/common.sh"
export PIP_CACHE_DIR=$ROOT/pipcache
apptainer exec "$SIF" python3 -m pip install -q --no-deps --target "$ROOT/overlay" \
  hypothesis sortedcontainers pytest-asyncio
echo "overlay ready: $(ls "$ROOT/overlay" | tr '\n' ' ')"
