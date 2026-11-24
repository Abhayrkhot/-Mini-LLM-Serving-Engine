# Shared setup for running minillm on a Slurm GPU node from the vLLM Apptainer image.
# Source it from the other scripts in this directory; it does nothing on its own.
#
#   MINILLM_ROOT  work dir on scratch (overlay, results, caches)   default /scratch/$USER/minillm
#   MINILLM_SIF   vLLM image (vLLM 0.34, torch 2.14+cu130, FlashInfer 0.7.3)
#   HF_HOME       Hugging Face cache holding the models (used offline)
#   EXTRA_PYTHONPATH  more package dirs for the container, e.g. $MINILLM_ROOT/overlay-fa
#                 (flash-attn, from build_flash_attn.sh)

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ROOT=${MINILLM_ROOT:-/scratch/$USER/minillm}
export SIF=${MINILLM_SIF:-/scratch/$USER/icp/vllm.sif}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1
# Keep ~/.local packages out of the container: they shadow the image's torch.
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
mkdir -p "$TRITON_CACHE_DIR" "$ROOT/results"

# The image is built for CUDA 13; a 570 (CUDA 12.8) driver needs its forward-compat
# libraries first. Extend LD_LIBRARY_PATH *inside* the container (single quotes): it
# already holds the driver libraries apptainer injects, NVML included, which vLLM needs.
INNER='export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec "$@"'

img() {  # img <command...>: run inside the image with the GPU, scratch and our overlay
  PYTHONPATH="$REPO:$ROOT/overlay${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}" \
    apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c "$INNER" _ "$@"
}

SERVER_PID=""
start_server() {  # start_server <log> <command...>: own process group, wait for /v1/models
  local log=$1; shift
  echo "[$(date +%T)] starting: $*"
  PYTHONPATH="$REPO:$ROOT/overlay" setsid apptainer exec --nv --bind "/scratch/$USER" "$SIF" \
    bash -c "$INNER" _ "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  local t0; t0=$(date +%s)
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "[$(date +%T)] ready after $(( $(date +%s) - t0 ))s"; return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited"; tail -40 "$log"; return 1; fi
    sleep 5
  done
  echo "server not ready after 20 min"; tail -40 "$log"; return 1
}

stop_server() {  # this job's server only (its process group), never a node-wide pkill
  [ -z "$SERVER_PID" ] && return
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 -- "-$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 3
}
trap stop_server EXIT
