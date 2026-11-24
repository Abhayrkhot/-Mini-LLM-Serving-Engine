#!/bin/bash
# Build flash-attn 2.8.3 from source against the image's torch (2.14, CUDA 13), for sm90
# only, into $ROOT/overlay-fa. No prebuilt flash-attn wheel loads against that torch, and
# this is what the `flash_attn` backend's GPU tests run on. torch 2.14's headers need
# C++20 while flash-attn 2.8.3 compiles as C++17, so its flags are patched. About 50 min
# on 24 cores (no GPU needed):
#   sbatch -p short -c 24 --mem=96G -t 2:00:00 scripts/explorer/build_flash_attn.sh
# Then put it on the path for a job, e.g.:
#   EXTRA_PYTHONPATH=/scratch/$USER/minillm/overlay-fa srun ... bash scripts/explorer/tests.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/common.sh" ] || HERE=${SLURM_SUBMIT_DIR:-.}/scripts/explorer
source "$HERE/common.sh"
OUT=$ROOT/overlay-fa SRC=$ROOT/fa-src
rm -rf "$SRC" && mkdir -p "$OUT" "$SRC"
export PIP_CACHE_DIR=$ROOT/cache/pip
apptainer exec --bind "/scratch/$USER" "$SIF" bash -c "
  set -euo pipefail
  export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 \
         MAX_JOBS=12 NVCC_THREADS=2
  cd '$SRC'
  pip download --no-deps --no-build-isolation --no-binary :all: flash-attn==2.8.3 -d .
  tar xzf flash_attn-2.8.3.tar.gz && cd flash_attn-2.8.3
  sed -i s/c++17/c++20/g setup.py
  pip install --no-build-isolation --no-deps --target '$OUT' . > '$ROOT/fa-build.log' 2>&1
"
ls "$OUT"
