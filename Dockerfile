# syntax=docker/dockerfile:1
#
# minillm serving image: PyTorch built for CUDA 12.8 and FlashInfer with
# precompiled kernels, so the image needs no CUDA toolkit or compiler. The CUDA
# runtime libraries come with the torch wheels; the driver comes from the host
# through the NVIDIA Container Toolkit (`docker run --gpus all`). Needs an
# NVIDIA driver that supports CUDA 12.8 (570 or newer).
#
#   docker build -t minillm .
#   docker run --gpus all -p 8000:8000 -v hf-cache:/data minillm \
#     --model Qwen/Qwen2.5-0.5B-Instruct

ARG PYTHON_IMAGE=python:3.12-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.9

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- minillm wheel
FROM ${PYTHON_IMAGE} AS wheel
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY minillm ./minillm
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    UV_PYTHON_DOWNLOADS=never uv build --wheel --out-dir /dist

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

# torch 2.11 is the newest release with cu128 wheels. FlashInfer 0.6.17 is the
# newest release with a cu128 build of flashinfer-jit-cache, its precompiled
# kernels; without that package FlashInfer compiles kernels with nvcc on first use.
ARG CUDA_TAG=cu128
ARG TORCH_VERSION=2.11.0
ARG FLASHINFER_VERSION=0.6.17

LABEL org.opencontainers.image.title="minillm" \
      org.opencontainers.image.description="Single-GPU LLM serving engine with an OpenAI-compatible API" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin minillm \
 && mkdir -p /data \
 && chown minillm:minillm /data

# Dependencies first, so this layer (almost all of the image) is rebuilt only
# when pyproject.toml or the versions above change. uv is mounted, not installed.
# Bytecode is compiled up front: the server runs as a user that cannot write
# __pycache__ into the virtualenv.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/pyproject.toml \
    export UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
 && uv venv /opt/venv \
 && uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
      "torch==${TORCH_VERSION}+${CUDA_TAG}" \
 && uv pip install --extra-index-url "https://flashinfer.ai/whl/${CUDA_TAG}" \
      "flashinfer-python==${FLASHINFER_VERSION}" \
      "flashinfer-jit-cache==${FLASHINFER_VERSION}+${CUDA_TAG}" \
      -r /tmp/pyproject.toml

RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=from=wheel,source=/dist,target=/dist \
    UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 uv pip install --no-deps /dist/*.whl

USER minillm
WORKDIR /home/minillm
VOLUME ["/data"]
EXPOSE 8000

# /health answers once the model is loaded and the CUDA graphs are captured,
# which includes the first download of the weights, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --start-interval=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

ENTRYPOINT ["minillm-serve"]
