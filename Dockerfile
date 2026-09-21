ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
ARG UV_VERSION=0.12.9

FROM ${ISAACSIM_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG UV_VERSION
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG RLINF_TORCH_VERSION=2.7.1
ARG RLINF_TORCHVISION_VERSION=0.22.1
ARG RLINF_TORCHAUDIO_VERSION=2.7.1
ARG RLINF_OPENPI_REF=c5dc4b9296a1a4739bf52828f28a579f12dce763
ARG RLINF_LEROBOT_REF=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ARG RLINF_FLASH_ATTN_VERSION=2.7.4.post1

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ISAACSIM_PATH=/root/VTLA-RL/isaacsim \
    ISAACLAB_PATH=/root/VTLA-RL/IsaacLab \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    ROOT=/root/VTLA-RL

USER root

RUN --mount=type=cache,id=vtla-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=vtla-apt-cache,target=/var/cache/apt,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
        python3-dev \
        rsync \
        unzip \
        vim \
        wget \
        zstd \
        ffmpeg \
    && git lfs install --system

# Pin uv independently from the project dependency lockfiles.  Downloading
# the release binary here avoids requiring a separate GHCR image pull; the
# optional build proxy is available through the ARGs above.
RUN UV_PROXY="${HTTPS_PROXY:-${HTTP_PROXY:-}}" \
    && curl --fail --retry 5 --retry-delay 2 --location --proxy "${UV_PROXY}" \
       "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
       --output /tmp/uv.tar.gz \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

# Use a configurable domestic mirror for ordinary Python packages. Commands
# that require a CUDA-specific wheel index keep their explicit upstream index.
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
ARG HF_ENDPOINT=https://hf-mirror.com

WORKDIR /root/VTLA-RL

# Copy source repositories, including their .git metadata, so origin URLs,
# branches and commit history remain available inside the image.  isaacsim is
# provided once by the base image and linked below.
COPY IsaacLab /root/VTLA-RL/IsaacLab
COPY T2-VLA /root/VTLA-RL/T2-VLA
COPY Tabero_X /root/VTLA-RL/Tabero_X
COPY RLinf /root/VTLA-RL/RLinf

# Isaac Sim is provided by the NVIDIA base image at /isaac-sim.  Keep a
# single installation and expose the paths expected by this workspace,
# IsaacLab, and RLinf through symbolic links.
RUN test -x /isaac-sim/python.sh \
    && rm -f /root/VTLA-RL/isaacsim \
    && ln -s /isaac-sim /root/VTLA-RL/isaacsim \
    && rm -f /root/VTLA-RL/IsaacLab/_isaac_sim \
    && ln -s /root/VTLA-RL/isaacsim /root/VTLA-RL/IsaacLab/_isaac_sim \
    && rm -f /root/VTLA-RL/RLinf/isaac_sim \
    && ln -s ../isaacsim /root/VTLA-RL/RLinf/isaac_sim

# Three uv environments are intentionally kept separate.  IsaacLab and
# Tabero_X share one environment; RLinf and T2-VLA retain their own dependency
# contracts.  uv manages all three environments, even though Isaac Sim's
# bundled python.sh remains the simulator launcher.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    UV_DEFAULT_INDEX="${PYPI_INDEX_URL}" \
    PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    && uv python install 3.11 \
    && uv venv --seed --python 3.11 /root/VTLA-RL/IsaacLab/.venv \
    && uv venv --seed --python 3.11 /root/VTLA-RL/RLinf/.venv \
    && uv venv --seed --python 3.11 /root/VTLA-RL/T2-VLA/.venv

# Seed the IsaacLab/Tabero_X environment with the CUDA 12.8 Torch tuple in a
# separate cacheable layer.  IsaacLab's installer detects the exact versions
# and will not download them a second time.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    && uv pip install --no-deps --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       --index-url "${PYTORCH_INDEX_URL}" \
       'torch==2.7.0' \
       'torchvision==0.22.0'

# IsaacLab's launcher selects $CONDA_PREFIX/bin/python when CONDA_PREFIX is
# present.  Point it at the uv environment so the extensions and their Python
# dependencies are installed into IsaacLab/.venv instead of Isaac Sim's Kit
# Python.  Keep network asset retrieval out of this layer so a transient
# Hugging Face failure does not invalidate the completed environment install.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=vtla-pip-cache,target=/root/.cache/pip,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    UV_DEFAULT_INDEX="${PYPI_INDEX_URL}" \
    PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    && cd /root/VTLA-RL \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       'setuptools==75.8.2' 'wheel==0.45.1' 'huggingface-hub==0.36.0' \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       --no-build-isolation 'flatdict==4.0.1' \
    && CONDA_PREFIX=/root/VTLA-RL/IsaacLab/.venv \
       VIRTUAL_ENV=/root/VTLA-RL/IsaacLab/.venv \
       ISAACLAB_PATH=/root/VTLA-RL/IsaacLab \
       PIP_NO_BUILD_ISOLATION=1 \
       ./IsaacLab/isaaclab.sh --install none \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       --requirement /root/VTLA-RL/Tabero_X/requirements.txt \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       --no-index --no-deps \
       --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
       'torch-scatter==2.1.2+pt27cu128' \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       -e /root/VTLA-RL/Tabero_X/source/tac_manip

# Download the immutable Tabero_X asset snapshot through a configurable
# Hugging Face endpoint.  The cache mount preserves partial downloads across
# retries, while --mode copy places the verified assets in the final image
# instead of leaving links to the build-only cache directory.
RUN --mount=type=cache,id=vtla-tabero-assets,target=/root/.cache/tabero-assets,sharing=locked \
    --mount=type=secret,id=hf_token \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=60 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    && if [[ -s /run/secrets/hf_token ]]; then \
         export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
       fi \
    && cd /root/VTLA-RL/Tabero_X/scripts/tools/assets \
    && /root/VTLA-RL/IsaacLab/.venv/bin/python - <<'PY'
from functools import partial
from pathlib import Path

from fetch_hf_assets import fetch_assets
from huggingface_hub import snapshot_download

repo_root = Path("/root/VTLA-RL/Tabero_X")
snapshot_root = fetch_assets(
    repo_root=repo_root,
    lock_path=repo_root / "assets/hf_assets.lock.json",
    asset_store=Path("/root/.cache/tabero-assets"),
    mode="copy",
    repair_links=True,
    downloader=partial(snapshot_download, max_workers=1),
)
print(f"Assets ready at: {snapshot_root}")
PY

# Taxim's GPU backend is part of the Tabero_X runtime contract.  Keep the JAX
# CUDA 12 tuple in the same shared uv environment instead of relying on a
# host/conda installation.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    UV_DEFAULT_INDEX="${PYPI_INDEX_URL}" \
    PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    && uv pip install --python /root/VTLA-RL/IsaacLab/.venv/bin/python \
       'jax==0.5.3' \
       'jaxlib==0.5.3' \
       'jax-cuda12-pjrt==0.5.3' \
       'jax-cuda12-plugin==0.5.3'

# Seed T2-VLA's exact CUDA 12.8 Torch tuple through the configurable mirrors.
# Its transitive NVIDIA wheels are locked to PyPI, so use the ordinary mirror
# as the fallback index while keeping the CUDA Torch index authoritative.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    && uv pip install \
       --python /root/VTLA-RL/T2-VLA/.venv/bin/python \
       --index "${PYTORCH_INDEX_URL}" \
       --default-index "${PYPI_INDEX_URL}" \
       'torch==2.7.1+cu128' \
       'torchvision==0.22.1+cu128'

# T2-VLA has an authoritative uv.lock and is installed frozen. Seed the locked
# registry packages (including their hashes) from the mirror first, because
# frozen sync otherwise follows files.pythonhosted.org URLs from the lock.
# uv pip install transformers==4.53.2 && cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
# for transformers_replace is not installed correctly.
#
# uv run python - <<'PY'
# from openpi.shared import download

# path = download.maybe_download(
#     "gs://big_vision/paligemma_tokenizer.model",
#     gs={"token": "anon"},
# )

# print("Downloaded to:", path)
# PY
# for loading the tokenizer model in container.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    UV_DEFAULT_INDEX="${PYPI_INDEX_URL}" \
    PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    && cd /root/VTLA-RL/T2-VLA \
    && uv export --frozen --no-emit-workspace \
       --no-emit-package lerobot --no-emit-package torch \
       --no-emit-package torchvision --output-file /tmp/t2-locked.txt \
    && uv pip install --no-deps \
       --python /root/VTLA-RL/T2-VLA/.venv/bin/python \
       --index-url "${PYPI_INDEX_URL}" \
       --requirement /tmp/t2-locked.txt \
    && UV_PROJECT_ENVIRONMENT=/root/VTLA-RL/T2-VLA/.venv \
       uv sync --frozen --python 3.11 \
    && SITE_PACKAGES=/root/VTLA-RL/T2-VLA/.venv/lib/python3.11/site-packages \
    && cp -a ./src/openpi/models_pytorch/transformers_replace/. "${SITE_PACKAGES}/transformers/" \
    && /root/VTLA-RL/T2-VLA/.venv/bin/python - <<'PY'
from openpi.shared import download

path = download.maybe_download(
   "gs://big_vision/paligemma_tokenizer.model",
   gs={"token": "anon"},
)

print("Downloaded to:", path)
PY

# RLinf currently has no trusted lockfile in this checkout.  Install the
# validated CUDA 12.8 Torch tuple first, then resolve RLinf's embodied extras
# under explicit constraints instead of copying the host .venv.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    && printf '%s\n' \
       "torch==${RLINF_TORCH_VERSION}" \
       "torchvision==${RLINF_TORCHVISION_VERSION}" \
       "torchaudio==${RLINF_TORCHAUDIO_VERSION}" \
       'datasets==3.6.0' \
       'torchcodec==0.2' \
       'gymnasium==0.29.1' \
       > /tmp/rlinf-constraints.txt \
    && uv pip install --python /root/VTLA-RL/RLinf/.venv/bin/python \
       --index-url "${PYTORCH_INDEX_URL}" \
       "torch==${RLINF_TORCH_VERSION}" \
       "torchvision==${RLINF_TORCHVISION_VERSION}" \
       "torchaudio==${RLINF_TORCHAUDIO_VERSION}" \
    && uv pip install --python /root/VTLA-RL/RLinf/.venv/bin/python \
       --no-index --no-deps \
       --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
       'torch-scatter==2.1.2+pt27cu128' \
    && uv pip install --python /root/VTLA-RL/RLinf/.venv/bin/python \
       --index-url "${PYPI_INDEX_URL}" \
       --constraint /tmp/rlinf-constraints.txt \
       -e '/root/VTLA-RL/RLinf[embodied]' \
    && rm -f /tmp/rlinf-constraints.txt

# The RLinf OpenPI training path has additional dependencies that are not part
# of RLinf[embodied].  Install them into the existing RLinf environment without
# recreating it, keep the validated Torch tuple, and apply OpenPI's Transformers
# replacement exactly as RLinf's official installer does.
RUN --mount=type=cache,id=vtla-uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=vtla-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=secret,id=hf_token \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    UV_DEFAULT_INDEX="${PYPI_INDEX_URL}" \
    PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=60 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    && RLINF_PYTHON=/root/VTLA-RL/RLinf/.venv/bin/python \
    && if [[ -s /run/secrets/hf_token ]]; then \
         export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
       fi \
    && uv pip install --python "${RLINF_PYTHON}" \
       "git+https://github.com/huggingface/lerobot.git@${RLINF_LEROBOT_REF}" \
    && uv pip install --python "${RLINF_PYTHON}" \
       "git+https://github.com/RLinf/openpi.git@${RLINF_OPENPI_REF}#subdirectory=packages/openpi-client" \
       "git+https://github.com/RLinf/openpi.git@${RLINF_OPENPI_REF}" \
    && uv pip install --python "${RLINF_PYTHON}" \
       --requirement /root/VTLA-RL/RLinf/requirements/embodied/models/openpi.txt \
       'torchcodec==0.2.1' \
       'numpydantic==1.7.0' \
       'pydantic==2.11.7' \
       'numpy==1.26.0' \
    && SITE_PACKAGES="$("${RLINF_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')" \
    && test -d "${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace" \
    && cp -a "${SITE_PACKAGES}/openpi/models_pytorch/transformers_replace/." \
       "${SITE_PACKAGES}/transformers/" \
    && PY_TAG="$("${RLINF_PYTHON}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')" \
    && TORCH_MM="$("${RLINF_PYTHON}" -c 'import torch; print(".".join(torch.__version__.split("+", 1)[0].split(".")[:2]))')" \
    && CXX11_ABI="$("${RLINF_PYTHON}" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')" \
    && FLASH_ATTN_WHEEL="flash_attn-${RLINF_FLASH_ATTN_VERSION}%2Bcu12torch${TORCH_MM}cxx11abi${CXX11_ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl" \
    && uv pip install --python "${RLINF_PYTHON}" \
       "https://github.com/Dao-AILab/flash-attention/releases/download/v${RLINF_FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL}" \
    && TOKENIZER_DIR=/root/.cache/openpi \
    && TOKENIZER_PATH="${TOKENIZER_DIR}/big_vision/paligemma_tokenizer.model" \
    && if [[ ! -s "${TOKENIZER_PATH}" ]]; then \
         mkdir -p "${TOKENIZER_DIR}"; \
         PATH="/root/VTLA-RL/RLinf/.venv/bin:${PATH}" \
           hf download RLinf/openpi_tokenizer \
           big_vision/paligemma_tokenizer.model \
           --local-dir "${TOKENIZER_DIR}"; \
       fi \
    && test -s "${TOKENIZER_PATH}" \
    && (uv pip uninstall --python "${RLINF_PYTHON}" pynvml || true) \
    && uv pip check --python "${RLINF_PYTHON}"

# Fail the build immediately if either runtime is incomplete.  Importing these
# modules does not start Isaac Sim or allocate a GPU.
RUN /root/VTLA-RL/IsaacLab/.venv/bin/python - <<'PY'
from importlib.metadata import version

import cv2
import huggingface_hub
import isaaclab
import msgpack
import torch
import torch_scatter
import tree
import tyro
import websockets

if int(version("numpy").split(".", 1)[0]) >= 2:
    raise SystemExit(f'Expected numpy<2, found {version("numpy")}')
print("Tabero_X requirements import check passed")
PY

RUN PYTHONPATH="/root/VTLA-RL/RLinf/.venv/lib/python3.11/site-packages:/root/VTLA-RL/IsaacLab/.venv/lib/python3.11/site-packages" \
    /root/VTLA-RL/RLinf/.venv/bin/python - <<'PY'
from importlib.metadata import version

import cv2
import flatdict
import flash_attn
import jax
import lerobot
import openpi
import openpi_client
import torch

expected = {
    "jax": "0.5.3",
    "numpy": "1.26.0",
    "orbax-checkpoint": "0.11.13",
    "torch": "2.7.1",
    "transformers": "4.53.2",
    "tyro": "1.0.13",
}
for package, required in expected.items():
    actual = version(package).split("+", 1)[0]
    if actual != required:
        raise SystemExit(f"Expected {package}=={required}, found {actual}")
print("RLinf OpenPI training imports passed")
PY

# Training processes are supervised from a named tmux session. Install tmux
# late so changes to this operational tool never invalidate the Python layers.
RUN --mount=type=cache,id=vtla-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=vtla-apt-cache,target=/var/cache/apt,sharing=locked \
    export HTTP_PROXY="${HTTP_PROXY}" \
    HTTPS_PROXY="${HTTPS_PROXY}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends tmux

# Preserve the paths used by the checked-in Tabero RL/SFT launchers while the
# actual repositories and bind-mounted assets live under /root/VTLA-RL.
COPY docker /root/VTLA-RL/docker

RUN mkdir -p /root/VTLA-RL/datasets /root/VTLA-RL/models /root/VTLA-RL/results /root/VTLA-RL/Record \
    /data/home/sim6g/code \
    && ln -sfn datasets /root/VTLA-RL/datas \
    && ln -sfn Tabero_X /root/VTLA-RL/Tabero \
    && ln -sfn /root/VTLA-RL /data/home/sim6g/code/tabero \
    && chmod +x /root/VTLA-RL/docker/entrypoint.sh /root/VTLA-RL/docker/env_setup/*.sh

# Materialize the Libero USD bundle in the final layer.  The source repository
# is pinned independently from the Tabero_X code so rebuilding this fix can
# reuse every dependency layer above it.
ARG LIBERO_ASSET_REPO_ID=china-sae-robotics/IsaacLabPlayGround_Dataset
ARG LIBERO_ASSET_REVISION=803e10a630b8cdc0a5ea032bb162de92cb43a9bf
ARG LIBERO_HTTP_PROXY
ARG LIBERO_HTTPS_PROXY
ENV LIBERO_ASSETS_DATA_DIR=/root/VTLA-RL/Tabero_X/benchmarks/datasets/libero/USD
RUN --mount=type=cache,id=vtla-libero-assets,target=/root/.cache/libero-assets,sharing=locked \
    --mount=type=secret,id=hf_token \
    export HTTP_PROXY="${LIBERO_HTTP_PROXY:-${HTTP_PROXY}}" \
    HTTPS_PROXY="${LIBERO_HTTPS_PROXY:-${HTTPS_PROXY}}" \
    ALL_PROXY="${ALL_PROXY}" \
    http_proxy="${HTTP_PROXY}" \
    https_proxy="${HTTPS_PROXY}" \
    all_proxy="${ALL_PROXY}" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=60 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    LIBERO_REPO_ID="${LIBERO_ASSET_REPO_ID}" \
    LIBERO_REVISION="${LIBERO_ASSET_REVISION}" \
    && if [[ -s /run/secrets/hf_token ]]; then \
         export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
       fi \
    && cd /root/VTLA-RL/Tabero_X \
    && /root/VTLA-RL/IsaacLab/.venv/bin/python - <<'PY'
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from pathlib import Path

repo_root = Path.cwd()
asset_store = Path("/root/.cache/libero-assets")
asset_root = repo_root / "benchmarks/datasets/libero/USD"
endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
repo_id = os.environ["LIBERO_REPO_ID"]
revision = os.environ["LIBERO_REVISION"]
download_root = asset_store / "raw"
metadata_url = (
    f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/"
    f"{quote('libero/USD', safe='')}?recursive=true"
)
metadata = subprocess.run(
    [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        "--silent",
        "--show-error",
        metadata_url,
    ],
    check=True,
    capture_output=True,
    text=True,
)
repo_files = [entry for entry in json.loads(metadata.stdout) if entry.get("type") == "file"]
if not repo_files:
    raise FileNotFoundError("Hub revision contains no libero/USD files")
def download_entry(entry):
    relative_path = Path(entry["path"])
    destination = download_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry.get("size")
    if destination.is_file() and expected_size is not None and destination.stat().st_size == expected_size:
        return
    url = (
        f"{endpoint}/datasets/{repo_id}/resolve/{revision}/"
        f"{quote(entry['path'], safe='/')}"
    )
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
        "--continue-at",
        "-",
        "--output",
        str(destination),
        "--silent",
        "--show-error",
    ]
    command.append(url)
    subprocess.run(command, check=True)
    if expected_size is not None and destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {entry['path']}: "
            f"expected {expected_size}, found {destination.stat().st_size}"
        )


with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(download_entry, repo_files))

asset_root.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(download_root / "libero/USD", asset_root, dirs_exist_ok=True)

expected_hash = "63dfcc779e931529f9ab5e248ab316cc"
expected_file_count = 121
expected_total_bytes = 144866192
hash_file = asset_root / ".asset_hash"
if hash_file.read_text(encoding="utf-8").strip() != expected_hash:
    raise RuntimeError(f"unexpected Libero asset hash in {hash_file}")

files = [path for path in asset_root.rglob("*") if path.is_file()]
if any(path.is_symlink() for path in asset_root.rglob("*")):
    raise RuntimeError("Libero USD bundle contains symlinks")
if len(files) != expected_file_count:
    raise RuntimeError(f"expected {expected_file_count} Libero files, found {len(files)}")
total_bytes = sum(path.stat().st_size for path in files)
if total_bytes != expected_total_bytes:
    raise RuntimeError(f"expected {expected_total_bytes} Libero bytes, found {total_bytes}")

config_root = repo_root / "benchmarks/datasets/libero/config"
object_types = set()
for config_path in sorted(config_root.glob("libero_*.json")):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for task in config.get("tasks", []):
        object_types.update(info.get("type") for info in task.get("objects", {}).values())
missing = sorted(
    object_type
    for object_type in object_types
    if not (asset_root / object_type / f"{object_type}.usd").is_file()
)
if missing:
    raise FileNotFoundError("missing Libero USD objects: " + ", ".join(missing))
print(
    f"Libero USD assets ready: files={len(files)} bytes={total_bytes} "
    f"objects={len(object_types)} revision={os.environ['LIBERO_REVISION']}"
)
PY

ENTRYPOINT ["/root/VTLA-RL/docker/entrypoint.sh"]
CMD ["shell"]
