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

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ISAACSIM_PATH=/root/Tabero/isaacsim \
    ISAACLAB_PATH=/root/Tabero/IsaacLab \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

USER root

RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    apt-get update \
    && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
       -u http_proxy -u https_proxy -u all_proxy \
       apt-get install -y --no-install-recommends \
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
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

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

WORKDIR /root/Tabero

# Copy source repositories, including their .git metadata, so origin URLs,
# branches and commit history remain available inside the image.  isaacsim is
# provided once by the base image and linked below.
COPY IsaacLab /root/Tabero/IsaacLab
COPY T2-VLA /root/Tabero/T2-VLA
COPY Tabero_X /root/Tabero/Tabero_X
COPY RLinf /root/Tabero/RLinf

RUN test -x /isaac-sim/python.sh \
    && ln -s /isaac-sim /root/Tabero/isaacsim \
    && rm -f /root/Tabero/IsaacLab/_isaac_sim \
    && ln -s /root/Tabero/isaacsim /root/Tabero/IsaacLab/_isaac_sim \
    && rm -f /root/Tabero/RLinf/isaac_sim \
    && ln -s ../isaacsim /root/Tabero/RLinf/isaac_sim \
    && test "$(git -C /root/Tabero/IsaacLab branch --show-current)" = "release/2.3.0" \
    && test "$(git -C /root/Tabero/T2-VLA branch --show-current)" = "rl" \
    && test "$(git -C /root/Tabero/Tabero_X branch --show-current)" = "rl" \
    && test "$(git -C /root/Tabero/RLinf branch --show-current)" = "tabero" \
    && test "$(git -C /root/Tabero/IsaacLab remote get-url origin)" = "git@github.com:isaac-sim/IsaacLab.git" \
    && test "$(git -C /root/Tabero/T2-VLA remote get-url origin)" = "git@github.com:NathanWu7/T2-VLA.git" \
    && test "$(git -C /root/Tabero/Tabero_X remote get-url origin)" = "git@github.com:xiangixn/Tabero_X.git" \
    && test "$(git -C /root/Tabero/RLinf remote get-url origin)" = "git@github.com:Demonmasterlqx/RLinf.git"

# Three uv environments are intentionally kept separate.  IsaacLab and
# Tabero_X share one environment; RLinf and T2-VLA retain their own dependency
# contracts.  uv manages all three environments, even though Isaac Sim's
# bundled python.sh remains the simulator launcher.
RUN --mount=type=bind,from=uv-cache,target=/root/.cache/uv,rw \
    uv python install 3.11 \
    && uv venv --seed --python 3.11 /root/Tabero/IsaacLab/.venv \
    && uv venv --seed --python 3.11 /root/Tabero/RLinf/.venv \
    && uv venv --seed --python 3.11 /root/Tabero/T2-VLA/.venv

# Seed the IsaacLab/Tabero_X environment with the CUDA 12.8 Torch tuple in a
# separate cacheable layer.  IsaacLab's installer detects the exact versions
# and will not download them a second time.
RUN --mount=type=bind,from=uv-cache,target=/root/.cache/uv,rw \
    uv pip install --offline --no-deps --python /root/Tabero/IsaacLab/.venv/bin/python \
       --index-url https://download.pytorch.org/whl/cu128 \
       'torch==2.7.0' \
       'torchvision==0.22.0'

# IsaacLab's patched launcher accepts an explicit uv interpreter.
RUN --mount=type=bind,from=uv-cache,target=/root/.cache/uv,rw \
    --mount=type=bind,from=pip-cache,target=/root/.cache/pip,rw \
    cd /root/Tabero \
    && uv pip install --python /root/Tabero/IsaacLab/.venv/bin/python \
       'setuptools==75.8.2' 'wheel==0.45.1' \
    && uv pip install --python /root/Tabero/IsaacLab/.venv/bin/python \
       --no-build-isolation 'flatdict==4.0.1' \
    && VIRTUAL_ENV=/root/Tabero/IsaacLab/.venv \
       ISAACLAB_PYTHON=/root/Tabero/IsaacLab/.venv/bin/python \
       ISAACLAB_PATH=/root/Tabero/IsaacLab \
       PIP_NO_BUILD_ISOLATION=1 \
       ./IsaacLab/isaaclab.sh --install none \
    && uv pip install --python /root/Tabero/IsaacLab/.venv/bin/python \
       -e /root/Tabero/Tabero_X/source/tac_manip

# Taxim's GPU backend is part of the Tabero_X runtime contract.  Keep the JAX
# CUDA 12 tuple in the same shared uv environment instead of relying on a
# host/conda installation.
RUN --mount=type=bind,from=uv-cache,target=/root/.cache/uv,rw \
    uv pip install --python /root/Tabero/IsaacLab/.venv/bin/python \
       'jax==0.5.3' \
       'jaxlib==0.5.3' \
       'jax-cuda12-pjrt==0.5.3' \
       'jax-cuda12-plugin==0.5.3'

# T2-VLA has an authoritative uv.lock and is installed frozen.
RUN --mount=type=bind,from=uv-cache,target=/root/.cache/uv,rw \
    cd /root/Tabero/T2-VLA \
    && UV_PROJECT_ENVIRONMENT=/root/Tabero/T2-VLA/.venv \
       uv sync --frozen --no-install-package lerobot --python 3.11 \
    && LEROBOT_ARCHIVE="$(dirname "$(find /root/.cache/uv/archive-v0 -type f -path '*/lerobot-0.3.3.dist-info/METADATA' -print -quit)")" \
    && test -n "${LEROBOT_ARCHIVE}" \
    && LEROBOT_ARCHIVE="${LEROBOT_ARCHIVE%/lerobot-0.3.3.dist-info}" \
    && SITE_PACKAGES=/root/Tabero/T2-VLA/.venv/lib/python3.11/site-packages \
    && cp -a "${LEROBOT_ARCHIVE}/lerobot" "${SITE_PACKAGES}/" \
    && cp -a "${LEROBOT_ARCHIVE}/lerobot-0.3.3.dist-info" "${SITE_PACKAGES}/"

# RLinf currently has no trusted lockfile in this checkout.  Install the
# validated CUDA 12.8 Torch tuple first, then resolve RLinf's embodied extras
# under explicit constraints instead of copying the host .venv.
RUN printf '%s\n' \
       "torch==${RLINF_TORCH_VERSION}" \
       "torchvision==${RLINF_TORCHVISION_VERSION}" \
       "torchaudio==${RLINF_TORCHAUDIO_VERSION}" \
       'datasets==3.6.0' \
       'torchcodec==0.2' \
       'gymnasium==0.29.1' \
       > /tmp/rlinf-constraints.txt \
    && uv pip install --python /root/Tabero/RLinf/.venv/bin/python \
       --index-url https://download.pytorch.org/whl/cu128 \
       "torch==${RLINF_TORCH_VERSION}" \
       "torchvision==${RLINF_TORCHVISION_VERSION}" \
       "torchaudio==${RLINF_TORCHAUDIO_VERSION}" \
    && uv pip install --python /root/Tabero/RLinf/.venv/bin/python \
       --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
       'torch-scatter==2.1.2+pt27cu128' \
    && uv pip install --python /root/Tabero/RLinf/.venv/bin/python \
       --index-url https://pypi.org/simple \
       --constraint /tmp/rlinf-constraints.txt \
       -e '/root/Tabero/RLinf[embodied]' \
    && rm -f /tmp/rlinf-constraints.txt

COPY docker/entrypoint.sh /root/Tabero/docker/entrypoint.sh
COPY docker/validate.sh /root/Tabero/docker/validate.sh
COPY docker/pull_repos.sh /root/Tabero/docker/pull_repos.sh
COPY docker/README.md /root/Tabero/docker/README.md
COPY activate_uv.sh /root/Tabero/activate_uv.sh
RUN chmod +x /root/Tabero/docker/entrypoint.sh /root/Tabero/docker/validate.sh /root/Tabero/docker/pull_repos.sh \
    /root/Tabero/activate_uv.sh \
    && mkdir -p /root/Tabero/datasets /root/Tabero/models /root/Tabero/results /root/Tabero/Record

ENTRYPOINT ["/root/Tabero/docker/entrypoint.sh"]
CMD ["shell"]
