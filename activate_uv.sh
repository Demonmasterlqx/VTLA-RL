#!/usr/bin/env bash

# Source this file from an interactive Bash shell to use the host IsaacLab /
# Tabero_X environment.  It intentionally does not create, remove, or
# activate a conda environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '请使用 source %s\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

UV_TABERO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UV_TABERO_ENV="${UV_TABERO_ROOT}/IsaacLab/.venv"
UV_TABERO_SIM="${UV_TABERO_ROOT}/isaacsim"

if [[ ! -x "${UV_TABERO_ENV}/bin/python" ]]; then
  printf 'IsaacLab uv 环境不存在：%s\n' "${UV_TABERO_ENV}" >&2
  return 1
fi
if [[ ! -x "${UV_TABERO_SIM}/python.sh" ]]; then
  printf 'Isaac Sim 二进制不存在：%s\n' "${UV_TABERO_SIM}" >&2
  return 1
fi

# Load workspace credentials/settings when present.  .env remains local and
# is ignored by git; this is the same contract used by the Docker entrypoint.
if [[ -f "${UV_TABERO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${UV_TABERO_ROOT}/.env"
  set +a
fi

export VIRTUAL_ENV="${UV_TABERO_ENV}"
export ISAACLAB_PYTHON="${UV_TABERO_ENV}/bin/python"
export ISAACLAB_PATH="${UV_TABERO_ROOT}/IsaacLab"
export ISAACSIM_PATH="${UV_TABERO_SIM}"
export PYTHONNOUSERSITE=1
export PATH="${UV_TABERO_ENV}/bin:${PATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER

# Isaac Sim supplies the Kit/Omniverse bindings; Python packages and the
# IsaacLab/Tabero_X editable checkouts come from the shared uv environment.
set +u
source "${UV_TABERO_SIM}/setup_python_env.sh"
set -u
export PYTHONPATH="${UV_TABERO_ENV}/lib/python3.11/site-packages:${UV_TABERO_ROOT}/Tabero_X/source/tac_manip:${UV_TABERO_ROOT}/IsaacLab/source/isaaclab:${UV_TABERO_ROOT}/IsaacLab/source/isaaclab_assets:${UV_TABERO_ROOT}/IsaacLab/source/isaaclab_mimic:${UV_TABERO_ROOT}/IsaacLab/source/isaaclab_rl:${UV_TABERO_ROOT}/IsaacLab/source/isaaclab_tasks:${PYTHONPATH:-}"

isaaclab() {
  "${UV_TABERO_ROOT}/IsaacLab/isaaclab.sh" "$@"
}

printf '已加载 Tabero uv 环境：%s\n' "${UV_TABERO_ENV}"
printf 'Isaac Sim：%s\n' "${UV_TABERO_SIM}"
