#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/VTLA-RL
ISAAC_ENV="$ROOT/IsaacLab/.venv"
RLINF_ENV="$ROOT/RLinf/.venv"
T2_ENV="$ROOT/T2-VLA/.venv"

# Credentials and user-specific runtime settings are supplied as a read-only
# bind mount. They are deliberately not copied into the image.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# A host .env may contain host-only Isaac paths. Keep such settings from
# leaking into the container while preserving an explicitly valid override.
if [[ ! -x "${ISAACSIM_PATH:-}/python.sh" ]]; then
  export ISAACSIM_PATH="$ROOT/isaacsim"
fi
if [[ ! -d "${ISAACLAB_PATH:-}" ]]; then
  export ISAACLAB_PATH="$ROOT/IsaacLab"
fi
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

ISAAC_PYTHONPATH="$ISAAC_ENV/lib/python3.11/site-packages:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/Tabero_X/source/tac_manip"

case "${1:-shell}" in
  shell)
    shift || true
    # Make an interactive container shell use the same uv contract as the
    # host.  Credentials are still supplied through the external .env mount.
    # shellcheck disable=SC1091
    exec /bin/bash -l "$@"
    ;;
  tabero_x)
    shift
    # IsaacLab and Tabero_X share the uv environment.  Isaac Sim's setup
    # script contributes Kit/Omniverse bindings while isaaclab.sh selects the
    # uv interpreter through VIRTUAL_ENV/ISAACLAB_PYTHON.
    source "$ROOT/docker/env_setup/tabero_x.sh"
    exec /bin/bash -l "$@"
    ;;
  isaac-python)
    shift
    export PYTHONPATH="$ISAAC_PYTHONPATH:${PYTHONPATH:-}"
    unset CONDA_PREFIX VIRTUAL_ENV
    exec "$ISAACSIM_PATH/python.sh" "$@"
    ;;
  rlinf)
    shift
    source "$ROOT/docker/env_setup/rlinf.sh"
    exec /bin/bash -l "$@"
    ;;
  t2-vla)
    shift
    source "$ROOT/docker/env_setup/t2-vla.sh"
    exec /bin/bash -l "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
