export ROOT=/root/VTLA-RL
export ISAACSIM_PATH="$ROOT/isaacsim"
export ISAACLAB_PATH="$ROOT/IsaacLab"

cd "$ROOT/RLinf"
source "$ROOT/RLinf/.venv/bin/activate"

# Isaac Sim's script reads optional variables without nounset guards. Preserve
# the caller's strict-shell setting while loading the simulator runtime paths.
set +u
source "$ISAACSIM_PATH/setup_conda_env.sh"
set -u

export ISAAC_PATH="$ISAACSIM_PATH"
export CARB_APP_PATH="$ISAACSIM_PATH/kit"
export EXP_PATH="$ISAACSIM_PATH/apps"
# Isaac Sim adds bundled Python packages (including a different Torch) to
# PYTHONPATH. Resolve training dependencies from RLinf's environment first,
# then expose the separately managed IsaacLab environment for simulator-only
# dependencies such as flatdict. Keeping the IsaacLab site-packages after the
# RLinf site-packages prevents its Torch tuple from replacing RLinf's tuple.
ISAAC_ENV_SITE="$ROOT/IsaacLab/.venv/lib/python3.11/site-packages"
export PYTHONPATH="$ROOT/RLinf/.venv/lib/python3.11/site-packages:$ISAACSIM_PATH/extscache/omni.warp.core-1.8.2+lx64:$ISAACSIM_PATH/extscache/omni.warp-1.8.2:$ROOT/RLinf:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/Tabero_X/source/tac_manip:$ISAAC_ENV_SITE:${PYTHONPATH:-}"
export EMBODIED_PATH="$ROOT/RLinf/examples/embodiment"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
