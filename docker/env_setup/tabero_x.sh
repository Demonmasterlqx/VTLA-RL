export ROOT=/root/VTLA-RL
cd "$ROOT/Tabero_X"
source "$ROOT/IsaacLab/.venv/bin/activate"

# Isaac Sim reads optional environment variables without nounset guards.
set +u
source "$ROOT/isaacsim/setup_python_env.sh"
set -u

export ISAAC_PATH="$ROOT/isaacsim"
export CARB_APP_PATH="$ROOT/isaacsim/kit"
export EXP_PATH="$ROOT/isaacsim/apps"
export PYTHONPATH="$ROOT/isaacsim/extscache/omni.warp.core-1.8.2+lx64:$ROOT/isaacsim/extscache/omni.warp-1.8.2:${PYTHONPATH:-}"
export HDF5_TRAJ_SOURCE_DIR="$ROOT/Tabero_X/benchmarks/datasets/realworld/assembled_hdf5"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

# "$ROOT/IsaacLab/.venv/bin/python" -u \
# "$ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py" \
#     --server-host 127.0.0.1 \
#     --server-port 10010 \
#     --task Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0 \
#     --task-domain realworld \
#     --task-suite gentle_grasp \
#     --task-id 6 \
#     --control-mode hybrid_tactile \
#     --hdf5-folder "$HDF5_TRAJ_SOURCE_DIR" \
#     --num-total-experiments 50 \
#     --max-inference-steps 30 \
#     --language_instruction "place the Vitasoy, and basket on the table" \
#     --replan-steps 10 \
#     --debug-mode 6 \
#     --debug-path "$ROOT/results/openpi_video" \
#     --debug-run-dir "$ROOT/results/openpi_video/artifacts" \
#     --no-smooth-action-chunks \
#     --device cuda:0 \
#     --headless
