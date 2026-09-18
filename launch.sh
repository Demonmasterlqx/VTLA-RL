#!/usr/bin/env bash
set -euo pipefail

# Two terminals:
#   SERVER_GPU=0 bash launch.sh server
#   CLIENT_GPU=1 bash launch.sh client 6
# IsaacLab and Tabero_X use the shared uv environment under IsaacLab/.venv;
# T2-VLA keeps its own uv environment.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL="${MODEL:-$ROOT/models/pi05_realworld_replay_task820_firm_mixed_tacfield_sft_step22000}"
PORT="${PORT:-8000}"

case "${1:-}" in
  server)
    cd "$ROOT/T2-VLA"
    exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${SERVER_GPU:-0}" \
      JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
      "$ROOT/T2-VLA/.venv/bin/python" - "$MODEL" "$PORT" <<'PY'
import dataclasses
import logging
import sys

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config

logging.basicConfig(level=logging.INFO, force=True)
checkpoint, port = sys.argv[1:]
cfg = config.get_config("pi05_lora_tacfield_realworld_replayed_task820")
cfg = dataclasses.replace(
    cfg,
    data=dataclasses.replace(
        cfg.data,
        assets=config.AssetsConfig(asset_id="pi05_horizon50_tacfield_task820_firm_mixed"),
    ),
)
policy = policy_config.create_trained_policy(cfg, checkpoint)
websocket_policy_server.WebsocketPolicyServer(
    policy=policy, host="0.0.0.0", port=int(port), metadata=policy.metadata,
).serve_forever()
PY
    ;;
  client)
    TASK_ID="${2:?用法：CLIENT_GPU=1 bash launch.sh client <Task ID>}"
    RUN_DIR="${RUN_DIR:-$ROOT/results/$(date +%Y-%m-%d_%H-%M-%S)-task${TASK_ID}-no_adverb-step22000-smoke}"
    mkdir -p "$RUN_DIR/empty_hdf5"
    cd "$ROOT/Tabero_X"
    # shellcheck disable=SC1091
    source "$ROOT/activate_uv.sh"
    env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
      CUDA_VISIBLE_DEVICES="${CLIENT_GPU:-1}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
      VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
      PYTHONUNBUFFERED=1 TACTILE_BACKEND=taxim_fots \
      HDF5_TRAJ_SOURCE_DIR="$ROOT/Tabero_X/benchmarks/datasets/realworld/assembled_hdf5" \
      REALWORLD_CONFIG_DIR="$ROOT/Tabero_X/benchmarks/datasets/realworld/config" \
      PYTHONPATH="$ROOT/Tabero_X/source/tac_manip:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}" \
      "$ROOT/IsaacLab/isaaclab.sh" -p -u \
      benchmarks/openpi/openpi_inference_client.py \
      --server_host "${SERVER_HOST:-127.0.0.1}" --server_port "$PORT" \
      --task Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0 \
      --control_mode hybrid_tactile --task_domain realworld --task_suite gentle_grasp \
      --task_id "$TASK_ID" --hdf5_folder "$RUN_DIR/empty_hdf5" \
      --num_total_experiments "${EPISODES:-1}" \
      --max_inference_steps "${MAX_INFERENCE_STEPS:-30}" --replan_steps 10 \
      --num_success_steps 8 --seed 11 --prompt_seed 0 \
      --debug_mode 6 --debug_path "$RUN_DIR/raw" --debug_run_dir "$RUN_DIR/raw/artifacts" \
      --tactile_output_type tactile_rgb --device cuda:0 --headless \
      --invert_gripper_output --no-smooth-action-chunks \
      2>&1 | tee "$RUN_DIR/client.log"
    ;;
  *)
    echo "用法：bash launch.sh server | bash launch.sh client <Task ID>"
    exit 2
    ;;
esac
