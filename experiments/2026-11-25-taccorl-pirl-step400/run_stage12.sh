#!/usr/bin/env bash
# Host launcher; Python training runs in the source-audited 0.11.1 container.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT=2026-11-25-taccorl-pirl-step400
OUT="$WORKSPACE/results/$EXPERIMENT/8gpu"
CONTROL="$OUT/_launcher"
CONTAINER_ROOT=/root/VTLA-RL
RUN_DIR="$CONTAINER_ROOT/results/$EXPERIMENT/8gpu"
IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.11.1
CONTAINER=taccorl-pirl-step400-20261125-8gpu
MODE="${1:---dry-run}"
DOCKER=(docker --context default)
die() { echo "ERROR: $*" >&2; exit 1; }
case "$MODE" in
  --dry-run)
    echo 'Target host -> Docker 0.11.1 -> synchronous RLinf; no-state; gradient_checkpointing=enabled'
    echo 'env/actor/rollout 0-7; 128 train env (64 per task); evaluation disabled; micro48 x world8 = global384'
    echo 'Coca-Cola + Vitasoy; real-data SFT weight1; stage1 0->200 then full resume 200->400'
    echo 'Run --check, then --smoke and --run in a named host tmux session.'
    exit 0 ;;
  --check|--smoke|--run) ;;
  *) die 'Use --dry-run, --check, --smoke or --run' ;;
esac
[[ $# == 1 ]] || die 'Pass one mode'
if [[ "$MODE" != --check ]]; then
  [[ -n "${TMUX:-}" ]] || die 'Use a named host tmux session'
fi
mkdir -p "$CONTROL"
exec 9>"$CONTROL/launcher.lock"
flock -n 9 || die 'This experiment already has a launcher'
ENV_FILE="${TABERO_ENV_FILE:-$WORKSPACE/.env}"
[[ -f "$ENV_FILE" ]] || die "Missing $ENV_FILE"
set -a
# The shared file also contains unrelated interactive commands.
source <(sed -n -E '/^[[:space:]]*(export[[:space:]]+)?(HF_TOKEN|SWANLAB_API_KEY|HTTP_PROXY|HTTPS_PROXY|NO_PROXY|http_proxy|https_proxy|no_proxy)=/p' "$ENV_FILE")
set +a
: "${SWANLAB_API_KEY:?Set SWANLAB_API_KEY in .env}"
: "${TACCORL_GPU_UUIDS:?Set the eight target GPU UUIDs in PCI order}"
IFS=, read -r -a gpu_uuids <<< "$TACCORL_GPU_UUIDS"
[[ ${#gpu_uuids[@]} == 8 ]] || die 'This configuration requires eight GPU UUIDs'
IMAGE_ID="$("${DOCKER[@]}" image inspect "$IMAGE" --format '{{.Id}}')"
mkdir -p "$OUT/image" "$OUT/tools" "$OUT/validation"
if [[ ! -s "$OUT/image/image_id" ]]; then
  for name in source_manifest source_revisions; do
    "${DOCKER[@]}" run --rm --network none --entrypoint cat "$IMAGE_ID" "/opt/vtla/$name.json" > "$OUT/image/$name.json.tmp"
    mv "$OUT/image/$name.json.tmp" "$OUT/image/$name.json"
  done
  printf '%s\n' "$IMAGE_ID" > "$OUT/image/image_id"
fi
[[ -s "$OUT/image/image_id" && "$(cat "$OUT/image/image_id")" == "$IMAGE_ID" ]] || die 'Image identity is not audited'
for name in preflight audit_checkpoint audit_cloud audit_export poll; do
  cp "$SCRIPT_DIR/tools/$name.py" "$OUT/tools/$name.py"
done
export TACCORL_PIRL_MODEL_PATH="$CONTAINER_ROOT/results/2026-11-24-task927-firm-taccorl-sft-30k/step_30000"
export TACCORL_REAL_SFT_DATA_PATH="$CONTAINER_ROOT/results/2026-11-24-task927-firm-taccorl-sft-30k/data/realworld_task927_firm_4x4_model_coordinates"
export TACCORL_NORM_STATS_PATH="$TACCORL_PIRL_MODEL_PATH/assets/pi05_horizon50_taccorl_task927_replay/norm_stats.json"
export TACCORL_RUN_DIR="$RUN_DIR"
if [[ ! -d "$OUT/config" ]]; then
  cp -a "$WORKSPACE/RLinf/examples/embodiment/config" "$OUT/config"
fi
cmp "$WORKSPACE/RLinf/examples/embodiment/config/$EXPERIMENT/stage1.yaml" "$OUT/config/$EXPERIMENT/stage1.yaml"
cmp "$WORKSPACE/RLinf/examples/embodiment/config/$EXPERIMENT/stage2.yaml" "$OUT/config/$EXPERIMENT/stage2.yaml"
for stage in stage1 stage2; do
  if [[ ! -d "$OUT/env/$stage" ]]; then
    mkdir -p "$OUT/env"
    cp -a "$WORKSPACE/Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/$stage" "$OUT/env/$stage"
  fi
done
if [[ ! -s "$CONTROL/run_ids.json" ]]; then
  python3 - "$CONTROL/run_ids.json" <<'PY'
import json, secrets, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({name: secrets.token_hex(4) for name in ('smoke', 'train')}, indent=2))
PY
fi
if "${DOCKER[@]}" container inspect "$CONTAINER" >/dev/null 2>&1; then
  [[ "$("${DOCKER[@]}" inspect "$CONTAINER" --format '{{.Image}}')" == "$IMAGE_ID" ]] || die 'Existing container uses another image'
  [[ "$("${DOCKER[@]}" inspect "$CONTAINER" --format '{{.State.Running}}')" == true ]] || "${DOCKER[@]}" start "$CONTAINER" >/dev/null
else
  mkdir -p "$OUT/runtime/tmp" "$WORKSPACE/Record"
  "${DOCKER[@]}" run -d --name "$CONTAINER" \
    --gpus "\"device=$TACCORL_GPU_UUIDS\"" --network host --ipc host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TACCORL_RUN_DIR -e TACCORL_PIRL_MODEL_PATH -e TACCORL_REAL_SFT_DATA_PATH -e TACCORL_NORM_STATS_PATH \
    -v "$WORKSPACE/results:$CONTAINER_ROOT/results" \
    -v "$WORKSPACE/datasets:$CONTAINER_ROOT/datasets:ro" \
    -v "$WORKSPACE/models:$CONTAINER_ROOT/models:ro" \
    -v "$WORKSPACE/Record:$CONTAINER_ROOT/Record" \
    -v "$OUT/runtime/tmp:/tmp" \
    --entrypoint sleep "$IMAGE_ID" infinity >"$CONTROL/container_id"
fi
"${DOCKER[@]}" exec "$CONTAINER" bash -lc \
  'source /root/VTLA-RL/docker/env_setup/rlinf.sh; export PYTHONPATH="$ROOT/T2-VLA/src:$ROOT/Tabero_X:$PYTHONPATH"; python "$TACCORL_RUN_DIR/tools/preflight.py"' \
  2>&1 | tee "$CONTROL/preflight.log"
[[ "$MODE" != --check ]] || exit 0

checkpoint_path() { echo "$RUN_DIR/$1/$2/checkpoints/global_step_$3"; }
audit_checkpoint() {
  "${DOCKER[@]}" exec "$CONTAINER" bash -lc \
    'source /root/VTLA-RL/docker/env_setup/rlinf.sh; python "$TACCORL_RUN_DIR/tools/audit_checkpoint.py" "$@"' \
    bash "$1" "$2" "$3"
}
run_job() {
  local kind="$1" stage="$2" target="$3" resume="$4" label="$5"
  shift 5
  local id name log_path timestamp start rc
  id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CONTROL/run_ids.json" "$kind")"
  name="$EXPERIMENT"
  [[ "$kind" == train ]] || name="$EXPERIMENT-$kind"
  log_path="$RUN_DIR/$kind"
  mkdir -p "$OUT/$kind/attempts"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  start=0
  if [[ "$resume" != null ]]; then start="${resume##*_}"; fi
  export SWANLAB_RUN_ID="$id"
  export SWANLAB_RESUME=allow
  [[ "$resume" == null ]] || export SWANLAB_RESUME=must
  printf '%s\n' "gradient_checkpointing=enabled; phase=$label; step=$start->$target; SwanLab ID=$id"
  set +e
  "${DOCKER[@]}" exec \
    -e SWANLAB_API_KEY -e HF_TOKEN -e SWANLAB_RUN_ID -e SWANLAB_RESUME \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY -e http_proxy -e https_proxy -e no_proxy \
    "$CONTAINER" bash -lc '
      set -euo pipefail
      source /root/VTLA-RL/docker/env_setup/rlinf.sh
      export PYTHONPATH="$ROOT/T2-VLA/src:$ROOT/Tabero_X:$PYTHONPATH"
      exec python examples/embodiment/train_embodied_agent.py \
        --config-path "$TACCORL_RUN_DIR/config" "$@"
    ' bash --config-name "$EXPERIMENT/$stage" \
    "runner.logger.log_path=$log_path" "runner.logger.experiment_name=$name" \
    "runner.max_steps=$target" "runner.max_epochs=$target" "runner.resume_dir=$resume" \
    "runner.logger.log_start_step=$start" "$@" \
    2>&1 | tee "$OUT/$kind/attempts/$timestamp-$label.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" >"$CONTROL/$label.exit_code"
  [[ "$rc" == 0 ]] || die "$label failed with exit $rc; inspect its log before recovery"
  audit_checkpoint "$(checkpoint_path "$kind" "$name" "$target")" "$kind" "$target"
}
if [[ "$MODE" == --smoke ]]; then
  if [[ ! -s "$OUT/validation/smoke-step2.json" ]]; then
    run_job smoke stage1 2 null smoke-fresh runner.save_interval=2
  fi
  if [[ ! -s "$OUT/validation/smoke-step3.json" ]]; then
    run_job smoke stage2 3 "$(checkpoint_path smoke "$EXPERIMENT-smoke" 2)" smoke-resume runner.save_interval=2
  fi
  "${DOCKER[@]}" exec -e SWANLAB_API_KEY "$CONTAINER" bash -lc '
    source /root/VTLA-RL/docker/env_setup/rlinf.sh
    python "$TACCORL_RUN_DIR/tools/audit_cloud.py" smoke 3
  ' > "$OUT/validation/smoke-cloud-step3.log" 2>&1
  "${DOCKER[@]}" stop "$CONTAINER"
  printf '%s\n' "$IMAGE_ID" >"$CONTROL/smoke_passed"
  exit 0
fi
[[ -s "$CONTROL/smoke_passed" && "$(cat "$CONTROL/smoke_passed")" == "$IMAGE_ID" ]] || die 'Run and pass --smoke with this image first'
for stage in stage1 stage2; do
  target=200
  [[ "$stage" == stage1 ]] || target=400
  if [[ -s "$OUT/validation/train-step$target.json" ]]; then continue; fi
  latest="$(python3 - "$OUT/train/$EXPERIMENT/checkpoints" "$target" <<'PY'
from pathlib import Path
import sys
points=sorted((int(p.name.split('_')[-1]), p) for p in Path(sys.argv[1]).glob('global_step_*') if p.is_dir())
print(max((s for s,p in points if s<=int(sys.argv[2])), default=0))
PY
  )"
  resume=null
  if (( latest > 0 )); then
    resume="$(checkpoint_path train "$EXPERIMENT" "$latest")"
    audit_checkpoint "$resume" train "$latest"
  fi
  [[ "$stage" != stage2 || "$latest" -ge 200 ]] || die 'Stage2 requires complete step200'
  if (( latest < target )); then run_job train "$stage" "$target" "$resume" "$stage"; fi
done
if [[ ! -s "$OUT/validation/export-step400.json" ]]; then
  "${DOCKER[@]}" exec "$CONTAINER" bash -lc '
    set -euo pipefail
    source /root/VTLA-RL/docker/env_setup/rlinf.sh
    export PYTHONPATH="$ROOT/T2-VLA/src:$ROOT/Tabero_X:$PYTHONPATH"
    python -m rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla \
      --train_config_path "$TACCORL_RUN_DIR/train/tensorboard/config.yaml" \
      --ckpt_path "$TACCORL_RUN_DIR/train/2026-11-25-taccorl-pirl-step400/checkpoints/global_step_400/actor/model_state_dict/trainable_weights.pt" \
      --output_dir "$TACCORL_RUN_DIR/exports/step_400" --no_save_adapter
    source /root/VTLA-RL/docker/env_setup/t2-vla.sh
    export JAX_PLATFORMS=cpu
    python "$TACCORL_RUN_DIR/tools/audit_export.py"
  ' > "$OUT/validation/export-step400.log" 2>&1
fi
"${DOCKER[@]}" exec -e SWANLAB_API_KEY "$CONTAINER" bash -lc '
  source /root/VTLA-RL/docker/env_setup/rlinf.sh
  python "$TACCORL_RUN_DIR/tools/audit_cloud.py" train 400
' > "$OUT/validation/train-cloud-step400.log" 2>&1
"${DOCKER[@]}" stop "$CONTAINER"
date -u +%FT%TZ > "$CONTROL/completed_at"
echo 'Training complete: step400 checkpoint, merged weights, CPU load and cloud-log audit passed.'
