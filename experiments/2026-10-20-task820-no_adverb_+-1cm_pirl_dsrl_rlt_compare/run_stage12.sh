#!/usr/bin/env bash
# Run this script ON THE HOST. Only training/checks run inside Docker.
# Usage: bash run_stage12.sh --dry-run | --check | --run
# --run requires a named host tmux session. No Stage 3 is launched.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT=2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare
CONFIG_GROUP=2026-10-20-pirl-dsrl-rlt-compare
SCENE_GROUP=gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm
IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.9.1
MODE="${1:---dry-run}"
METHODS=(pirl dsrl rlt)

die() { echo "ERROR: $*" >&2; exit 1; }
config_name() {
    case "$1" in
        pirl) echo "$CONFIG_GROUP/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_${2}_10_times_normalized_reward_6_env_gpu_delta_500g_xy_1cm" ;;
        rlt) echo "$CONFIG_GROUP/isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage${2}" ;;
        dsrl) echo "$CONFIG_GROUP/isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage${2}" ;;
    esac
}
case "$MODE" in
    --help|-h)
        echo "Host launcher: PiRL Stage 1/2 -> DSRL Stage 1/2 -> RLT Stage 1/2."
        echo "bash $0 --dry-run    Print the plan; no Docker or filesystem writes."
        echo "bash $0 --check      Pull/inspect image, snapshot and preflight; no training."
        echo "tmux new -s task820-compare"
        echo "bash $0 --run        Run all six stages inside that host tmux session."
        exit 0 ;;
    --dry-run|--check|--run) ;;
    *) die "Unknown option: $MODE; use --help" ;;
esac
[[ $# -le 1 ]] || die "Only one mode argument is accepted"
for method in "${METHODS[@]}"; do
    for stage in 1 2; do
        cfg="$(config_name "$method" "$stage")"
        [[ -f "$WORKSPACE/RLinf/examples/embodiment/config/$cfg.yaml" ]] || die "Missing config: $cfg"
    done
done
if [[ "$MODE" == --dry-run ]]; then
    echo "HOST Bash orchestrator; Docker image: $IMAGE"
    for method in "${METHODS[@]}"; do
        for stage in 1 2; do
            resume=never
            checkpoint=null
            reward=success
            if [[ "$stage" == 2 ]]; then
                resume=must
                checkpoint=global_step_200
                reward=success+normalized_force
            fi
            echo "$method stage$stage: target=$((stage * 200)), resume=$checkpoint, reward=$reward"
            echo "  docker compose run ... rlinf rlinf -lc 'python examples/embodiment/train_embodied_agent.py --config-path <saved-config> --config-name $(config_name "$method" "$stage")'"
            echo "  WANDB_RUN_ID=<persisted-$method-id> WANDB_RESUME=$resume"
        done
    done
    exit 0
fi
[[ "$MODE" != --run || -n "${TMUX:-}" ]] || die "Run --run inside a named HOST tmux session; see --help"
for tool in docker sha256sum flock find sort xargs diff cmp cp timeout pkill; do
    command -v "$tool" >/dev/null || die "Missing host command: $tool"
done
if [[ "$MODE" == --run ]]; then
    command -v nvidia-smi >/dev/null || die "Missing nvidia-smi"
    command -v tmux >/dev/null || die "Missing tmux"
fi

# Load credentials without printing them. Container entrypoint also sources this file.
ENV_FILE="${TABERO_ENV_FILE:-$WORKSPACE/.env}"
[[ -f "$ENV_FILE" ]] || die "Missing credential file: $ENV_FILE"
set -a
set +u
# shellcheck disable=SC1090
source "$ENV_FILE"
set -u
set +a
export TABERO_ENV_FILE="$ENV_FILE"
export TABERO_HOST="${TABERO_HOST:-$WORKSPACE}"
export TABERO_MODELS_HOST="${TABERO_MODELS_HOST:-$WORKSPACE/models}"
export TABERO_DATASETS_HOST="${TABERO_DATASETS_HOST:-$WORKSPACE/datasets}"
export TABERO_RESULTS_HOST="${TABERO_RESULTS_HOST:-$WORKSPACE/results}"
export TABERO_RECORD_HOST="${TABERO_RECORD_HOST:-$WORKSPACE/Record}"
for directory in "$TABERO_MODELS_HOST" "$TABERO_DATASETS_HOST" "$TABERO_RESULTS_HOST" "$TABERO_RECORD_HOST"; do
    [[ -d "$directory" && "$directory" == /* ]] || die "Mount paths must exist and be absolute: $directory"
done
CONTROL="$TABERO_RESULTS_HOST/$EXPERIMENT/_launcher"
CONTAINER_CONTROL="/root/VTLA-RL/results/$EXPERIMENT/_launcher"
mkdir -p "$CONTROL"
exec 9>"$CONTROL/.lock"
flock -n 9 || die "Another launcher is using $CONTROL"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE"
fi
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
if [[ -f "$CONTROL/image_id" ]]; then
    [[ "$(cat "$CONTROL/image_id")" == "$IMAGE_ID" ]] || die "Image changed since this experiment snapshot"
else
    echo "$IMAGE_ID" >"$CONTROL/image_id"
fi
export TABERO_IMAGE="$IMAGE_ID"
COMPOSE=(docker compose --project-directory "$WORKSPACE" --env-file "$ENV_FILE" -f "$WORKSPACE/docker-compose.yaml")
"${COMPOSE[@]}" config --quiet

snapshot() {
    if [[ -e "$2" ]]; then
        diff -qr --exclude=__pycache__ "$1" "$2" || die "Existing snapshot differs: $2; do not overwrite an active run"
    else
        cp -aL "$1" "$2"
    fi
}
snapshot "$WORKSPACE/RLinf/examples/embodiment/config" "$CONTROL/config"
snapshot "$WORKSPACE/Tabero_X/benchmarks/datasets/realworld/config" "$CONTROL/env"
if [[ -f "$CONTROL/run_stage12.sh" ]]; then
    cmp -s "$SCRIPT_DIR/run_stage12.sh" "$CONTROL/run_stage12.sh" || die "Launcher changed since the snapshot"
else
    cp "$SCRIPT_DIR/run_stage12.sh" "$CONTROL/run_stage12.sh"
fi
(
    cd "$WORKSPACE"
    find RLinf/rlinf RLinf/examples/embodiment T2-VLA/src T2-VLA/packages/openpi-client/src Tabero_X/source IsaacLab/source docker/entrypoint.sh docker/env_setup -type f \( -name '*.py' -o -name '*.sh' -o -name '*.toml' \) ! -path '*/__pycache__/*' ! -path '*/.venv/*' ! -path '*/.git/*' -print0 |
        LC_ALL=C sort -z | xargs -0 sha256sum
) >"$CONTROL/sources.current.sha256"
if [[ -f "$CONTROL/sources.sha256" ]]; then
    cmp -s "$CONTROL/sources.sha256" "$CONTROL/sources.current.sha256" || die "Training source changed since the snapshot"
else
    cp "$CONTROL/sources.current.sha256" "$CONTROL/sources.sha256"
fi

# Container-only checker: parses Hydra and verifies checkpoint artifacts.
# The host orchestrator and all stage ordering remain in THIS shell script.
cat >"$CONTROL/check_stage12.py" <<'PY'
import hashlib, json, os, re, shlex, sys
from pathlib import Path

IMAGE = "ccr.ccs.tencentyun.com/vtla/vtla:0.9.1"


EXPERIMENT = "2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare"


CONFIG_GROUP = "2026-10-20-pirl-dsrl-rlt-compare"


SCENE_GROUP = "gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm"


METHODS = ("pirl", "dsrl", "rlt")


CONTAINER_ROOT = Path("/root/VTLA-RL")


CONTAINER_CONTROL = CONTAINER_ROOT / "results" / EXPERIMENT / "_launcher"


SOURCE_TREES = (
    "RLinf/rlinf", "RLinf/examples/embodiment", "T2-VLA/src",
    "T2-VLA/packages/openpi-client/src", "Tabero_X/source", "IsaacLab/source",
    "docker/entrypoint.sh", "docker/env_setup",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def source_hashes(root):
    result = {}
    for relative in SOURCE_TREES:
        base = root / relative
        require(base.exists(), f"Missing source tree: {base}")
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            rel = path.relative_to(root)
            if (path.is_file() and path.suffix in {".py", ".sh", ".toml"}
                    and not {"__pycache__", ".venv", ".git"}.intersection(rel.parts)):
                result[str(rel)] = digest(path)
    return result


def config_names(root):
    result = {}
    for method in METHODS:
        for stage in (1, 2):
            paths = [p for p in (root / CONFIG_GROUP).glob(f"*pi05_{method}_*.yaml")
                     if re.search(rf"stage_?{stage}(?:_|\.yaml)", p.name)]
            require(len(paths) == 1, f"Expected one {method} Stage {stage} YAML: {paths}")
            result[f"{method}:{stage}"] = f"{CONFIG_GROUP}/{paths[0].stem}"
    return result


def training_args(names, method, stage):
    scene = CONTAINER_CONTROL / "env" / SCENE_GROUP / f"stage{stage}"
    return [str(CONTAINER_ROOT / "RLinf/.venv/bin/python"),
            str(CONTAINER_ROOT / "RLinf/examples/embodiment/train_embodied_agent.py"),
            "--config-path", str(CONTAINER_CONTROL / "config"),
            "--config-name", names[f"{method}:{stage}"],
            f"env.train.init_params.realworld_config_dir={scene}",
            f"env.eval.init_params.realworld_config_dir={scene}"]


def composed_config(method, stage):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    names = config_names(CONTAINER_CONTROL / "config")
    args = training_args(names, method, stage)
    with initialize_config_dir(config_dir=str(CONTAINER_CONTROL / "config"), version_base="1.1"):
        cfg = compose(config_name=names[f"{method}:{stage}"], overrides=args[6:])
    OmegaConf.resolve(cfg)
    return cfg


def preflight():
    """No Ray, simulator, model load, or W&B run is started here."""
    from omegaconf import OmegaConf

    expected = {}
    for line in (CONTAINER_CONTROL / "sources.sha256").read_text().splitlines():
        checksum, path = line.split("  ", 1)
        expected[path] = checksum
    actual = source_hashes(CONTAINER_ROOT)
    mismatches = sorted(k for k in expected.keys() | actual.keys()
                        if expected.get(k) != actual.get(k))
    require(not mismatches, "Image/local training code mismatch. Stop and confirm which "
            "version to use or rebuild the image; no automatic synchronization.\n"
            + "\n".join(mismatches[:40]))
    require(os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is missing from the mounted .env")
    base = CONTAINER_ROOT / "models/VTLA-RL-sft-lora-franka-no-adverb-05-effort/step_30000"
    encoder = CONTAINER_ROOT / "models/VTLA-RL-sft-lora-franka-no-adverb-05-effort-rlt-stage1/rlt_encoder.safetensors"
    contracts = {}
    for method in METHODS:
        for stage in (1, 2):
            cfg = composed_config(method, stage)
            model = cfg.rollout.rlt_feature_model if method == "rlt" else cfg.actor.model
            name = f"2026-10-20-task820-no_adverb_xy_1cm_{method}_compare"
            log_root = CONTAINER_ROOT / "results" / EXPERIMENT / method
            require(cfg.runner.logger.experiment_name == name, f"Unexpected run name: {method}")
            require(cfg.runner.logger.log_path == str(log_root), "Unexpected log directory")
            require(set(cfg.runner.logger.logger_backends) == {"tensorboard", "wandb"}, "Both loggers required")
            require(cfg.runner.max_steps == cfg.runner.max_epochs == stage * 200, "Wrong stage target")
            resume = None if stage == 1 else str(log_root / name / "checkpoints/global_step_200")
            require(cfg.runner.resume_dir == resume, "Stage 2 must resume its own Stage 1")
            require(dict(cfg.cluster.component_placement) == {"env": "0-5", "rollout": "0-7", "actor": "0-7"}, "GPU placement changed")
            require(cfg.cluster.num_nodes == 1, "This launcher is single-node")
            require(not cfg.actor.fsdp_config.get("gradient_checkpointing", False)
                    and not model.get("gradient_checkpointing", False), "Expected gradient checkpointing disabled")
            require(not model.openpi.discrete_state_input and "no_state" in model.openpi.config_name,
                    "Expected no-state VLA")
            require(model.openpi.tactile_prefix_dim_in == 7920, "Expected TacField input")
            require(model.model_path == str(base) and base.is_dir(), "SFT model directory missing or changed")
            require(Path(model.openpi_data.norm_stats_path).is_file(), "Missing normalization statistics")
            if method == "rlt":
                require(model.rlt_encoder_path == str(encoder) and encoder.is_file(), "Missing RLT encoder")
            for split in ("train", "eval"):
                env = cfg.env[split]
                scene = Path(env.init_params.realworld_config_dir) / "gentle_grasp.json"
                require(scene.is_file(), f"Missing scene: {scene}")
                require(env.init_params.success.normalize_effort_reward.enabled == (stage == 2),
                        "Expected success-only Stage 1 and normalized force reward in Stage 2")
                require(Path(env.init_params.realworld_assets_dir).is_dir(), "Missing USD assets in image")
            contracts[f"{method}:{stage}"] = OmegaConf.to_container(cfg, resolve=True)
    settings = []
    for method in METHODS:
        logger = contracts[f"{method}:1"]["runner"]["logger"]
        for field in ("experiment_name", "log_path", "project_name", "wandb_entity"):
            settings.append(f"{field.upper()}_{method.upper()}=" + shlex.quote(str(logger[field])))
    (CONTAINER_CONTROL / "settings.env").write_text("\n".join(settings) + "\n")
    write_json(CONTAINER_CONTROL / "contracts.json", contracts)
    print(f"PREFLIGHT_PASS: {len(expected)} source files, six configs; gradient_checkpointing=disabled")


def checkpoint_check(method, stage):
    """Check artifact structure and steps; Stage 2 performs the real distributed restore."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    cfg = composed_config(method, stage)
    step = stage * 200
    checkpoint = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name / f"checkpoints/global_step_{step}"
    actor = checkpoint / "actor"

    def nonempty(path):
        require(path.is_file() and path.stat().st_size > 0, f"Missing/empty checkpoint artifact: {path}")

    def state(path):
        nonempty(path)
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    def dcp(directory):
        nonempty(directory / ".metadata")
        metadata = FileSystemReader(str(directory)).read_metadata()
        keys = list(metadata.state_dict_metadata)
        for field in ("model", "optimizers", "lr_schedulers", "rng"):
            require(any(k == f"fsdp_checkpoint.{field}" or k.startswith(f"fsdp_checkpoint.{field}.") for k in keys),
                    f"DCP missing {field}: {directory}")
        for item in metadata.storage_data.values():
            nonempty(directory / item.relative_path)
            require((directory / item.relative_path).stat().st_size >= item.offset + item.length,
                    f"Truncated DCP shard: {item.relative_path}")

    export = state(actor / "model_state_dict/trainable_weights.pt")
    require(export["metadata"]["step"] == step, "Trainable export step mismatch")
    del export
    require(read_json(checkpoint / "multi_task_state.json")["step"] == step, "Task controller step mismatch")
    if cfg.actor.fsdp_config.checkpoint_format == "dcp":
        dcp(actor / "dcp_checkpoint")
    else:
        for rank in range(8):
            nonempty(actor / f"local_shard_checkpoint/checkpoint_rank_{rank}.pt")
    if method != "pirl":
        sac = actor / "sac_components"
        nonempty(sac / "target_model/checkpoint_rank_0.pt")
        if cfg.algorithm.entropy_tuning.alpha_type != "fixed_alpha":
            dcp(sac / "alpha/dcp_checkpoint")
        for rank in range(8):
            sidecar = state(sac / f"multi_task_rank_{rank}.pt")
            require(sidecar["step"] == step and sidecar["rank"] == rank and sidecar["world_size"] == 8,
                    "SAC sidecar step/rank mismatch")
            replay = sac / f"replay_buffer/rank_{rank}"
            require(read_json(replay / "metadata.json")["multi_task_format"] == "sac_multitask_replay_v1",
                    "Unexpected replay format")
            index = read_json(replay / "trajectory_index.json")
            require(index["trajectory_id_list"], "Empty replay checkpoint")
            for tid in index["trajectory_id_list"]:
                info = index["trajectory_index"][str(tid)]
                nonempty(replay / f"trajectory_{tid}_{info['model_weights_id']}.pt")
            if method == "rlt":
                schedule = state(sac / f"rlt_schedule/checkpoint_rank_{rank}.pt")
                require(schedule["global_step"] == step and schedule["update_step"] == sidecar["update_step"],
                        "RLT schedule step mismatch")
    print(f"CHECKPOINT_ARTIFACTS_PASS {method} stage{stage}: {checkpoint}")


if __name__ == "__main__":
    os.environ["ROOT"] = str(CONTAINER_ROOT)
    if sys.argv[1] == "preflight":
        preflight()
    else:
        checkpoint_check(sys.argv[2], int(sys.argv[3]))
PY
container_check() {
    "${COMPOSE[@]}" run --pull never --rm -T rlinf rlinf -lc 'exec "$ROOT/RLinf/.venv/bin/python" "$@"' bash "$CONTAINER_CONTROL/check_stage12.py" "$@"
}
container_check preflight 2>&1 | tee "$CONTROL/preflight.log"
[[ "$MODE" == --run ]] || exit 0
# This contains only shell-quoted experiment names/paths and W&B entity/project.
# shellcheck disable=SC1091
source "$CONTROL/settings.env"

idle_gpus() {
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits >"$CONTROL/gpus.before.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits >"$CONTROL/processes.before.csv"
    local index gpu memory count=0
    while IFS=, read -r index gpu memory; do
        index="${index// /}"
        gpu="${gpu// /}"
        memory="${memory// /}"
        if (( index < 8 )); then
            ((count += 1))
            [[ "$memory" =~ ^[0-9]+$ ]] && (( memory < 1024 )) || die "GPU $index is already using $memory MiB"
            if grep -Fq "$gpu," "$CONTROL/processes.before.csv"; then
                die "GPU $index has an existing compute process; no jobs will be displaced"
            fi
        fi
    done <"$CONTROL/gpus.before.csv"
    [[ "$count" == 8 ]] || die "The YAML requires eight GPUs (actor/rollout 0-7, env 0-5)"
}

if [[ -f "$CONTROL/record_path" ]]; then
    REPORT="$(cat "$CONTROL/record_path")"
else
    mkdir -p "$TABERO_RECORD_HOST/experiments"
    REPORT="$TABERO_RECORD_HOST/experiments/$(date +%Y-%m-%d-%H-%M-%S)-task820-no_adverb-pirl_dsrl_rlt-1cm_compare.md"
    echo "$REPORT" >"$CONTROL/record_path"
    {
        echo "# $EXPERIMENT"
        echo
        echo "Started: $(date -Is)"
        echo "Image: $IMAGE ($IMAGE_ID)"
        echo "Host tmux: $(tmux display-message -p '#S:#I.#P')"
        echo "Order: PiRL 1/2 -> DSRL 1/2 -> RLT 1/2."
        echo "No-state VLA; TacField; synchronous RLinf; XY +/-1 cm."
        echo "Stage 1: 0-200, success reward. Stage 2: 200-400, +500 g and normalized force reward."
        echo "Gradient checkpointing: disabled. Logging: TensorBoard + W&B."
        echo "Full parameters, model paths, and copied scenes: $CONTROL/contracts.json, $CONTROL/config, $CONTROL/env"
        echo "Launcher/source snapshots and stage logs: $CONTROL"
    } >"$REPORT"
fi
for method in "${METHODS[@]}"; do
    key="EXPERIMENT_NAME_${method^^}"; name="${!key}"
    key="LOG_PATH_${method^^}"; log_path="${!key}"
    key="WANDB_ENTITY_${method^^}"; entity="${!key}"
    key="PROJECT_NAME_${method^^}"; project="${!key}"
    if [[ ! -f "$CONTROL/$method.wandb_id" ]]; then
        method_output="$TABERO_RESULTS_HOST/$EXPERIMENT/$method"
        [[ ! -d "$method_output" || -z "$(ls -A "$method_output")" ]] || die "Existing results without a saved W&B ID: $method_output"
        read -r run_id </proc/sys/kernel/random/uuid
        run_id="${run_id//-/}"
        echo "${run_id:0:8}" >"$CONTROL/$method.wandb_id"
        run_id="$(cat "$CONTROL/$method.wandb_id")"
        url="https://wandb.ai/$entity/$project/runs/$run_id"
        echo "$url" >"$CONTROL/$method.wandb_url.txt"
        {
            echo
            echo "## $method"
            echo "Experiment: $name"
            echo "W&B (one ID for both stages; planned until training initializes): $url"
            echo "Results: $log_path"
        } >>"$REPORT"
    fi
done

ACTIVE_CONTAINER=
ACTIVE_STAGE_DIR=
MONITOR_PID=
START_SECONDS=$SECONDS
stop_monitor() {
    [[ -n "$MONITOR_PID" ]] || return 0
    pkill -TERM -P "$MONITOR_PID" 2>/dev/null || true
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
}
finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        docker stop --time 60 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
    stop_monitor
    if [[ -n "$ACTIVE_STAGE_DIR" ]]; then
        echo failed >"$ACTIVE_STAGE_DIR/status"
        echo "$rc" >"$ACTIVE_STAGE_DIR/exit_code"
        echo "Interrupted/failed: $ACTIVE_STAGE_DIR; exit=$rc; $(date -Is)" >>"$REPORT"
    fi
    echo "Launcher invocation ended: $(date -Is); duration=$((SECONDS - START_SECONDS)) s; exit=$rc" >>"$REPORT"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

for method in "${METHODS[@]}"; do
    run_id="$(cat "$CONTROL/$method.wandb_id")"
    key="EXPERIMENT_NAME_${method^^}"; name="${!key}"
    key="LOG_PATH_${method^^}"; log_path="${!key}"
    key="WANDB_ENTITY_${method^^}"; entity="${!key}"
    key="PROJECT_NAME_${method^^}"; project="${!key}"
    for stage in 1 2; do
        stage_dir="$CONTROL/$method-stage$stage"
        if [[ -f "$stage_dir/status" ]]; then
            [[ "$(cat "$stage_dir/status")" == complete ]] || die "$method Stage $stage was interrupted/failed; inspect its logs and review recovery before restarting"
            container_check checkpoint "$method" "$stage"
            echo "SKIP verified completed stage: $method $stage"
            continue
        fi
        [[ "$stage" != 2 ]] || container_check checkpoint "$method" 1
        idle_gpus
        mkdir -p "$stage_dir"
        ACTIVE_STAGE_DIR="$stage_dir"
        ACTIVE_CONTAINER="task820-$method-stage$stage-$run_id"
        cfg="$(config_name "$method" "$stage")"
        wandb_resume=never
        [[ "$stage" != 2 ]] || wandb_resume=must
        COMMAND=("${COMPOSE[@]}" run --pull never --rm -T --name "$ACTIVE_CONTAINER"
            -e "WANDB_RUN_ID=$run_id" -e "WANDB_RESUME=$wandb_resume" -e WANDB_MODE=online
            rlinf rlinf -lc '
set -euo pipefail
# Set identity AFTER entrypoint reads .env; use the host-persisted ID.
export WANDB_RUN_ID="$1" WANDB_RESUME="$2" WANDB_MODE=online
export WANDB_PROJECT="$7" WANDB_ENTITY="$8" PYTHONUNBUFFERED=1
unset RAY_ADDRESS
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Use YAML placement, not CUDA_VISIBLE_DEVICES" >&2; exit 1; }
cd "$ROOT/RLinf"
echo "gradient_checkpointing=disabled; W&B run=$WANDB_RUN_ID; resume=$WANDB_RESUME"
"$ROOT/RLinf/.venv/bin/python" examples/embodiment/train_embodied_agent.py --config-path "$3/config" --config-name "$4" "env.train.init_params.realworld_config_dir=$3/env/$5/stage$6" "env.eval.init_params.realworld_config_dir=$3/env/$5/stage$6"
' bash "$run_id" "$wandb_resume" "$CONTAINER_CONTROL" "$cfg" "$SCENE_GROUP" "$stage" "$project" "$entity")
        {
            echo '#!/usr/bin/env bash'
            printf '%q ' "${COMMAND[@]}"
            printf '\n'
        } >"$stage_dir/command.sh"
        echo running >"$stage_dir/status"
        date -Is >"$stage_dir/started_at"
        stage_start=$SECONDS
        echo "RUN $method Stage $stage: $(cat "$CONTROL/$method.wandb_url.txt")"
        echo "Log: $stage_dir/launcher.log"
        "${COMMAND[@]}" > >(tee "$stage_dir/launcher.log") 2>&1 &
        docker_pid=$!
        (
            trap - EXIT
            trap 'jobs -pr | xargs -r kill 2>/dev/null; exit 0' TERM INT
            while kill -0 "$docker_pid" 2>/dev/null; do
                date -Is
                timeout 20 nvidia-smi || true
                sleep 15
            done
        ) >"$stage_dir/gpu.log" 2>&1 &
        MONITOR_PID=$!
        rc=0
        wait "$docker_pid" || rc=$?
        stop_monitor
        ACTIVE_CONTAINER=
        echo "$rc" >"$stage_dir/docker_exit_code"
        [[ "$rc" == 0 ]] || die "$method Stage $stage failed (exit $rc); see $stage_dir/launcher.log"
        container_check checkpoint "$method" "$stage" 2>&1 | tee "$stage_dir/checkpoint.log"
        "${COMPOSE[@]}" run --pull never --rm -T rlinf rlinf -lc '
"$ROOT/RLinf/.venv/bin/python" - "$1" "$2" <<'"'"'PY'"'"'
import sys, time, wandb
for attempt in range(6):
    try:
        run = wandb.Api().run(sys.argv[1])
        if run.name != sys.argv[2]:
            raise RuntimeError("W&B experiment name mismatch")
        print("WANDB_VERIFIED", run.url)
        break
    except Exception:
        if attempt == 5:
            raise
        time.sleep(5)
PY
' bash "$entity/$project/$run_id" "$name" 2>&1 | tee "$stage_dir/wandb.log"
        date -Is >"$stage_dir/ended_at"
        echo "$((SECONDS - stage_start))" >"$stage_dir/duration_seconds"
        echo 0 >"$stage_dir/exit_code"
        echo complete >"$stage_dir/status"
        ACTIVE_STAGE_DIR=
        {
            echo
            echo "### $method Stage $stage complete"
            echo "Started: $(cat "$stage_dir/started_at"); ended: $(cat "$stage_dir/ended_at")"
            echo "Duration: $(cat "$stage_dir/duration_seconds") s; exit=0"
            echo "Logs / exact launch command: $stage_dir"
            echo "Checkpoint: $log_path/$name/checkpoints/global_step_$((stage * 200))"
            echo "TensorBoard: $log_path/tensorboard"
            cat "$stage_dir/wandb.log"
        } >>"$REPORT"
    done
done
echo "All six stages completed. Record: $REPORT"
