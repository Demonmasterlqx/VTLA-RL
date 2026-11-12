#!/usr/bin/env bash
# Run this script ON THE HOST. Only training/checks run inside Docker.
# Usage: bash run_stage12.sh --dry-run | --check | --run
# --run requires a named host tmux session. No Stage 3 is launched.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT=2026-11-12-tacimg-rlt-training
CONFIG_GROUP=2026-11-12-tacimg-rlt-training
SCENE_GROUP=gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm
IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.10.2
MODE="${1:---dry-run}"
METHODS=(rlt)

die() { echo "ERROR: $*" >&2; exit 1; }
stages_for() { echo "1 2"; }
method_key() { local key="${1^^}"; echo "${key//-/_}"; }
config_name() { echo "$CONFIG_GROUP/stage$2"; }
case "$MODE" in
    --help|-h)
        echo "Host launcher: TacImg RLT Stage 1 -> Stage 2."
        echo "Every 100 steps: upload a complete resumable checkpoint to a public HF model repo."
        echo "HF username comes from .env HF_TOKEN; repo name is the experiment name."
        echo "bash $0 --dry-run    Print the plan; no Docker or filesystem writes."
        echo "bash $0 --check      Inspect local image, snapshot and preflight; no training."
        echo "tmux new -s task820-tacimg-rlt"
        echo "bash $0 --run        Run both RLT stages inside that host tmux session."
        echo "Use --run only after reviewing the no-state/sync/environment contract in Readme.md."
        echo "Keep latest two valid resume points and all 100-step checkpoints."
        echo "Failed stages stop the sequence and require inspected recovery."
        exit 0 ;;
    --dry-run|--check|--run) ;;
    *) die "Unknown option: $MODE; use --help" ;;
esac
[[ $# -le 1 ]] || die "Only one mode argument is accepted"
for method in "${METHODS[@]}"; do
    for stage in $(stages_for "$method"); do
        cfg="$(config_name "$method" "$stage")"
        [[ -f "$WORKSPACE/RLinf/examples/embodiment/config/$cfg.yaml" ]] || die "Missing config: $cfg"
    done
done
if [[ "$MODE" == --dry-run ]]; then
    echo "HOST Bash orchestrator; Docker image: $IMAGE"
    echo "Checkpoint retention: latest two validated complete points plus all multiples of 100."
    echo "gradient_checkpointing=disabled; no-state VLA; residual RLT with proprio7; synchronous RLinf; 8 GPUs via YAML."
    for method in "${METHODS[@]}"; do
        for stage in $(stages_for "$method"); do
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
            echo "  HF: <HF_TOKEN owner>/2026-11-12-tacimg-rlt-training/global_step_{$((stage * 200 - 100)),$((stage * 200))}"
        done
    done
    exit 0
fi
[[ "$MODE" != --run || -n "${TMUX:-}" ]] || die "Run --run inside a named HOST tmux session; see --help"
for tool in docker sha256sum flock find sort xargs diff cmp cp timeout pkill git; do
    command -v "$tool" >/dev/null || die "Missing host command: $tool"
done
if [[ "$MODE" == --run ]]; then
    command -v nvidia-smi >/dev/null || die "Missing nvidia-smi"
    command -v tmux >/dev/null || die "Missing tmux"
fi

# The requested image must already be built; never pull or replace it silently.
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')" || die "Build $IMAGE first; no training started"

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

if [[ -f "$CONTROL/image_id" ]]; then
    [[ "$(cat "$CONTROL/image_id")" == "$IMAGE_ID" ]] || die "Image changed since this experiment snapshot"
fi
export TABERO_IMAGE="$IMAGE_ID"
COMPOSE=(docker compose --project-directory "$WORKSPACE" --env-file "$ENV_FILE" -f "$WORKSPACE/docker-compose.yaml")
NETWORK_ENV=(-e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e NO_PROXY
             -e http_proxy -e https_proxy -e all_proxy -e no_proxy)
"${COMPOSE[@]}" config --quiet
snapshot_compose="$CONTROL/docker-compose.yaml"
if [[ -f "$snapshot_compose" ]]; then
    cmp -s "$WORKSPACE/docker-compose.yaml" "$snapshot_compose" || die "Compose changed since snapshot"
else
    cp "$WORKSPACE/docker-compose.yaml" "$snapshot_compose"
fi


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
import hashlib, json, os, re, shlex, shutil, sys, time
from pathlib import Path

IMAGE = "ccr.ccs.tencentyun.com/vtla/vtla:0.10.2"


EXPERIMENT = "2026-11-12-tacimg-rlt-training"


CONFIG_GROUP = "2026-11-12-tacimg-rlt-training"


SCENE_GROUP = "gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm"


METHODS = ("rlt",)

ENCODER_URL = "https://huggingface.co/Demomasterlqx/VTLA-RL-sft-task820-firm-mixed-tacimg-nostate-rlt-stage1"
ENCODER_REVISION = "d8e990c80f4a975e5cadfc4c276d9b23721f7faf"
ENCODER_SHA256 = "4c0a5a3007d580eafce5228f611b6351ec97335172b5389ee893ad612383a1a8"
LOSS_TAGS = ("train/sac/actor_loss", "train/sac/critic_loss",
             "train/actor/bc_loss", "train/actor/weighted_q", "train/actor/weighted_bc")


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


def hf_api():
    import requests
    from huggingface_hub import HfApi, configure_http_backend

    require(os.environ.get("HF_TOKEN"), "HF_TOKEN is missing from the mounted .env")

    class BoundedSession(requests.Session):
        def request(self, *args, **kwargs):
            kwargs.setdefault("timeout", (10, 120))
            return super().request(*args, **kwargs)

    configure_http_backend(backend_factory=BoundedSession)
    return HfApi(endpoint="https://huggingface.co", token=os.environ["HF_TOKEN"])


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


def stages_for(method):
    require(method in METHODS, f"Unknown variant: {method}")
    return (1, 2)


def config_names(root):
    result = {}
    for method in METHODS:
        for stage in stages_for(method):
            name = f"{CONFIG_GROUP}/stage{stage}"
            require((root / f"{name}.yaml").is_file(), f"Missing config: {name}")
            result[f"{method}:{stage}"] = name
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


def validate_rlt_contract(cfg):
    """Check the reviewed experiment contract without loading policy tensors."""
    from omegaconf import OmegaConf

    expected = {
        "algorithm.loss_type": "rlt_ac",
        "algorithm.adv_type": "embodied_sac",
        "algorithm.rlt_route.actor_scope": "full_task",
        "algorithm.rlt_schedule.enable": True,
        "algorithm.rlt_schedule.transition_replay": True,
        "algorithm.rlt_schedule.max_updates_per_train_step": 500,
        "algorithm.rlt_schedule.warmup_min_size": 256,
        "algorithm.rlt_schedule.warmup_post_collect_updates": 2000,
        "algorithm.rlt_schedule.train_every_transitions": 32,
        "algorithm.rlt_schedule.train_every_episodes": 0,
        "algorithm.gamma": 0.995, "algorithm.tau": 0.005,
        "algorithm.agg_q": "min", "algorithm.actor_agg_q": "q1",
        "algorithm.q_weight": 0.45, "algorithm.bc_weight": 2.5,
        "algorithm.reference_dropout_prob": 0.0,
        "algorithm.actor_weight_schedule.enable": True,
        "algorithm.actor_weight_schedule.warmup_updates": 2000,
        "algorithm.actor_weight_schedule.ramp_updates": 5000,
        "algorithm.actor_weight_schedule.warmup_bc_weight": 7.0,
        "algorithm.actor_weight_schedule.warmup_q_weight": 0.05,
        "algorithm.actor_weight_schedule.online_bc_weight": 2.5,
        "algorithm.actor_weight_schedule.online_q_weight": 0.45,
        "algorithm.entropy_tuning.alpha_type": "fixed_alpha",
        "algorithm.entropy_tuning.initial_alpha": 0.0,
        "algorithm.critic_actor_ratio": 4, "algorithm.train_actor_steps": 1000,
        "algorithm.replay_buffer.enable_cache": True,
        "algorithm.replay_buffer.cache_size": 10000,
        "algorithm.replay_buffer.sample_window_size": 100000,
        "algorithm.replay_buffer.max_num_samples": 100000,
        "algorithm.replay_buffer.min_buffer_size": 1,
        "algorithm.replay_buffer.auto_save": True,
        "algorithm.replay_buffer.enable_preload": False,
        "rollout.collect_transitions": True, "rollout.collect_prev_infos": True,
        "actor.model.model_type": "rlt_mlp_policy",
        "actor.model.action_parameterization": "residual",
        "actor.model.normalized_action_bound": 4.0,
        "actor.model.fixed_std": 0.05,
        "actor.model.precision": "fp32",
        "actor.model.action_dim": 13, "actor.model.num_action_chunks": 10,
        "actor.model.ref_num_action_chunks": 10,
        "actor.model.z_dim": 2048, "actor.model.proprio_dim": 7,
        "actor.model.gradient_checkpointing": False,
        "actor.fsdp_config.gradient_checkpointing": False,
        "actor.fsdp_config.sharding_strategy": "no_shard",
        "actor.fsdp_config.use_orig_params": False,
        "actor.fsdp_config.checkpoint_format": "dcp",
        "actor.fsdp_config.save_trainable_model_weights": True,
        "actor.optim.lr": 1e-4, "actor.critic_optim.lr": 3e-4,
        "actor.optim.clip_grad": 3.5, "actor.critic_optim.clip_grad": 10.0,
        "rollout.rlt_feature_model.model_type": "openpi",
        "rollout.rlt_feature_model.gradient_checkpointing": False,
        "rollout.rlt_feature_model.openpi.config_name": "pi05_lora_tacimg_realworld_replayed_task820_force",
        "rollout.rlt_feature_model.openpi.discrete_state_input": False,
        "rollout.rlt_feature_model.openpi.num_images_in_input": 3,
        "rollout.rlt_feature_model.openpi.tactile_streams": [],
        "rollout.rlt_feature_model.openpi.use_rlt": True,
        "rollout.rlt_feature_model.openpi.rlt_train_module_only": False,
        "rollout.rlt_feature_model.openpi.rlt_stage2_encoder_only": True,
        "rollout.rlt_feature_model.openpi.rlt_action_space": "model_normalized",
        "rollout.rlt_feature_model.openpi.rlt_use_normalized_proprio": True,
        "rollout.rlt_feature_model.openpi.rlt_input_dim": 2048,
        "rollout.rlt_feature_model.openpi.rlt_embed_dim": 2048,
        "rollout.rlt_feature_model.openpi.rlt_num_rl_tokens": 1,
        "rollout.rlt_feature_model.openpi.rlt_num_layers": 2,
        "rollout.rlt_feature_model.openpi.rlt_num_heads": 8,
        "rollout.rlt_feature_model.openpi.rlt_mlp_ratio": 4.0,
        "rollout.rlt_feature_model.openpi.rlt_prefix_seq_len": 1024,
        "rollout.rlt_feature_model.openpi.rlt_image_only": False,
        "rollout.rlt_feature_model.openpi.rlt_use_mask": True,
        "rollout.rlt_feature_model.openpi.action_horizon": 50,
        "rollout.rlt_feature_model.openpi.action_chunk": 10,
        "rollout.rlt_feature_model.openpi.num_steps": 10,
    }
    for key, value in expected.items():
        require(OmegaConf.select(cfg, key) == value, f"RLT contract changed: {key}={value!r} required")
    for key in ("tactile_prefix_dim_in", "tactile_prefix_history", "tactile_prefix_encoder_type",
                "tactile_prefix_use_reference_frame", "tactile_prefix_diff_from_reference"):
        require(cfg.rollout.rlt_feature_model.openpi[key] is None, "TacField prefix must be disabled")
    require("backend" not in cfg.algorithm.replay_buffer, "RLT requires trajectory replay, not compact DSRL replay")
    for key, value in cfg.rollout.model.items():
        if key != "model_path":
            require(value == cfg.actor.model[key], f"Actor/rollout mismatch: {key}")
    for optim in (cfg.actor.optim, cfg.actor.critic_optim):
        require((optim.adam_beta1, optim.adam_beta2, optim.adam_eps, optim.weight_decay,
                 optim.lr_scheduler) == (0.9, 0.999, 1e-8, 0.0, "torch_constant"),
                "RLT filtered Adam defaults changed")
    require(dict(cfg.actor.fsdp_config.trainable_checkpoint_metadata.rlt_action_contract) == {
        "action_parameterization": "residual",
        "action_composition": "reference_plus_bounded_delta", "residual_action_bound": 4.0,
    }, "Trainable export must declare residual semantics")
    return len(expected)


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
    # Configs and the host launcher must also be included in the requested image.
    for stage in (1, 2):
        relative = Path(CONFIG_GROUP) / f"stage{stage}.yaml"
        require(digest(CONTAINER_ROOT / "RLinf/examples/embodiment/config" / relative)
                == digest(CONTAINER_CONTROL / "config" / relative), "Image/local YAML mismatch")
    require(digest(CONTAINER_ROOT / "experiments" / EXPERIMENT / "run_stage12.sh")
            == digest(CONTAINER_CONTROL / "run_stage12.sh"), "Image/local launcher mismatch")
    require(os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is missing from the mounted .env")
    require(shutil.disk_usage("/tmp").free >= 2 * 2**30, "Container /tmp needs at least 2 GiB free")
    require(shutil.disk_usage(CONTAINER_CONTROL).free >= 256 * 2**30,
            "Results filesystem needs at least 256 GiB free for full RLT checkpoints")
    from rlinf.utils.multi_task import task_env_cfg
    from rlinf.utils.rlt_multi_task import validate_config
    from rlinf.envs.isaaclab.tasks.tabero_force_reward import validate_force_reward_cfg
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    base = CONTAINER_ROOT / "models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000"
    encoder = CONTAINER_ROOT / "models/VTLA-RL-sft-task820-firm-mixed-tacimg-nostate-rlt-stage1/step_10000"
    base_config = read_json(base / "config.json")
    export_meta = read_json(base / "export_meta.json")
    sft_contract = read_json(base / "training_contract.json")
    manifest = read_json(encoder / "manifest.json")
    require(base_config["discrete_state_input"] is False and sft_contract["no_state"] is True,
            "Base model must be no-state")
    require(base_config["num_images_in_input"] == 3 and base_config["tactile_streams"] == []
            and sft_contract["tactile_input"] == "tactile_image", "Base must be TacImg")
    require(sft_contract["future_force_loss_weight"] == 0.05, "Wrong SFT force-loss lineage")
    require(export_meta["global_step"] == 30000 and export_meta["is_final"] is True,
            "Expected final SFT step30000 export")
    base_hash = digest(base / "model.safetensors")
    require(base_hash == export_meta["model_sha256"] == manifest["base_model"]["model_sha256"]
            == "6e549631ace7d4ddad23ffefbb0e6b7d1b8ee41ba43beecf93a22b4be9f9fe0c",
            "Base/encoder lineage mismatch")
    norm = base / "assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json"
    norm_hash = digest(norm)
    norm_export_hash = hashlib.sha256(json.dumps(read_json(norm), indent=2).encode()).hexdigest()
    require(export_meta["norm_stats_sha256"] in (norm_hash, norm_export_hash),
            "Base normalization values differ from export provenance")
    encoder_norm = encoder / manifest["normalization_file"]
    require(read_json(encoder_norm) == read_json(norm)
            and digest(encoder_norm) == manifest["normalization_sha256"], "Encoder normalization mismatch")
    require(manifest["global_step"] == 10000 and manifest["is_final"] is True
            and manifest["stage2_actor_present"] is False, "Expected completed Stage1 encoder SFT")
    encoder_hash = digest(encoder / "rlt_encoder.safetensors")
    require(encoder_hash == ENCODER_SHA256, "Encoder differs from the pinned Stage1 export")
    contracts = {}
    for method in METHODS:
        for stage in stages_for(method):
            cfg = composed_config(method, stage)
            model = cfg.actor.model
            feature = cfg.rollout.rlt_feature_model
            validate_rlt_contract(cfg)
            validate_config(cfg)
            name = EXPERIMENT
            log_root = CONTAINER_ROOT / "results" / EXPERIMENT / method
            require(cfg.runner.logger.experiment_name == name, "Unexpected run name")
            require(cfg.runner.logger.log_path == str(log_root), "Unexpected log directory")
            require(set(cfg.runner.logger.logger_backends) == {"tensorboard", "wandb"}, "Both loggers required")
            require(cfg.runner.max_steps == cfg.runner.max_epochs == stage * 200, "Wrong stage target")
            resume = None if stage == 1 else str(log_root / name / "checkpoints/global_step_200")
            require(cfg.runner.resume_dir == resume, "Unexpected Stage1 checkpoint source")
            require(dict(cfg.cluster.component_placement) == {"env": "0-5", "rollout": "0-7", "actor": "0-7"}, "GPU placement changed")
            require(cfg.cluster.num_nodes == 1, "This launcher is single-node")
            require(feature.model_path == str(base)
                    and feature.rlt_encoder_path == str(encoder / "rlt_encoder.safetensors"), "Model paths changed")
            require(feature.openpi_data.norm_stats_path == str(norm), "Normalization path changed")
            for key in ("rlt_input_dim", "rlt_embed_dim", "rlt_num_rl_tokens", "rlt_num_layers",
                        "rlt_num_heads", "rlt_mlp_ratio", "rlt_prefix_seq_len"):
                require(feature.openpi[key] == manifest["metadata"][key], f"Encoder architecture mismatch: {key}")
            require(feature.openpi.rlt_image_only == manifest["rlt_image_only"]
                    and feature.openpi.rlt_use_mask == manifest["rlt_use_mask"], "Encoder prefix mask mismatch")
            train = get_openpi_config(feature.openpi.config_name, model_path=feature.model_path,
                                      data_kwargs=feature.openpi_data)
            for key, value in feature.openpi.items():
                object.__setattr__(train.model, key, value)
            require(train.data.create(train.assets_dirs, train.model).norm_stats is not None,
                    "Normalization statistics did not load")
            require(cfg.actor.micro_batch_size == 48 and cfg.actor.global_batch_size == 384,
                    "Expected micro/global batch 48/384")
            require(cfg.runner.save_interval == 10 and cfg.runner.val_check_interval == -1,
                    "Expected checkpoint interval 10 and no periodic evaluation")
            require(cfg.actor.seed == 42, "Seed changed")
            for split in ("train", "eval"):
                env = cfg.env[split]
                scene = Path(env.init_params.realworld_config_dir) / "gentle_grasp.json"
                require(scene.is_file(), f"Missing scene: {scene}")
                require(env.total_num_envs == (192 if split == "train" else 96)
                        and env.max_steps_per_rollout_epoch == env.max_episode_steps == 300,
                        "Reference environment budget changed")
                for index, task in enumerate(env.multi_task.tasks):
                    task_cfg = task_env_cfg(env, index)
                    reward = validate_force_reward_cfg(task_cfg.init_params.success, 1.0)
                    require(reward["enabled"] == (stage == 2), "Unexpected force reward phase")
                    effort = task_cfg.init_params.success.normalize_effort_reward
                    require((effort.min_effort, effort.max_effort) == [(2, 36), (4, 12), (3, 30)][index]
                            and effort.coefficient == 10, "Reference effort thresholds changed")
                require(Path(env.init_params.realworld_assets_dir).is_dir(), "Missing USD assets in image")
            contracts[f"{method}:{stage}"] = OmegaConf.to_container(cfg, resolve=True)
    owner = hf_api().whoami()["name"]
    owner_path = CONTAINER_CONTROL / "hf_owner"
    if owner_path.exists():
        require(owner_path.read_text().strip() == owner, "HF_TOKEN owner changed since the snapshot")
    else:
        owner_path.write_text(owner + "\n")
    settings = ["HF_OWNER=" + shlex.quote(owner)]
    for method in METHODS:
        logger = contracts[f"{method}:1"]["runner"]["logger"]
        for field in ("experiment_name", "log_path", "project_name", "wandb_entity"):
            settings.append(f"{field.upper()}_{method.upper()}=" + shlex.quote(str(logger[field])))
    (CONTAINER_CONTROL / "settings.env").write_text("\n".join(settings) + "\n")
    write_json(CONTAINER_CONTROL / "contracts.json", contracts)
    write_json(CONTAINER_CONTROL / "base_provenance.json", {
        "source": "https://huggingface.co/Demomasterlqx/" + base.parent.name,
        "model_path": str(base), "export_meta": export_meta, "training_contract": sft_contract,
        "norm_stats_file_sha256": norm_hash, "norm_stats_export_json_sha256": norm_export_hash,
        "encoder_path": str(encoder), "encoder_source": ENCODER_URL,
        "encoder_revision": ENCODER_REVISION, "encoder_sha256": encoder_hash,
        "encoder_manifest": manifest,
    })
    print(f"PREFLIGHT_PASS: {len(expected)} source files, two configs; residual; gradient_checkpointing=disabled")
    print(f"HF_UPLOAD_PLAN: public model repositories under {owner}; complete checkpoints every 100 steps")


def checkpoint_check(method, stage, step=None, resume=False):
    """Check artifact structure and steps; Stage 2 performs the real distributed restore."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    cfg = composed_config(method, stage)
    if resume:
        require(stage == 2 and step is None and cfg.runner.resume_dir,
                "Only Stage 2 has a resume checkpoint at step 200")
        step = 200
        checkpoint = Path(cfg.runner.resume_dir)
    else:
        step = stage * 200 if step is None else step
        require(step % 10 == 0 and (stage - 1) * 200 < step <= stage * 200,
                "Checkpoint step must be a 10-step boundary within this stage")
        checkpoint = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name / f"checkpoints/global_step_{step}"
    require(not checkpoint.is_symlink(), "Checkpoint directory cannot be a symlink")
    for path in checkpoint.rglob("*"):
        require(not path.is_symlink(), f"Checkpoint cannot contain symlinks: {path}")
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
        require(metadata.storage_data, "DCP storage index is empty")
        for item in metadata.storage_data.values():
            relative = Path(item.relative_path)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "DCP storage path escapes checkpoint")
            nonempty(directory / item.relative_path)
            require((directory / item.relative_path).stat().st_size >= item.offset + item.length,
                    f"Truncated DCP shard: {item.relative_path}")

    export = state(actor / "model_state_dict/trainable_weights.pt")
    from rlinf.utils.rlt_action_contract import validate_action_contract
    require(export.get("model"), "Trainable sidecar has no model tensors")
    metadata = export["metadata"]
    require(metadata["step"] == metadata["global_step"] == step,
            "Trainable export step mismatch")
    require(metadata["rank"] == 0 and metadata["world_size"] == 8,
            "RLT actor/world-size mismatch")
    require(metadata["target_global_step"] == (200 if step <= 200 else 400), "Checkpoint stage target mismatch")
    require(metadata["base_model"] == cfg.rollout.rlt_feature_model.model_path
            and metadata["stage1_encoder_bundle"] == str(Path(cfg.rollout.rlt_feature_model.rlt_encoder_path).parent),
            "RLT checkpoint model lineage mismatch")
    validate_action_contract(metadata.get("rlt_action_contract"), 4.0, "residual")
    if step in (200, 400):
        require(metadata.get("is_final") is True, "Terminal checkpoint is not marked final")
    require(any(k.startswith("actor_") for k in export["model"])
            and any(k.startswith("q_head.") for k in export["model"]), "Missing actor/twin-Q weights")
    del export
    task_state = read_json(checkpoint / "multi_task_state.json")
    require(task_state["step"] == step and task_state["format"] == "rlt_multitask_v1",
            "Task controller step/format mismatch")
    require(task_state["names"] == [task.name for task in cfg.env.train.multi_task.tasks],
            "Task controller names mismatch")
    from rlinf.utils.multi_task import SuccessWeightController
    require(isinstance(task_state.get("contract"), dict), "Missing task controller contract")
    SuccessWeightController(task_state["names"], {}, {}, state_format="rlt_multitask_v1").restore(str(checkpoint), step)
    require(cfg.actor.fsdp_config.checkpoint_format == "dcp", "Expected DCP actor checkpoint")
    dcp(actor / "dcp_checkpoint")
    sac = actor / "sac_components"
    # RLT fixed alpha=0 has no optimizer checkpoint. Full target state is rank-0 only.
    require(cfg.algorithm.entropy_tuning.alpha_type == "fixed_alpha"
            and cfg.algorithm.entropy_tuning.initial_alpha == 0, "Expected fixed zero alpha")
    target = state(sac / "target_model/checkpoint_rank_0.pt")
    require(target and any(k.startswith("q_head.") for k in target), "Target Q weights missing")
    require(all(torch.is_tensor(t) and torch.isfinite(t).all() for t in target.values()), "Invalid target weights")
    del target
    consistency = set()
    schedule_fields = ("update_step", "transitions_since_train", "episodes_since_train",
                       "total_transitions_added", "total_episodes_added",
                       "_warmup_ready_total_transitions", "_warmup_ready_total_episodes",
                       "pending_update_budget", "global_step", "rlt_action_contract")
    for rank in range(8):
        sidecar = state(sac / f"multi_task_rank_{rank}.pt")
        require(sidecar["format"] == "rlt_multitask_v1" and sidecar["step"] == step
                and sidecar["rank"] == rank and sidecar["world_size"] == 8,
                "RLT multi-task sidecar step/rank mismatch")
        require(torch.is_tensor(sidecar["critic_rng_state"]) and sidecar["update_step"] > 0,
                "RLT critic RNG/update state missing")
        require("target_shadow_f32" in sidecar, "Missing target precision state")
        require(all(t.dtype == torch.float32 and torch.isfinite(t).all()
                    for t in sidecar["target_shadow_f32"].values()), "Invalid target shadow")
        schedule = state(sac / f"rlt_schedule/checkpoint_rank_{rank}.pt")
        require(all(k in schedule for k in schedule_fields), "Incomplete RLT schedule")
        require(schedule["global_step"] == step and schedule["update_step"] == sidecar["update_step"],
                "RLT schedule/multi-task steps disagree")
        validate_action_contract(schedule["rlt_action_contract"], 4.0, "residual")
        consistency.add(tuple(schedule[k] for k in ("global_step", "update_step",
                        "_warmup_ready_total_transitions", "_warmup_ready_total_episodes", "pending_update_budget")))
        replay_dir = sac / f"replay_buffer/rank_{rank}"
        replay = read_json(replay_dir / "metadata.json")
        index = read_json(replay_dir / "trajectory_index.json")
        require(replay["multi_task_format"] == "sac_multitask_replay_v1"
                and replay["task_names"] == task_state["names"] and replay["rng_state"],
                "RLT trajectory replay format/tasks/RNG missing")
        trajectories = index["trajectory_index"]
        ids = index["trajectory_id_list"]
        require(ids and len(ids) == len(set(ids)) and set(map(str, ids)) == set(trajectories),
                "RLT replay index inconsistent or empty")
        require(replay["total_samples"] == sum(info["num_samples"] for info in trajectories.values())
                and replay["total_samples"] > 0, "RLT replay sample count mismatch")
        extension = "pt" if replay["trajectory_format"] == "pt" else "pkl"
        for tid, info in trajectories.items():
            require(info["task_names"] == task_state["names"], "Trajectory task order mismatch")
            filename = f"trajectory_{tid}_{info['model_weights_id']}.{extension}"
            require(Path(filename).name == filename, "Replay path escapes checkpoint")
            nonempty(replay_dir / filename)
    require(len(consistency) == 1, "RLT schedule state differs across ranks")
    print(f"CHECKPOINT_ARTIFACTS_PASS {method} stage{stage}: {checkpoint}")
    return checkpoint


def upload_checkpoint(method, stage, step):
    """Publish the whole resume_dir in one commit after the runner's atomic marker."""
    require(os.environ.get("HF_TOKEN"), "HF_TOKEN is missing from the mounted .env")
    require(step % 100 == 0, "Only 100-step checkpoints are uploaded")
    cfg = composed_config(method, stage)
    owner = (CONTAINER_CONTROL / "hf_owner").read_text().strip()
    repo_id = f"{owner}/{cfg.runner.logger.experiment_name}"
    prefix = f"global_step_{step}"
    checkpoint = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name / "checkpoints" / prefix
    # The runner writes this atomically only AFTER every actor has finished saving.
    require(read_json(checkpoint / "multi_task_state.json")["step"] == step,
            "Checkpoint is not committed by the runner")

    def inventory():
        files = {}
        for path in checkpoint.rglob("*"):
            require(not path.is_symlink(), f"Checkpoint must contain actual files: {path}")
            if path.is_file():
                stat = path.stat()
                files[path.relative_to(checkpoint).as_posix()] = [stat.st_size, stat.st_mtime_ns]
        require(files, "Empty checkpoint")
        return files

    before = inventory()
    fingerprint = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    receipt_path = CONTAINER_CONTROL / "hf" / method / f"{prefix}.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt["repo_id"] == repo_id and receipt["fingerprint"] == fingerprint,
                "Previously uploaded checkpoint or HF destination changed")
        print("HF_UPLOAD_SKIP", receipt["url"], flush=True)
        return

    checkpoint_check(method, stage, step)
    api = hf_api()
    require(api.whoami()["name"] == owner, "HF_TOKEN owner changed before upload")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    require(not api.repo_info(repo_id=repo_id, repo_type="model").private,
            f"Existing HF repository must be public: {repo_id}")
    print(f"HF_UPLOAD_START {repo_id}/{prefix}: {len(before)} files, "
          f"{sum(v[0] for v in before.values())} bytes", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(checkpoint),
        path_in_repo=prefix, commit_message=f"Upload {method} complete checkpoint at step {step}",
    )
    require(inventory() == before, "Checkpoint changed during upload")
    remote = {
        item.path[len(prefix) + 1:]: item.size
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="model", path_in_repo=prefix,
                                       revision=commit.oid, recursive=True)
        if hasattr(item, "size")
    }
    require(remote == {name: values[0] for name, values in before.items()},
            "HF checkpoint file list or sizes differ from the local resume_dir")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/tree/{commit.oid}/{prefix}"
    write_json(receipt_path, {"repo_id": repo_id, "revision": commit.oid, "url": url,
                             "step": step, "method": method, "stage": stage,
                             "files": len(before), "bytes": sum(v[0] for v in before.values()),
                             "fingerprint": fingerprint, "complete_resume_dir": True})
    print("HF_UPLOAD_VERIFIED", url, flush=True)


def prepare_hf():
    api = hf_api()
    owner = (CONTAINER_CONTROL / "hf_owner").read_text().strip()
    require(api.whoami()["name"] == owner, "HF owner changed")
    for method in METHODS:
        repo_id = f"{owner}/{composed_config(method, 1).runner.logger.experiment_name}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        require(not api.repo_info(repo_id=repo_id, repo_type="model").private,
                f"Existing HF repository must be public: {repo_id}")
        print("HF_REPO_READY", repo_id, flush=True)


def checkpoint_inventory(checkpoint):
    require(not checkpoint.is_symlink(), "Checkpoint directory cannot be a symlink")
    files = {}
    for path in checkpoint.rglob("*"):
        require(not path.is_symlink(), f"Checkpoint cannot contain symlinks: {path}")
        if path.is_file():
            stat = path.stat()
            files[path.relative_to(checkpoint).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def prune_checkpoints(method, stage, cache=None):
    """Delete only validated old directories in this run; preserve incomplete saves."""
    cfg = composed_config(method, stage)
    directory = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name / "checkpoints"
    cache = {} if cache is None else cache
    valid = []
    for path in directory.glob("global_step_*"):
        match = re.fullmatch(r"global_step_([1-9][0-9]*)", path.name)
        if not match or path.is_symlink() or not path.is_dir():
            continue
        step = int(match[1])
        if step > stage * 200 or step % 10 or not (path / "multi_task_state.json").is_file():
            continue
        try:
            inventory = checkpoint_inventory(path)
            if cache.get(step) != inventory:
                checkpoint_check(method, 1 if step <= 200 else 2, step)
                require(checkpoint_inventory(path) == inventory, "Checkpoint changed during validation")
                cache[step] = inventory
            valid.append((step, path, inventory))
        except Exception as error:
            cache.pop(step, None)
            print(f"RETENTION_PRESERVE_INVALID {path}: {error}", flush=True)
    valid.sort(key=lambda item: item[0], reverse=True)
    protected = {step for step, _, _ in valid[:2]}
    removed = []
    for step, path, inventory in valid:
        if step in protected or step % 100 == 0:
            continue
        # Recheck all protection points immediately before deleting anything.
        for _, keep, signature in valid[:2]:
            require(checkpoint_inventory(keep) == signature, "Protected checkpoint changed; stop pruning")
        require(path.parent == directory and checkpoint_inventory(path) == inventory,
                "Checkpoint changed before pruning")
        shutil.rmtree(path)
        cache.pop(step, None)
        removed.append(step)
        print("RETENTION_DELETED", path, flush=True)
    return removed


def progress(method, stage, final=False):
    import math
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = composed_config(method, stage)
    directory = Path(cfg.runner.logger.log_path) / "tensorboard"
    if not directory.is_dir():
        require(not final, "Missing TensorBoard logs")
        return None
    events = EventAccumulator(str(directory), size_guidance={"scalars": 32}).Reload()
    tags = events.Tags().get("scalars", [])
    latest = None
    loss_steps = {}
    for tag in tags:
        if not tag.startswith("train/"):
            continue
        values = events.Scalars(tag)
        require(all(math.isfinite(event.value) for event in values), f"Non-finite metric: {tag}")
        if tag == "train/rlt/update_step" and values:
            latest = values[-1].step
        if tag in LOSS_TAGS and values:
            loss_steps[tag] = values[-1].step
    if final:
        require(latest == stage * 200 - 1, f"TensorBoard last training step is {latest}, expected {stage * 200 - 1}")
        require(len(loss_steps) == len(LOSS_TAGS) and all(step == latest for step in loss_steps.values()),
                "TensorBoard missing final RLT actor/critic/BC/Q metrics")
    return latest


def maintain(method, stage):
    cache = {}
    previous = None
    changed_at = time.monotonic()
    while True:
        latest = progress(method, stage)
        if latest != previous:
            previous, changed_at = latest, time.monotonic()
        require(time.monotonic() - changed_at < 1800, "No fresh RLT training step for 30 minutes")
        require(shutil.disk_usage("/tmp").free >= 2**30, "Container /tmp free space below 1 GiB")
        require(shutil.disk_usage(CONTAINER_CONTROL).free >= 64 * 2**30,
                "Results free space below 64 GiB; stop before exhausting checkpoint storage")
        print(f"MONITOR {time.strftime('%Y-%m-%dT%H:%M:%S')} {method} stage{stage} "
              f"last_logged_step={latest}", flush=True)
        prune_checkpoints(method, stage, cache)
        time.sleep(15)


def verify_logging(method, stage, run_id):
    import math
    import wandb

    cfg = composed_config(method, stage)
    logger = cfg.runner.logger
    target = stage * 200 - 1  # Runner logs zero-based rollout rounds.
    progress(method, stage, final=True)
    for attempt in range(6):
        try:
            run = wandb.Api(timeout=30).run(f"{logger.wandb_entity}/{logger.project_name}/{run_id}")
            require(run.name == logger.experiment_name and run.id == run_id, "W&B identity mismatch")
            loss_tags = list(LOSS_TAGS)
            rows = list(run.scan_history(keys=["_step", *loss_tags],
                                         min_step=target, max_step=target + 1))
            require(any(row["_step"] == target and all(math.isfinite(row[tag]) for tag in loss_tags)
                        for row in rows), f"W&B missing finite training metric at step {target}")
            print("LOGGING_VERIFIED", run.url, f"TensorBoard/W&B last step={target}", flush=True)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(5)


if __name__ == "__main__":
    os.environ["ROOT"] = str(CONTAINER_ROOT)
    if sys.argv[1] == "preflight":
        preflight()
    elif sys.argv[1] == "prepare-hf":
        prepare_hf()
    elif sys.argv[1] == "maintain":
        maintain(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "prune":
        prune_checkpoints(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "logging":
        verify_logging(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "upload":
        upload_checkpoint(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "resume":
        checkpoint_check(sys.argv[2], int(sys.argv[3]), resume=True)
    else:
        checkpoint_check(sys.argv[2], int(sys.argv[3]))
PY
container_check() {
    "${COMPOSE[@]}" run --pull never --rm -T "${NETWORK_ENV[@]}" rlinf rlinf -lc 'exec "$ROOT/RLinf/.venv/bin/python" "$@"' bash "$CONTAINER_CONTROL/check_stage12.py" "$@"
}
container_check preflight 2>&1 | tee "$CONTROL/preflight.log"
echo "$IMAGE_ID" >"$CONTROL/image_id"
[[ "$MODE" == --run ]] || exit 0
# This contains only shell-quoted experiment names/paths and W&B entity/project.
# shellcheck disable=SC1091
source "$CONTROL/settings.env"
# Source provenance is captured independently for each nested repository.
if [[ ! -f "$CONTROL/git-refs.txt" ]]; then
    for repo in . RLinf T2-VLA Tabero_X IsaacLab; do
        echo "$repo $(git -C "$WORKSPACE/$repo" rev-parse HEAD)"
        git -C "$WORKSPACE/$repo" status --short
    done >"$CONTROL/git-refs.txt"
fi

upload_ready_checkpoints() {
    local method="$1" stage="$2" require_all="${3:-false}" key name step attempt rc
    key="EXPERIMENT_NAME_$(method_key "$method")"; name="${!key}"
    for ((step = stage * 200 - 100; step <= stage * 200; step += 100)); do
        if [[ ! -f "$TABERO_RESULTS_HOST/$EXPERIMENT/$method/$name/checkpoints/global_step_$step/multi_task_state.json" ]]; then
            [[ "$require_all" != true ]] || { echo "HF upload missing completed checkpoint: $method step $step" >&2; return 1; }
            continue
        fi
        # A successful, remotely verified upload is immutable and needs no retry.
        [[ "$require_all" == true || ! -f "$CONTROL/hf/$method/global_step_$step.json" ]] || continue
        for attempt in 1 2 3; do
            rc=0
            "${COMPOSE[@]}" run --pull never --rm -T --name "$UPLOAD_CONTAINER" "${NETWORK_ENV[@]}" rlinf rlinf -lc \
                'exec "$ROOT/RLinf/.venv/bin/python" "$@"' bash \
                "$CONTAINER_CONTROL/check_stage12.py" upload "$method" "$stage" "$step" || rc=$?
            [[ "$rc" != 0 ]] || break
            echo "HF upload failed: $method step $step; attempt $attempt/3; exit=$rc" >&2
            [[ "$attempt" == 3 ]] || sleep "$((attempt * 10))"
        done
        [[ "$rc" == 0 ]] || return "$rc"
    done
}

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
    REPORT="$TABERO_RECORD_HOST/experiments/$(date +%Y-%m-%d-%H-%M-%S)-task820-no_adverb-rlt-tacimg_stage12.md"
    echo "$REPORT" >"$CONTROL/record_path"
    {
        echo "# $EXPERIMENT"
        echo
        echo "Started: $(date -Is)"
        echo "Image: $IMAGE ($IMAGE_ID)"
        echo "Host tmux: $(tmux display-message -p '#S:#I.#P')"
        echo "Order: TacImg RLT Stage 1 -> Stage 2."
        echo "No-state TacImg VLA; RLT actor/critic retain 7-D EEF/gripper proprio; residual actions; synchronous RLinf; XY +/-1 cm."
        echo "Stage 1: 0-200, success reward. Stage 2: 200-400, adaptive mass and normalized force reward."
        echo "Gradient checkpointing: disabled. Logging: TensorBoard + W&B."
        echo "Full parameters, model paths, and copied scenes: $CONTROL/contracts.json, $CONTROL/config, $CONTROL/env"
        echo "Base/encoder hashes, HF revision and normalization provenance: $CONTROL/base_provenance.json"
        echo "Frozen encoder: https://huggingface.co/Demomasterlqx/VTLA-RL-sft-task820-firm-mixed-tacimg-nostate-rlt-stage1"
        echo "Launcher/source snapshots and stage logs: $CONTROL"
        echo "Repository refs/status: $CONTROL/git-refs.txt"
        echo "Checkpoint retention: latest two complete points plus every multiple of 100; incomplete points are preserved."
        echo "No downstream simulator evaluation is included in this launcher."
    } >"$REPORT"
fi
for method in "${METHODS[@]}"; do
    key="EXPERIMENT_NAME_$(method_key "$method")"; name="${!key}"
    key="LOG_PATH_$(method_key "$method")"; log_path="${!key}"
    key="WANDB_ENTITY_$(method_key "$method")"; entity="${!key}"
    key="PROJECT_NAME_$(method_key "$method")"; project="${!key}"
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
            echo "W&B (one ID for RLT across both stages; planned until training initializes): $url"
            echo "HF (public, full resume_dir every 100 steps): https://huggingface.co/$HF_OWNER/$name"
            echo "Results: $log_path"
        } >>"$REPORT"
    fi
done

MAINT_PID=
MAINT_CONTAINER=
ACTIVE_CONTAINER=
ACTIVE_STAGE_DIR=
MONITOR_PID=
UPLOAD_PID=
UPLOAD_CONTAINER=
START_SECONDS=$SECONDS
stop_monitor() {
    [[ -n "$MONITOR_PID" ]] || return 0
    pkill -TERM -P "$MONITOR_PID" 2>/dev/null || true
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
}
stop_upload() {
    if [[ -n "$UPLOAD_CONTAINER" ]]; then
        docker stop --time 15 "$UPLOAD_CONTAINER" >/dev/null 2>&1 || true
    fi
    if [[ -n "$UPLOAD_PID" ]]; then
        pkill -TERM -P "$UPLOAD_PID" 2>/dev/null || true
        kill "$UPLOAD_PID" 2>/dev/null || true
        wait "$UPLOAD_PID" 2>/dev/null || true
    fi
    UPLOAD_PID=
    UPLOAD_CONTAINER=
}
stop_maintenance() {
    if [[ -n "$MAINT_CONTAINER" ]]; then
        docker stop --time 30 "$MAINT_CONTAINER" >/dev/null 2>&1 || true
    fi
    if [[ -n "$MAINT_PID" ]]; then
        wait "$MAINT_PID" 2>/dev/null || true
    fi
    MAINT_PID=
    MAINT_CONTAINER=
}
finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        docker stop --time 60 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
    stop_monitor
    stop_maintenance
    stop_upload
    if [[ -n "$ACTIVE_STAGE_DIR" ]]; then
        echo failed >"$ACTIVE_STAGE_DIR/status"
        echo "$rc" >"$ACTIVE_STAGE_DIR/exit_code"
        date -Is >"$ACTIVE_STAGE_DIR/ended_at"
        if [[ -n "${stage_start:-}" ]]; then
            echo "$((SECONDS - stage_start))" >"$ACTIVE_STAGE_DIR/duration_seconds"
        fi
        echo "Interrupted/failed: $ACTIVE_STAGE_DIR; exit=$rc; $(date -Is)" >>"$REPORT"
    fi
    echo "Launcher invocation ended: $(date -Is); duration=$((SECONDS - START_SECONDS)) s; exit=$rc" >>"$REPORT"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
container_check prepare-hf 2>&1 | tee "$CONTROL/hf-prepare.log"

for method in "${METHODS[@]}"; do
    run_id="$(cat "$CONTROL/$method.wandb_id")"
    key="EXPERIMENT_NAME_$(method_key "$method")"; name="${!key}"
    key="LOG_PATH_$(method_key "$method")"; log_path="${!key}"
    key="WANDB_ENTITY_$(method_key "$method")"; entity="${!key}"
    key="PROJECT_NAME_$(method_key "$method")"; project="${!key}"
    for stage in $(stages_for "$method"); do
        stage_dir="$CONTROL/$method-stage$stage"
        UPLOAD_CONTAINER="tacimg-rlt-hf-$method-stage$stage-$run_id"
        if [[ -f "$stage_dir/status" ]]; then
            [[ "$(cat "$stage_dir/status")" == complete ]] || die "$method Stage $stage was interrupted/failed; inspect its logs and review recovery before restarting"
            container_check checkpoint "$method" "$stage"
            upload_ready_checkpoints "$method" "$stage" true 2>&1 | tee -a "$stage_dir/hf.log"
            UPLOAD_CONTAINER=
            echo "SKIP verified completed stage: $method $stage"
            continue
        fi
        [[ "$stage" != 2 ]] || container_check resume "$method" "$stage"
        idle_gpus
        mkdir -p "$stage_dir"
        ACTIVE_STAGE_DIR="$stage_dir"
        ACTIVE_CONTAINER="tacimg-rlt-$method-stage$stage-$run_id"
        cfg="$(config_name "$method" "$stage")"
        wandb_resume=never
        [[ "$stage" != 2 ]] || wandb_resume=must
        COMMAND=("${COMPOSE[@]}" run --pull never --rm -T --name "$ACTIVE_CONTAINER"
            "${NETWORK_ENV[@]}" -e "WANDB_RUN_ID=$run_id" -e "WANDB_RESUME=$wandb_resume" -e WANDB_MODE=online
            rlinf rlinf -lc '
set -euo pipefail
# Set identity AFTER entrypoint reads .env; use the host-persisted ID.
export WANDB_RUN_ID="$1" WANDB_RESUME="$2" WANDB_MODE=online
export WANDB_PROJECT="$7" WANDB_ENTITY="$8" PYTHONUNBUFFERED=1
export WANDB_NAME="$9"
unset RAY_ADDRESS
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Use YAML placement, not CUDA_VISIBLE_DEVICES" >&2; exit 1; }
cd "$ROOT/RLinf"
echo "gradient_checkpointing=disabled; action_parameterization=residual; W&B run=$WANDB_RUN_ID; resume=$WANDB_RESUME"
"$ROOT/RLinf/.venv/bin/python" examples/embodiment/train_embodied_agent.py --config-path "$3/config" --config-name "$4" "env.train.init_params.realworld_config_dir=$3/env/$5/stage$6" "env.eval.init_params.realworld_config_dir=$3/env/$5/stage$6"
' bash "$run_id" "$wandb_resume" "$CONTAINER_CONTROL" "$cfg" "$SCENE_GROUP" "$stage" "$project" "$entity" "$name")
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
        MAINT_CONTAINER="tacimg-rlt-maint-$method-stage$stage-$run_id"
        "${COMPOSE[@]}" run --pull never --rm -T --name "$MAINT_CONTAINER" rlinf rlinf -lc \
            'exec "$ROOT/RLinf/.venv/bin/python" "$@"' bash \
            "$CONTAINER_CONTROL/check_stage12.py" maintain "$method" "$stage" >"$stage_dir/maintenance.log" 2>&1 &
        MAINT_PID=$!
        (
            trap - EXIT
            trap 'jobs -pr | xargs -r kill 2>/dev/null; exit 0' TERM INT
            while kill -0 "$docker_pid" 2>/dev/null; do
                upload_ready_checkpoints "$method" "$stage" || echo "HF upload will retry while training continues"
                sleep 15
            done
            upload_ready_checkpoints "$method" "$stage"
        ) >"$stage_dir/hf.log" 2>&1 &
        UPLOAD_PID=$!
        (
            trap - EXIT
            trap 'jobs -pr | xargs -r kill 2>/dev/null; exit 0' TERM INT
            while kill -0 "$docker_pid" 2>/dev/null; do
                date -Is
                timeout 20 nvidia-smi || true
                df -h "$TABERO_RESULTS_HOST" || true
                if ! kill -0 "$MAINT_PID" 2>/dev/null; then
                    echo "Maintenance/metric monitor exited; stopping owned training container"
                    touch "$stage_dir/supervision_failed"
                    docker stop --time 60 "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
                    exit 1
                fi
                sleep 15
            done
        ) >"$stage_dir/gpu.log" 2>&1 &
        MONITOR_PID=$!
        rc=0
        wait "$docker_pid" || rc=$?
        echo "$((SECONDS - stage_start))" >"$stage_dir/training_duration_seconds"
        stop_monitor
        maintenance_rc=0
        if ! kill -0 "$MAINT_PID" 2>/dev/null; then
            wait "$MAINT_PID" || maintenance_rc=$?
            touch "$stage_dir/supervision_failed"
        fi
        stop_maintenance
        ACTIVE_CONTAINER=
        echo "$rc" >"$stage_dir/docker_exit_code"
        [[ ! -f "$stage_dir/supervision_failed" ]] || die "Metric/retention supervision failed; inspect $stage_dir/maintenance.log (exit=$maintenance_rc)"
        [[ "$rc" == 0 ]] || die "$method Stage $stage failed (exit $rc); see $stage_dir/launcher.log"
        # Let an in-flight upload finish. Retry failures once the stage has stopped;
        # do not mark a stage complete until both 100-step checkpoints are uploaded.
        wait "$UPLOAD_PID" || echo "HF background upload failed; retrying from saved checkpoints" >&2
        UPLOAD_PID=
        upload_ready_checkpoints "$method" "$stage" true 2>&1 | tee -a "$stage_dir/hf.log"
        UPLOAD_CONTAINER=
        container_check checkpoint "$method" "$stage" 2>&1 | tee "$stage_dir/checkpoint.log"
        container_check logging "$method" "$stage" "$run_id" 2>&1 | tee "$stage_dir/wandb.log"
        container_check prune "$method" "$stage" 2>&1 | tee -a "$stage_dir/maintenance.log"
        date -Is >"$stage_dir/ended_at"
        echo "$((SECONDS - stage_start))" >"$stage_dir/duration_seconds"
        echo 0 >"$stage_dir/exit_code"
        echo complete >"$stage_dir/status"
        ACTIVE_STAGE_DIR=
        {
            echo
            echo "### $method Stage $stage complete"
            echo "Started: $(cat "$stage_dir/started_at"); ended: $(cat "$stage_dir/ended_at")"
            echo "Training container duration: $(cat "$stage_dir/training_duration_seconds") s"
            echo "Stage duration including upload/validation: $(cat "$stage_dir/duration_seconds") s; exit=0"
            echo "Logs / exact launch command: $stage_dir"
            echo "Checkpoint: $log_path/$name/checkpoints/global_step_$((stage * 200))"
            echo "TensorBoard: $log_path/tensorboard"
            echo "HF uploads: https://huggingface.co/$HF_OWNER/$name; verified receipts: $CONTROL/hf/$method"
            cat "$stage_dir/wandb.log"
        } >>"$REPORT"
    done
done
total_training_seconds=0
for method in "${METHODS[@]}"; do
    for stage in 1 2; do
        duration="$(cat "$CONTROL/$method-stage$stage/training_duration_seconds")"
        [[ "$duration" =~ ^[0-9]+$ ]] || die "Invalid recorded duration"
        total_training_seconds=$((total_training_seconds + duration))
    done
done
echo "Both RLT stages verified: $(date -Is); total training container duration=$total_training_seconds s" >>"$REPORT"
echo "Both RLT stages completed. Record: $REPORT"
