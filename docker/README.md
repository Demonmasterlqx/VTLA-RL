# VTLA Docker 环境

默认镜像名：

```text
ccr.ccs.tencentyun.com/vtla/vtla:0.4
```

镜像工作区位于 `/root/VTLA-RL`，包含 IsaacLab、T2-VLA、Tabero_X、RLinf，
并通过 `/root/VTLA-RL/isaacsim -> /isaac-sim` 使用 NVIDIA 基础镜像自带的
Isaac Sim。GPU 驱动由宿主机 NVIDIA Container Toolkit 提供，不写入镜像。

Python 环境彼此隔离：

| 组件 | Python 环境 | 运行时初始化 |
| --- | --- | --- |
| Tabero_X / IsaacLab | `/root/VTLA-RL/IsaacLab/.venv` | `docker/env_setup/tabero_x.sh` |
| RLinf | `/root/VTLA-RL/RLinf/.venv` | `docker/env_setup/rlinf.sh` |
| T2-VLA | `/root/VTLA-RL/T2-VLA/.venv` | `docker/env_setup/t2-vla.sh` |

## 凭据

不要把真实 token 写进 Dockerfile、镜像层或 Git。复制示例文件到宿主机安全
位置并限制权限：

```bash
cp docker/.env.example /secure/path/vtla.env
chmod 600 /secure/path/vtla.env
```

填写：

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

Compose 默认将仓库根目录的 `.env` 只读挂载到
`/root/VTLA-RL/.env`。使用其他文件时设置：

```bash
export TABERO_ENV_FILE=/secure/path/vtla.env
```

构建所需的 Hugging Face token 通过 BuildKit secret 提供，不会保存在镜像层。

## 构建

Dockerfile 使用 BuildKit 原生 cache mount 缓存 apt、uv 和 pip 下载。缓存保存在
Docker builder 中，不再把宿主机的 `$HOME/.cache/uv` 和 `$HOME/.cache/pip`
作为几十 GB 的额外构建上下文上传。第一次构建仍需下载依赖，后续构建会复用缓存。

Compose 中只有 `shell` 服务声明 `build`，其余服务复用相同的 `TABERO_IMAGE`，
因此一次 `docker compose build` 只会提交一个镜像构建。

使用宿主机代理构建时，由于构建使用 host network，容器可以访问宿主机的
`127.0.0.1` 代理：

```bash
export HTTP_PROXY=http://127.0.0.1:10080
export HTTPS_PROXY=http://127.0.0.1:10080
export ALL_PROXY=
export PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
export HF_ENDPOINT=https://hf-mirror.com
export TABERO_IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.4

BUILDKIT_PROGRESS=plain docker compose build
```

`PYPI_INDEX_URL` 用于普通 PyPI 包以及 IsaacLab 内部的 pip 安装；
`PYTORCH_INDEX_URL` 用于 CUDA 12.8 的 PyTorch wheel；`HF_ENDPOINT` 用于
Tabero_X 固定版本资产。资产下载单独成层，并用 BuildKit cache mount 保存未完成
下载；构建时禁用会绕过镜像的 Hugging Face Xet/CAS 传输，网络中断后重建不会
重新安装 IsaacLab。PyG wheel和 GitHub 仓库仍使用各自
专用来源。需要切回官方源时分别设置
`PYPI_INDEX_URL=https://pypi.org/simple` 和
`PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`，并设置
`HF_ENDPOINT=https://huggingface.co`。

T2-VLA 的锁文件包含官方 wheel 直链，仅设置 index 不会替换这些地址。
构建会先从国内源安装锁定的 CUDA 12.8 Torch，再导出带哈希的锁定依赖清单
预装普通包，最后执行 `uv sync --frozen`。该过程不改写源码中的 `uv.lock`。

本地验证可使用独立标签，避免覆盖正式标签：

```bash
export TABERO_IMAGE=vtla:env-fix
BUILDKIT_PROGRESS=plain docker compose build
```

查看 BuildKit 缓存占用：

```bash
docker buildx du
```

推送正式镜像：

```bash
docker login ccr.ccs.tencentyun.com
docker push ccr.ccs.tencentyun.com/vtla/vtla:0.4
```

## 启动

四个服务共享同一个镜像，但进入不同的运行环境：

```bash
docker compose run --rm shell
docker compose run --rm tabero_x
docker compose run --rm rlinf
docker compose run --rm t2-vla
```

- `shell`：通用登录 Shell，不自动激活 Python 环境；可手动 source
  `docker/env_setup/` 下的对应脚本。
- `tabero_x`：激活 IsaacLab `.venv`，加载 Isaac Sim 环境变量并进入 Tabero_X。
- `rlinf`：激活 RLinf `.venv`，加载 Isaac Sim、IsaacLab 与 Tabero_X 的运行路径。
- `t2-vla`：激活 T2-VLA `.venv`。

Compose 将宿主机目录挂载到容器：

| 宿主机目录 | 容器目录 | 权限 |
| --- | --- | --- |
| `datasets` | `/root/VTLA-RL/datasets` | 只读 |
| `models` | `/root/VTLA-RL/models` | 只读 |
| `results` | `/root/VTLA-RL/results` | 可写 |
| `Record` | `/root/VTLA-RL/Record` | 可写 |

默认目录来自 `${TABERO_HOST:-.}`。可分别用 `TABERO_DATASETS_HOST`、
`TABERO_MODELS_HOST`、`TABERO_RESULTS_HOST`、`TABERO_RECORD_HOST` 指定完整路径。
共享盘空间不足时，在运行 Compose 的宿主机 Shell 中设置：

```bash
export TABERO_RESULTS_HOST=/path/on/large-disk/results
export TABERO_RECORD_HOST=/path/on/large-disk/Record
mkdir -p "$TABERO_RESULTS_HOST" "$TABERO_RECORD_HOST/experiments"
docker compose config --quiet
docker compose run --rm rlinf rlinf -lc 'df -h "$ROOT/results" "$ROOT/Record"'
```

这些变量只改变挂载位置，不会迁移旧文件。若使用独立的凭据文件，其内容由容器
入口读取；Compose 挂载变量需要在宿主机导出，或写入 Compose 使用的 `.env`。

镜像还提供旧训练配置使用的兼容链接：

```text
/root/VTLA-RL/datas -> datasets
/root/VTLA-RL/Tabero -> Tabero_X
/data/home/sim6g/code/tabero -> /root/VTLA-RL
```

直接使用 `docker run` 时：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  --env-file /secure/path/vtla.env \
  -v /secure/path/vtla.env:/root/VTLA-RL/.env:ro \
  -v "$PWD/datasets:/root/VTLA-RL/datasets:ro" \
  -v "$PWD/models:/root/VTLA-RL/models:ro" \
  -v "$PWD/results:/root/VTLA-RL/results" \
  -v "$PWD/Record:/root/VTLA-RL/Record" \
  ccr.ccs.tencentyun.com/vtla/vtla:0.4 shell
```

## 更新代码

仓库源码在构建时写入镜像。当前镜像不包含自动执行 `git pull` 的入口；代码更新后
应从宿主机工作区重新构建镜像，避免容器内代码与本地工作区产生不可追踪的差异。

## 验证

先检查 Dockerfile 和 Compose：

```bash
docker buildx build --check -f Dockerfile .
docker compose config --quiet
```

镜像构建完成后检查三个解释器和关键依赖：

```bash
docker compose run --rm shell bash -lc '\
  /root/VTLA-RL/IsaacLab/.venv/bin/python -c "import cv2, torch, tyro" && \
  /root/VTLA-RL/RLinf/.venv/bin/python -c "import flash_attn, jax, lerobot, openpi, torch" && \
  /root/VTLA-RL/T2-VLA/.venv/bin/python -c "import openpi, torch"'
```

需要检查 RLinf 的完整 Isaac 运行变量时，进入已自动初始化的服务：

```bash
docker compose run --rm rlinf
python -c 'import importlib.util; print(importlib.util.find_spec("isaaclab"))'
```


## 训练、导出与评测

下面的命令都从仓库根目录执行，并统一使用已构建镜像，不隐式拉取新镜像：

```bash
export TABERO_IMAGE=vtla:env-fix
export TABERO_MODELS_HOST=/absolute/path/to/models
export TABERO_DATASETS_HOST=/absolute/path/to/datasets
export TABERO_RESULTS_HOST=/absolute/path/to/results
export TABERO_RECORD_HOST="$PWD/Record"

docker compose config --quiet
```

本镜像已验证的兼容版本是：

| 环境 | PEFT | Torch |
| --- | --- | --- |
| T2-VLA | `0.20.0` | `2.7.1+cu128` |
| RLinf | `0.21.0` | `2.7.1+cu128` |

不要在现有 `.venv` 中就地升级这些包。PIRL 从 RLinf 导出给 T2-VLA 时，bundle
必须用 T2 兼容的 PEFT 0.20 实现生成；两边版本不同是明确的部署边界，不是待升级项。
只读核对命令：

```bash
docker compose run --pull never --rm -T shell shell -lc '
"$ROOT/T2-VLA/.venv/bin/python" -c "import peft,torch; print(peft.__version__,torch.__version__)"
"$ROOT/RLinf/.venv/bin/python" -c "import peft,torch; print(peft.__version__,torch.__version__)"'
```

所有训练都应放进命名 `tmux` 会话并保存 launcher、日志和退出码。以下训练块应写入
对应运行目录的 `launch.sh` 后由 `tmux new-session -d -s <name> bash <launch.sh>`
启动。RLinf 的 GPU 只能通过 `cluster.component_placement` 分配，不要为 RLinf 设置
`CUDA_VISIBLE_DEVICES`。

### SFT smoke：pi05_base_pytorch + task820 firm mixed

```bash
RUN=task820-no-adverb-sft-smoke
docker compose run --pull never --rm -T \
  -e TASK820_MIXED_TACFIELD_RUN_DIR="/root/VTLA-RL/results/$RUN" \
  -e TASK820_MIXED_TACFIELD_RUN_NAME="$RUN" \
  -e TASK820_MIXED_TACFIELD_CHECKPOINT_ROOT="/root/VTLA-RL/results/$RUN/checkpoints" \
  -e WANDB_RUN_ID="$RUN" -e WANDB_RESUME=allow \
  rlinf rlinf -lc '
python examples/sft/train_vla_sft.py \
  --config-path "$ROOT/RLinf/examples/sft/config" \
  --config-name realworld_replay_task820_firm_mixed_pi05_tacfield_no_state_smoke \
  "cluster.component_placement.actor=\"1,7\"" \
  runner.max_steps=1 runner.save_interval=1 \
  actor.micro_batch_size=1 actor.global_batch_size=2 \
  actor.model.precision=bf16 \
  data.train_data_paths.0.dataset_path="$ROOT/datasets/realworld_replay_task820_firm_mixed" \
  actor.model.model_path="$ROOT/models/pi05_base_pytorch" \
  actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1'
```

把 `runner.max_steps=1`、保存间隔和 smoke config 改成正式值前，需要先审核完整训练
合同；不要直接把 smoke 命令当正式训练。

### PIRL smoke

```bash
RUN=task820-no-adverb-pirl-smoke
docker compose run --pull never --rm -T \
  -e WANDB_RUN_ID="$RUN" -e WANDB_RESUME=allow \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  rlinf rlinf -lc '
python examples/embodiment/train_embodied_agent.py \
  --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_benchmark \
  "cluster.component_placement.actor=\"1,6\"" \
  "cluster.component_placement.rollout=\"7\"" \
  "cluster.component_placement.env=\"7\"" \
  runner.logger.log_path="$ROOT/results/'"$RUN"'" \
  runner.logger.experiment_name='"$RUN"' \
  runner.max_epochs=1 runner.max_steps=1 runner.save_interval=1 \
  runner.resume_dir=null runner.overlap_env_bootstrap=false \
  env.train.total_num_envs=2 env.train.rollout_epoch=1 \
  env.train.max_steps_per_rollout_epoch=300 \
  env.train.max_episode_steps=300 env.train.init_params.max_episode_steps=300 \
  rollout.model.model_path="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  actor.model.model_path="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  actor.micro_batch_size=1 actor.global_batch_size=2 \
  actor.fsdp_config.gradient_checkpointing=false \
  actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1'
```

### DSRL smoke

```bash
RUN=task6-no-adverb-dsrl-smoke
docker compose run --pull never --rm -T \
  -e WANDB_RUN_ID="$RUN" -e WANDB_RESUME=allow \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_SHM_DISABLE=1 -e NCCL_P2P_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo \
  rlinf rlinf -lc '
python examples/embodiment/train_async.py \
  --config-name isaaclab_pi05_dsrl_task820_tacfield_8gpu_smoke \
  "cluster.component_placement.actor=\"3\"" \
  "cluster.component_placement.rollout=\"4\"" \
  "cluster.component_placement.env=\"7\"" \
  runner.logger.log_path="$ROOT/results/'"$RUN"'" \
  runner.logger.experiment_name='"$RUN"' \
  runner.max_epochs=1 runner.max_steps=1 runner.save_interval=1 \
  runner.resume_dir=null runner.overlap_env_bootstrap=false \
  env.train.total_num_envs=1 env.train.rollout_epoch=1 \
  env.train.max_steps_per_rollout_epoch=20 \
  env.train.max_episode_steps=20 env.train.init_params.max_episode_steps=20 \
  rollout.model.model_path="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  actor.model.model_path="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  actor.micro_batch_size=1 actor.global_batch_size=1 \
  actor.model.openpi.discrete_state_input=false \
  +actor.model.openpi.dsrl_actor_use_state=true \
  actor.fsdp_config.gradient_checkpointing=false \
  actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1'
```

### RLT Stage 1 smoke

```bash
RUN=task820-no-adverb-rlt-stage1-smoke
docker compose run --pull never --rm -T \
  -e TASK820_MIXED_TACFIELD_RUN_DIR="/root/VTLA-RL/results/$RUN" \
  -e TASK820_MIXED_TACFIELD_RUN_NAME="$RUN" \
  -e TASK820_MIXED_TACFIELD_CHECKPOINT_ROOT="/root/VTLA-RL/results/$RUN/checkpoints" \
  -e WANDB_RUN_ID="$RUN" -e WANDB_RESUME=allow \
  rlinf rlinf -lc '
python examples/sft/train_vla_sft.py \
  --config-name realworld_replay_task820_firm_mixed_pi05_tacfield_no_state_smoke \
  "cluster.component_placement.actor=\"3,7\"" \
  runner.max_steps=1 runner.save_interval=1 \
  data.train_data_paths.0.dataset_path="$ROOT/datasets/realworld_replay_task820_firm_mixed" \
  actor.micro_batch_size=1 actor.global_batch_size=2 \
  actor.model.model_path="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  actor.model_weight_ema_decay=null actor.model.precision=null \
  actor.model.frozen_parameter_precision=bf16 actor.model.trainable_parameter_precision=bf16 \
  actor.model.is_lora=false actor.model.num_action_chunks=10 \
  actor.model.checkpoint_load_allowed_missing_prefixes=[rlt_module.] \
  actor.model.openpi.action_chunk=10 actor.model.openpi.train_expert_only=true \
  +actor.model.openpi.use_rlt=true +actor.model.openpi.rlt_train_module_only=true \
  +actor.model.openpi.rlt_input_dim=2048 +actor.model.openpi.rlt_embed_dim=2048 \
  +actor.model.openpi.rlt_num_rl_tokens=1 +actor.model.openpi.rlt_prefix_seq_len=1024 \
  +actor.model.openpi.rlt_num_layers=2 +actor.model.openpi.rlt_num_heads=8 \
  actor.optim.total_training_steps=1 actor.optim.lr_warmup_steps=0 \
  actor.optim.lr_decay_steps=1 actor.fsdp_config.gradient_checkpointing=false \
  actor.fsdp_config.save_full_model_weights=true \
  actor.fsdp_config.trainable_checkpoint_metadata.method=rlt_stage1 \
  actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1'
```

### RLT Stage 2

Stage 2 的配置应先保存到宿主机结果目录，例如
`$TABERO_RESULTS_HOST/$RUN/rlt_stage2.yaml`。配置内用
`cluster.component_placement` 指定 actor/env 和 rollout GPU，并指向 Stage 1 的 final
checkpoint；不要用环境变量替代 GPU placement。

```bash
RUN=task6-no-adverb-rlt-stage2-smoke
docker compose run --pull never --rm -T \
  -e WANDB_RUN_ID="$RUN" -e WANDB_RESUME=allow \
  -e NCCL_SHM_DISABLE=1 -e NCCL_P2P_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo \
  rlinf rlinf -lc '
python examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/results/'"$RUN"'" \
  --config-name rlt_stage2'
```

本轮通过的最小配置使用 actor/env GPU 4、rollout GPU 3、20 个 simulator control
steps、两个 10-step transition chunks 和一次 actor/critic 更新。正式 Stage 2 必须从已
审核的配置启动，不能把这组 smoke 上限直接当正式训练参数。

### 导出 PIRL、DSRL 和 RLT bundle

导出在 RLinf 环境执行。`TRAIN_CONFIG` 必须是训练时保存的完整 resolved YAML，不能是
未展开的 Hydra 配置。

PIRL/LoRA：RLinf 训练环境保留 PEFT 0.21，导出时把 PEFT 0.20 安装到运行目录的
overlay，不修改任何 `.venv`。`PYTHONPATH` 只对这次导出命令生效：

```bash
docker compose run --pull never --rm -T rlinf rlinf -lc '
OVERLAY="$ROOT/results/<run>/t2_peft020_export_overlay"
uv pip install --target "$OVERLAY" --no-deps "peft==0.20.0"
PYTHONPATH="$OVERLAY:${PYTHONPATH:-}" python -c \
  "import peft; assert peft.__version__ == '\''0.20.0'\''"
PYTHONPATH="$OVERLAY:${PYTHONPATH:-}" \
python -m rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla \
  --train_config_path "$TRAIN_CONFIG" \
  --ckpt_path "$PIRL_CHECKPOINT" \
  --output_dir "$ROOT/results/<run>/pirl_t2_checkpoint" \
  --bundle_dir "$ROOT/results/<run>/pirl_lora_bundle"'
```

DSRL：

```bash
docker compose run --pull never --rm -T rlinf rlinf -lc '
python -m rlinf.utils.ckpt_convertor.export_tabero_dsrl_for_t2vla \
  --trainable-checkpoint "$DSRL_TRAINABLE_CHECKPOINT" \
  --train-config "$TRAIN_CONFIG" \
  --observation-sample "$OBSERVATION_SAMPLE" \
  --output-dir "$ROOT/results/<run>/dsrl_bundle" \
  --base-model "$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  --expected-base-model-sha256 "$BASE_MODEL_SHA256"'
```

RLT：

```bash
docker compose run --pull never --rm -T rlinf rlinf -lc '
python -m rlinf.utils.ckpt_convertor.export_tabero_rlt_for_t2vla \
  --stage1-checkpoint "$RLT_STAGE1_TRAINABLE_WEIGHTS" \
  --stage2-checkpoint "$RLT_STAGE2_FULL_WEIGHTS" \
  --output-dir "$ROOT/results/<run>/rlt_bundle_t2" \
  --base-model "$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model" \
  --stage2-global-step 1 \
  --base-config-name pi05_lora_tacfield_realworld_replayed_task820_firm_mixed_no_state \
  --base-norm-asset-id pi05_horizon50_tacfield_task820_firm_mixed \
  --base-use-quantile-norm \
  --base-model-sha256 "$BASE_MODEL_SHA256"'
```

这里的 `--base-config-name` 是 T2-VLA **部署配置名**。不要填写 RLinf 的训练配置名
`pi05_lora_tacfield_realworld_replayed_task820`；二者用途不同，即使模型权重相同也会
解析出不同的 state/norm 合同。

### 启动 T2-VLA 服务器

先确认端口未占用，再按 bundle 类型三选一：`--lora-bundle`、`--dsrl-bundle` 或
`--rlt-bundle`。三个参数互斥。以下是 RLT 示例；PIRL/DSRL 只需替换 bundle flag。

```bash
ss -ltn | rg ':10010 ' || true

docker compose run --pull never --rm -T \
  --name vtla-policy-server \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e JAX_PLATFORMS=cuda \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  t2-vla t2-vla -lc '
python -u scripts/serve_policy.py \
  --port 10010 \
  --rlt-bundle "$ROOT/results/<run>/rlt_bundle_t2" \
  policy:checkpoint \
  --policy.config=pi05_lora_tacfield_realworld_replayed_task820_firm_mixed_no_state \
  --policy.dir="$ROOT/models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model"'
```

PIRL 使用 `--lora-bundle "$ROOT/results/<run>/pirl_lora_bundle"`；DSRL 使用
`--dsrl-bundle "$ROOT/results/<run>/dsrl_bundle"`。只有看到
`server listening on 0.0.0.0:<port>` 且进程仍存活后才能启动 Isaac 客户端。
验证普通 SFT/base checkpoint 时省略 bundle flag，保留
`policy:checkpoint --policy.config=... --policy.dir=...` 即可。

### 启动 Tabero_X 客户端并保存视频

默认 reset 可使用运行目录下的显式空 HDF5 目录。若要求 HDF5 reset，则必须先审核
文件内容，不能自动退回默认 reset。

```bash
RUN=task6-policy-eval
mkdir -p "$TABERO_RESULTS_HOST/$RUN/empty_hdf5" \
         "$TABERO_RESULTS_HOST/$RUN/raw/task_6"

docker compose run --pull never --rm -T \
  --name vtla-policy-client \
  -e CUDA_VISIBLE_DEVICES=7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VTLA_RENDERER_PHYSICAL_GPU=7 \
  tabero_x tabero_x -lc '
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY
python -u "$ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py" \
  --server-host 127.0.0.1 --server-port 10010 \
  --task Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0 \
  --control-mode hybrid_tactile \
  --task-domain realworld --task-suite gentle_grasp --task-id 6 \
  --task-config-path "$ROOT/Tabero_X/benchmarks/datasets/realworld/config/final_vtla_rl_config" \
  --language-instruction "pick up the Vitasoy and put it into the basket" \
  --hdf5-folder "$ROOT/results/'"$RUN"'/empty_hdf5" \
  --num-total-experiments 1 --max-inference-steps 1 \
  --replan-steps 10 --num-steps-wait 5 --num-success-steps 8 \
  --seed 11 --prompt-seed 11 --prompt-adverb "" \
  --debug-mode 6 \
  --debug-path "$ROOT/results/'"$RUN"'/raw/task_6" \
  --debug-run-dir "$ROOT/results/'"$RUN"'/raw/task_6/artifacts" \
  --device cuda:0 --invert-gripper-output --no-smooth-action-chunks --headless'
```

若 Isaac/Omniverse 必须保留物理 Vulkan GPU 编号，而 PyTorch 只看到 process-local
`cuda:0`，使用运行目录内审计过的 split-GPU helper 包装客户端；不要把物理编号直接
当作 process-local CUDA 编号。把上面客户端命令的 Python 入口替换为：

```bash
python -u "$ROOT/results/$RUN/run_client_split_gpu.py" \
  "$ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py" \
  --server-host 127.0.0.1 --server-port 10010 \
  --task Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0 \
  --device cuda:0 --headless
```

该 helper 属于具体实验，应保存在对应 `results/<run>` 目录并在实验记录中写入哈希，
不要为一次性 GPU 映射把它加入主仓库；实际运行还需保留上一命令中的 suite、task、
reset、prompt、步数和 debug 参数。`debug mode 6` 会在每个 episode 下生成
`preview.mp4`、`forces.jsonl` 和 `exp_meta.json`。视频要求 FFmpeg 支持 H.264/
`libx264` 与 `yuv420p`；Dockerfile 不固定某个 FFmpeg 小版本，本镜像已通过实际解码
帧数校验。

### 校验 mode 6 制品

客户端结束后始终运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  .codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py \
  "$TABERO_RESULTS_HOST/$RUN"

PYTHONDONTWRITEBYTECODE=1 python3 \
  .codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py \
  --json "$TABERO_RESULTS_HOST/$RUN" \
  > "$TABERO_RESULTS_HOST/$RUN/artifact_validation.json"
```

只有视频帧、环境步、force 行数和元数据全部一致时，Isaac 在制品写完后的 weak-reference
退出错误才可归类为 post-artifact teardown warning。0/1 success 是 smoke 的策略结果，
不能据此判断正式策略质量。
