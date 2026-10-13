# 2026-10-13-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-larger-normalized-reward-6-gpu-env

参考 [2026-10-10 实验](../2026-10-10-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-normalized-reward-6-gpu-env/Readme.md)，从相同 Stage 2 `global_step_400` 起点训练至累计 step 800。本次增大归一化力奖励系数及力阈值范围，作为独立实验记录。

配置：[isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_larger_normalized_reward_6_env_gpu.yaml](../../RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_larger_normalized_reward_6_env_gpu.yaml)。镜像：`ccr.ccs.tencentyun.com/vtla/vtla:0.8.4`。

## 参数与训练起点

| 参数 | 本次设置 |
|---|---|
| 框架 / 方法 | RLinf，同步 PiRL/PPO，多任务 vitasoy / coca_cola / cookie |
| 模型输入 | no-state；`discrete_state_input: false`，沿用参考实验 |
| 基模 | `models/VTLA-RL-sft-lora-franka-no-adverb-05-effort/step_30000` |
| 恢复 checkpoint | `models/2026-9-29-task820-no_adverb-pirl-multi_tasks-30kbase-300step/global_step_400` |
| 环境 | `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`；配置中的 stage3 环境目录 |
| GPU | env `0-5`，rollout / actor `0-7`；`pipeline_stage_num: 1` |
| 并行环境 | train 192，每个 env worker 32，每任务 64；eval 96 |
| batch | micro/global 48/384，actor world size 8，梯度累积 1 |
| 训练预算 | 累计 runner step 400 → 800，400 次迭代；每次 5760 个 chunk slots、15 个优化器 batch |
| 训练模块 | action expert LoRA rank 32、value head；`freeze_non_lora: true` |
| 优化器 | actor LR `1e-6`、value LR `1e-4`；Adam betas 0.9/0.95、eps `1e-8`、weight decay 0.01、clip grad 1.0；constant scheduler、warmup 0 |
| 精度 / seed | 冻结参数 bf16、可训练参数 fp32、autocast bf16；seed 42 |
| Gradient checkpointing | disabled |
| 观测 / 动作 | 双相机、触觉历史 8 帧；action horizon 50、执行 chunk 10、动作维度 13 |
| 保存 / 评估 | 每 10 个 runner step 保存；`val_check_interval: -1`，关闭周期评估 |

10 月 10 日 README 指定的是 Stage 2 step 400，没有记载机器上的绝对路径。本次显式使用本地已下载的同一 Stage 2 checkpoint；来源见 [step400 下载记录](../../Record/experiments/20261005-task820-no_adverb-pirl-multitasks-step400-600-bundle-export.md)。`runner.resume_dir` 必须指向上述完整目录，而非其中的 `bundle` 或 `exported_model`。基模仍作为模型结构及冻结权重来源，不是从 SFT step 30000 重新开始 PPO。

归一化力分量先裁剪，再乘 `coefficient: 2.0`：

$$
r_{\mathrm{effort}}=2\operatorname{clip}\left(\frac{\mathrm{mid\_effort}-\mathrm{effort}}{\mathrm{mid\_effort}-\mathrm{min\_effort}},-1,1\right)
$$

| 任务 | min_effort | mid_effort | 推导 max_effort | coefficient |
|---|---:|---:|---:|---:|
| vitasoy | 2.0 | 34.5 | 67.0 | 2.0 |
| coca_cola | 4.0 | 9.2 | 14.4 | 2.0 |
| cookie | 3.0 | 26.0 | 49.0 | 2.0 |

有效成功时发放 `terminal_reward + r_effort`，基础成功奖励为 1，因此在环境外部倍率前范围为 `[-1, 3]`；成功判定独立于奖励正负。样本少于 4 时力分量为零，`contact_epsilon: 1.0`。配置声明的 `normalize_advantages: false` 在当前 actor → GAE 路径未传递，实际仍采用 GAE 默认归一化。

## 启动命令

当前仅准备实验和构建镜像，尚未启动训练。以下命令用于完成运行前确认后的正式启动；占用全部 8 张 GPU，沿用容器、no-state、同步及上述环境。

仓库根目录执行（本机使用已构建镜像；其他机器需先取得该镜像）：

```bash
tmux new -s task820-20261013-stage3
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:0.8.4"
docker compose run --rm rlinf
```

容器内执行，显式覆盖恢复路径及独立日志目录，无需修改公共 YAML：

```bash
export EXPERIMENT_NAME="2026-10-13-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-larger-normalized-reward-6-gpu-env"
export EXPERIMENT_DIR="$ROOT/results/$EXPERIMENT_NAME"
export STAGE2_CHECKPOINT="$ROOT/models/2026-9-29-task820-no_adverb-pirl-multi_tasks-30kbase-300step/global_step_400"
export WANDB_RUN_ID="task820-20261013-stage3-larger"
export WANDB_RESUME=allow
test -d "$STAGE2_CHECKPOINT/actor/dcp_checkpoint"
test -f "$STAGE2_CHECKPOINT/multi_task_state.json"
mkdir -p "$EXPERIMENT_DIR/train"
set -o pipefail
python ./examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/RLinf/examples/embodiment/config" \
  --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_larger_normalized_reward_6_env_gpu \
  runner.resume_dir="$STAGE2_CHECKPOINT" \
  runner.logger.log_path="$EXPERIMENT_DIR/train" \
  runner.logger.experiment_name="$EXPERIMENT_NAME" \
  2>&1 | tee -a "$EXPERIMENT_DIR/train/launch.log"
```

同一实验重启时保持 W&B name / ID 不变，并将 `resume_dir` 改为本实验最后完整 checkpoint；以上 step400 命令仅用于本实验首次启动。
