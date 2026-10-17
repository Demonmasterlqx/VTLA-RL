# Task820 三任务 PiRL：±2 cm 位置随机化与质量课程

使用基模 [Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort](https://huggingface.co/Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort) 的 `step_30000`，共同训练 Vitasoy、Coca-Cola、cookie。
模型为 no-state Pi0.5，输入双视角图像和 TacField；训练 action-expert LoRA 与 value head，采用同步 PPO/GAE，梯度检查点关闭。

| 阶段 | 配置 | 累计训练区间 | 环境与奖励 |
| --- | --- | --- | --- |
| Stage 1 | [stage1.yaml](../../RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_1_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm.yaml) | 0 → 200 | 三个物体 X/Y 各 ±2 cm，质量固定，仅成功奖励 |
| Stage 2 | [stage2.yaml](../../RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_2_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm.yaml) | 200 → 400 | 保留位置随机化，增加下表质量随机化，仅成功奖励 |
| Stage 3 | [stage3.yaml](../../RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm.yaml) | 400 → 800 | 保留 Stage 2 随机化，增加系数 10 的归一化力奖励 |

三个阶段使用目标环境目录各自的 `stage1/2/3/gentle_grasp.json`，生效的位置等级为 `xy_2cm`。三个物体的 X、Y 分别围绕原中心独立随机 ±0.02 m；Z 和姿态固定，篮子位置固定，不交换物体位置。夹爪静/动摩擦系数固定为 0.5/0.5。

Stage 2/3 保留目标 JSON 原有的质量范围，每次 reset 均匀采样：

| 对象 | 质量范围 |
| --- | --- |
| Vitasoy | 0.1–0.6 kg |
| Coca-Cola | 0.2–0.7 kg |
| cookie | 0.3–0.8 kg |
| 篮子 | 0.2–0.5 kg |

`delta_500g` 是实验名称，实际质量以上表为准。Stage 3 的力奖励阈值沿用参考配置：Vitasoy min/mid 为 2/36 N，Coca-Cola 为 4/12 N，cookie 为 3/30 N；成功奖励为 1.0。

三个阶段均使用 env GPU 0–5、rollout/actor GPU 0–7；192 个训练环境，每任务 64 个。micro/global batch 为 48/384，梯度累积 1；每轮采样 300 步、5760 个动作 chunk 槽位，`update_epoch=1` 对应 15 个 PPO minibatch。actor/value 学习率分别为 1e-6/1e-4。

在仓库根目录启动容器，建议在宿主机持久 tmux 会话中执行：

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:0.8.7"
docker compose run --rm rlinf
```

镜像从 0.8.5 增量构建，包含上述配置和 USD 资产；基础模型通过宿主机 `models` 目录只读挂载，须包含 `VTLA-RL-sft-lora-franka-no-adverb-05-effort/step_30000` 及其 norm stats。凭据使用根目录 `.env` 挂载。

在容器内设置日志位置和 W&B 标识；同一实验的后续阶段和恢复沿用这些值：

```bash
cd "$ROOT/RLinf"
export RUN_LOG="$ROOT/results/2026-10-17-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-10-times-normalized-reward-6-gpu-env-delta-500g-+-2cm"
export RUN_NAME="task820_multi_tasks_pirl_delta_500g_xy_2cm"
export WANDB_RUN_ID="task820-pirl-20261017-delta500g-xy2cm"
export WANDB_RESUME=allow
mkdir -p "$RUN_LOG"
set -o pipefail
```

Stage 1：从 SFT 基模训练到第 200 轮。

```bash
python ./examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/RLinf/examples/embodiment/config" \
  --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_1_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm \
  runner.logger.log_path="$RUN_LOG" \
  runner.logger.experiment_name="$RUN_NAME" \
  2>&1 | tee -a "$RUN_LOG/stage1.log"
```

Stage 2：确认前一阶段完整 checkpoint 已保存后，从第 200 轮继续到第 400 轮。

```bash
export WANDB_RESUME=must
python ./examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/RLinf/examples/embodiment/config" \
  --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_2_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm \
  runner.logger.log_path="$RUN_LOG" \
  runner.logger.experiment_name="$RUN_NAME" \
  2>&1 | tee -a "$RUN_LOG/stage2.log"
```

Stage 3：确认前一阶段完整 checkpoint 已保存后，从第 400 轮继续到第 800 轮。

```bash
export WANDB_RESUME=must
python ./examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/RLinf/examples/embodiment/config" \
  --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_10_times_normalized_reward_6_env_gpu_delta_500g_xy_2cm \
  runner.logger.log_path="$RUN_LOG" \
  runner.logger.experiment_name="$RUN_NAME" \
  2>&1 | tee -a "$RUN_LOG/stage3.log"
```

Stage 2/3 YAML 已分别设置恢复 `$RUN_LOG/$RUN_NAME/checkpoints/global_step_200` 和 `global_step_400`，400/800 是累计目标。完整 checkpoint 包含 actor、优化器及多任务状态；不要用单独的模型权重替代。中断重启时使用 `runner.resume_dir=<同一阶段最新完整 checkpoint>`，并保持原 W&B run ID。

每 10 轮保存 checkpoint，关闭周期评估。TensorBoard 位于 `$RUN_LOG/tensorboard`，W&B 本地文件位于 `$RUN_LOG/wandb`，W&B 项目为 `tabero-rlinf`。实际训练时在 `Record` 记录启动命令、参数、日志、checkpoint、W&B 链接、事件和总时长。
