# 2026-10-29-melt

这里需要完成 pirl 的消融实验

所有模型的统一输入为：

* no-state (指的是不输入机械臂关节角度)

两阶段统一为

* stage1 +- 1cm 随机化 0-200 step
* stage2 +- 1cm 随机化 + 根据力大小手动设定的质量随机化，累计 200-400 step

两阶段使用的场景分别为：

* [gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1)
* [gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2)

本实验使用 `ccr.ccs.tencentyun.com/vtla/vtla:0.9.7`，从已验证的 `0.9.6` 增量构建，仅覆盖源码、配置和启动脚本，沿用原有运行环境。构建与验证记录在 `results/docker-melt-20261029/`。本次没有启动训练或仿真。

## 训练脚本

在宿主机通过 [run-melt.sh](run-melt.sh) 调用 Docker，顺序为 **vision-only stage1/2 → tacimg stage1/2 → success-only-reward stage1/2**。默认只显示计划，显式 `--run` 才启动训练。

```bash
bash experiments/2026-10-29-melt/run-melt.sh --dry-run
# 检查 0.9.7 镜像源码、模型、配置和 HF 身份：
bash experiments/2026-10-29-melt/run-melt.sh --check
# 审阅上述 no-state、同步 RLinf、场景及训练参数后，在宿主机 tmux 中启动：
tmux new -s task820-melt
bash experiments/2026-10-29-melt/run-melt.sh --run
```

`--check` 不启动训练、仿真或 W&B run，也不创建 HF 仓库/上传模型；它会在 `results/2026-10-29-melt/_launcher/` 保存配置、场景、启动脚本及源码哈希快照。镜像缺失时直接停止，不自动 pull/build；镜像源码与本地不一致时停止，不自动同步或切换镜像。`--run` 要求 GPU 0–7 空闲，不会终止其他任务。

每组两个阶段复用 `_launcher/<variant>.wandb_id` 保存的单一 W&B 身份；stage1 使用 `WANDB_RESUME=never`，stage2 使用 `must`。失败阶段停止整个顺序，再次执行会要求先检查失败原因，不自动重启；已完成阶段会核对 checkpoint 和上传凭据后跳过。运行日志、GPU/磁盘日志、每 15 秒的训练指标检查、清理日志、W&B 链接和完整启动命令保存在 `_launcher/<variant>-stageN/`。检测到非有限训练指标或维护进程异常时停止本次训练容器。

正式 `--run` 自动创建 `Record/experiments/<完整时间>-task820-no_adverb-pirl-melt.md`，记录参数快照、源码版本、tmux、每组 W&B/HF 链接、checkpoint、失败事件及训练耗时。此脚本不包含训练后的仿真评估。

脚本从仓库根目录 `.env` 获取 `HF_TOKEN`，通过 token 查询用户名，为每个实验创建一个 public 模型仓库，仓库名称为 `experiment_name`。每完成 100 步，后台上传完整的 `checkpoints/global_step_N/` 到该仓库的 `global_step_N/`，包括 actor DCP 权重、优化器、调度器、RNG、trainable weights sidecar 和多任务状态（PiRL 为 on-policy，不使用 replay buffer），下载后可直接作为 `resume_dir` 使用。

上传在独立后台任务中进行；失败重试，阶段结束时必须上传并核对该阶段两个整百 checkpoint 的远端文件清单及大小，才能进入下一阶段。上传凭据保存在 `_launcher/hf/<variant>/`，可重试而不重复上传已验证内容。

独立维护进程每 15 秒检查保存点，验证 DCP 权重/优化器/调度器/RNG、trainable sidecar 和多任务状态后，滚动保留最近两个可 resume 的续训点，所有整百点永久保留。没有完整结束标记、校验失败或正在写入的目录不会删除；上传只读取永久保留的整百点。

## Vision only

[sft 模型链接](https://huggingface.co/Demomasterlqx/task820_firm_mixed_pi05_vision_nostate_force005_sft_2a800_80k_20261017-222046/tree/main/step_30000)

只是关闭力输入，其他训练配置和 [2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md) 的 pirl stage1/2 没有区别

训练配置

* [2026-10-29-melt/vision-only/stage1](../../RLinf/examples/embodiment/config/2026-10-29-melt/vision-only/stage1.yaml)
* [2026-10-29-melt/vision-only/stage2](../../RLinf/examples/embodiment/config/2026-10-29-melt/vision-only/stage2.yaml)

## TacImage

[sft 模型链接](https://huggingface.co/Demomasterlqx/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/tree/main/step_30000)

只是将力输入配置从 TacField 改为 TacImage，其他训练配置和 [2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md) 的 pirl stage1/2 没有区别

训练配置

* [2026-10-29-melt/tacimg/stage1](../../RLinf/examples/embodiment/config/2026-10-29-melt/tacimg/stage1.yaml)
* [2026-10-29-melt/tacimg/stage2](../../RLinf/examples/embodiment/config/2026-10-29-melt/tacimg/stage2.yaml)

## Success-only reward

[sft 模型链接](https://huggingface.co/Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort/tree/main/step_30000)

其他训练配置以及模型和 [2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md) 的 pirl stage1/2 没有区别。唯一的区别为在 stage2 的时候关闭所有力奖励。

训练配置

* [2026-10-29-melt/success-only-reward/stage1](../../RLinf/examples/embodiment/config/2026-10-29-melt/success-only-reward/stage1.yaml)
* [2026-10-29-melt/success-only-reward/stage2](../../RLinf/examples/embodiment/config/2026-10-29-melt/success-only-reward/stage2.yaml)

## 已落实的配置与输入支持

| 实验 | 模型输入 | stage1 奖励 | stage2 奖励 |
| --- | --- | --- | --- |
| vision-only | 主相机 + 腕相机；不读取触觉图、marker motion 或测量力 | 成功 1 | 成功 1 + normalized effort ×10 |
| tacimg | 主相机 + 腕相机 + 224×224 触觉历史拼图（双指、8 帧） | 成功 1 | 成功 1 + normalized effort ×10 |
| success-only-reward | 主相机 + 腕相机 + TacField `[9,440,2]`，TCN prefix | 成功 1 | 成功 1，关闭全部力奖励 |

三组均使用 `model_type: openpi`、`discrete_state_input: false`。环境 state 是末端位置/姿态和夹爪开度，不含机械臂关节角度；保留 `use_proprio: true` 供归一化及 delta/absolute 动作转换，Pi0.5 不添加 state prompt tokens。关闭触觉输入不会关闭仿真触觉传感器、奖励侧测力或 13D 动作中的 6D 目标力输出。TacImage/TacField 是触觉表征，不是直接输入测量力数值。

RLinf 原有 TacImage 和 TacField 路径可复用；本次补齐 vision-only 的 [模型注册](../../RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py)、[输入转换](../../RLinf/rlinf/models/embodiment/openpi/policies/tabero_policy.py)、[环境观测路由](../../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py) 和 PiRL 接口校验。TacImage/vision 的 `tactile_streams: []`，不建立 TacField encoder。三组 norm stats 分别取对应基模的 `assets/pi05_horizon50_{vision,tacimg,tacfield}_task820_firm_mixed/norm_stats.json`，不混用旧的 `..._firm` 路径。

共同参数沿用参考实验：8 GPU（actor/rollout 0–7，env 0–5），192 个训练环境、96 个 eval 环境，每轮最多 300 环境步，horizon 50、执行 chunk 10；micro batch 48、global batch 384、seed 42；action-expert LoRA rank 32，VLM 冻结，value head 可训练，gradient checkpointing **disabled**。AdamW actor/value LR 为 `1e-6/1e-4`，betas `0.9/0.95`，eps `1e-8`，weight decay `0.01`，clip grad `1.0`，constant scheduler、warmup 0；冻结参数 bf16、训练参数 fp32、bf16 AMP。gamma `0.99`、GAE lambda `0.95`、每轮 update epoch 1。训练期间 eval 关闭，每 10 轮保存，TensorBoard/W&B 均开启。配置保留参考实验的 `tactile_loss_weight: 0.01`；它是 SFT 监督损失参数，不是 PPO 环境力奖励开关。

200/400 指累计 rollout 轮数，不是优化器更新次数。三组 stage1 均从各自 SFT `step_30000` 初始化；stage2 只恢复本组新产出的完整 `global_step_200`，不复用旧 TacField 实验 checkpoint。各组 `experiment_name` 为 `2026-10-29-melt-<variant>-pirl`，输出位于 `results/2026-10-29-melt/<variant>/`，checkpoint 位于其 `<experiment_name>/checkpoints/`；TensorBoard/W&B 本地目录分别为 `tensorboard/`、`wandb/`。同组两阶段由启动脚本复用同一 W&B run ID。

力奖励使用现有 `max_effort` 字段（替代已弃用的 `mid_effort`），Vitasoy/Coca-Cola/cookie 的 min/max effort 为 `2/36`、`4/12`、`3/30`，reward bounds 为 `-1/1`。success-only 在公共配置和每个物体覆盖中均关闭 normalized effort，旧 `force_bonus` 也关闭；质量随机化仍使用共同 stage2 场景。

验证：六份配置通过 Hydra 解析、PiRL 接口校验、基模/归一化文件路径、数据转换构造和 train/eval 逐物体奖励开关检查；7 项合成输入/接口单元检查通过，Ruff 检查通过。meta-device 模型与本地 safetensors 的键名/形状核对通过：vision/TacImage 各 812 个权重张量，TacField 828 个，仅允许新 value head 与 tied-weight alias 缺失；未实际加载权重或执行模型 forward。检查借用本地 0.9.6 镜像依赖并只读挂载当前源码，无 GPU、无训练/仿真。以上记录属于构建前检查；0.9.7 镜像本身的检查以 `results/docker-melt-20261029/release-validation.json` 为准，不代表 worker 启动、完整权重加载或训练效果通过。

脚本验证：Bash 语法、六阶段 dry-run、镜像缺失拒绝启动、合成 DCP/checkpoint 的完整性与保留规则，以及模拟 Docker 的阶段顺序、W&B 身份、上传、完成阶段跳过和失败停止均有独立检查，验证文件位于 `results/2026-10-29-melt/_script_validation/`。这些是离线/模拟检查，尚未实际启动训练、连接 W&B 或向 HF 上传。
