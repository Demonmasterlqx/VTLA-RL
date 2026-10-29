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

计划使用 0.5.0 版本的 Docker image，从 0.4.0 增量构建。**该镜像尚未构建，本次没有启动训练或仿真。**

## 训练脚本（待实现）

计划在宿主机通过 `run-melt.sh` 调用 Docker，依次完成三组实验各自的 stage1/2。当前已补齐六份训练配置，但 `run-melt.sh` 尚不存在；下述上传与清理是脚本需求，不会由 YAML 自动执行。

脚本从仓库根目录 `.env` 获取 `HF_TOKEN`，通过 token 查询用户名，为每个实验创建一个 public 模型仓库，仓库名称为 `experiment_name`。每完成 100 步，后台上传完整的 `checkpoints/global_step_N/` 到该仓库的 `global_step_N/`，包括 actor DCP 权重、优化器、调度器、RNG、trainable weights sidecar 和多任务状态（PiRL 为 on-policy，不使用 replay buffer），下载后可直接作为 `resume_dir` 使用。

脚本将会滚动保留最近两个可resume的续训点，整百的不会删除。

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

200/400 指累计 rollout 轮数，不是优化器更新次数。三组 stage1 均从各自 SFT `step_30000` 初始化；stage2 只恢复本组新产出的完整 `global_step_200`，不复用旧 TacField 实验 checkpoint。各组 `experiment_name` 为 `2026-10-29-melt-<variant>-pirl`，输出位于 `results/2026-10-29-melt/<variant>/`，checkpoint 位于其 `<experiment_name>/checkpoints/`；TensorBoard/W&B 本地目录分别为 `tensorboard/`、`wandb/`。同组两阶段应复用同一 W&B run ID，待启动脚本落实。

力奖励使用现有 `max_effort` 字段（替代已弃用的 `mid_effort`），Vitasoy/Coca-Cola/cookie 的 min/max effort 为 `2/36`、`4/12`、`3/30`，reward bounds 为 `-1/1`。success-only 在公共配置和每个物体覆盖中均关闭 normalized effort，旧 `force_bonus` 也关闭；质量随机化仍使用共同 stage2 场景。

验证：六份配置通过 Hydra 解析、PiRL 接口校验、基模/归一化文件路径、数据转换构造和 train/eval 逐物体奖励开关检查；7 项合成输入/接口单元检查通过，Ruff 检查通过。meta-device 模型与本地 safetensors 的键名/形状核对通过：vision/TacImage 各 812 个权重张量，TacField 828 个，仅允许新 value head 与 tied-weight alias 缺失；未实际加载权重或执行模型 forward。检查借用本地 0.9.6 镜像依赖并只读挂载当前源码，无 GPU、无训练/仿真。尚不代表目标 0.5.0 镜像、worker 启动、完整权重加载或训练效果通过；启动前仍需构建镜像并核对源码一致性。
