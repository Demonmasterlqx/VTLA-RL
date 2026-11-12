# 2026-11-12-tacimg-rlt-training

使用已完成的 TacImage Stage1 SFT encoder，训练 **RLT 残差 MLP actor 与双 Q**。目录和启动流程参考 [TacImage DSRL](../2026-11-10-tacimg-dsrl-training/Readme.md)，RLT 优化与 transition replay 沿用 [历史 RLT 配置](../../RLinf/examples/embodiment/config/2026-10-22-adaptive-pirl-dsrl-rlt-compare/isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage1.yaml)。

**配置和脚本已实现。镜像固定为 `ccr.ccs.tencentyun.com/vtla/vtla:0.10.2`，从已核验的 0.10.0 增量构建；镜像与发布验证状态见 [release-validation.json](../../results/2026-11-12-tacimg-rlt-training/image/release-validation.json) 和 [publication-validation.json](../../results/2026-11-12-tacimg-rlt-training/image/publication-validation.json)。没有启动训练或仿真；实际模型加载、多卡启动、完整恢复与 HF checkpoint 上传仍须另行验证。**

0.10.2 在继承环境中清理了 NumPy/SciPy wheel RECORD 之外的残留文件，解决基底 RLinf/T2-VLA 的混合版本导入问题。已安装 wheel 文件先做哈希校验；NumPy 版本仍分别为 1.26.0 / 1.26.4，RLinf SciPy 仍为 1.16.3，IsaacLab 环境不变。清理清单保存在镜像 `/opt/vtla-build/runtime-wheel-cleanup.json`。

## 模型来源与输入

| 组件 | 本地路径 | Hugging Face |
|---|---|---|
| 冻结 TacImage VLA 基模 | `models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000` | [基模仓库](https://huggingface.co/Demomasterlqx/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823) |
| 冻结 Stage1 encoder | `models/VTLA-RL-sft-task820-firm-mixed-tacimg-nostate-rlt-stage1/step_10000/rlt_encoder.safetensors` | [Encoder 仓库](https://huggingface.co/Demomasterlqx/VTLA-RL-sft-task820-firm-mixed-tacimg-nostate-rlt-stage1) |

基模是 SFT step30000 的 EMA merged-LoRA 导出。基模 revision 为 `4f03949602ea7115970a5c58fa1c73363eb1fe35`，subfolder 为 `step_30000`，模型 SHA-256 为 `6e549631ace7d4ddad23ffefbb0e6b7d1b8ee41ba43beecf93a22b4be9f9fe0c`。

Encoder 来自已经完成的 10,000 次更新的特征重建 SFT，HF revision 为 `d8e990c80f4a975e5cadfc4c276d9b23721f7faf`，`rlt_encoder.safetensors` SHA-256 为 `4c0a5a3007d580eafce5228f611b6351ec97335172b5389ee893ad612383a1a8`。启动检查会核对 encoder manifest 中的基模哈希、归一化来源和结构。这里的 RL Stage1/2 是下面两段奖励课程，**不会重复训练 encoder SFT**。

| 接口 | 约定 |
|---|---|
| 图像 | 主相机、腕部相机、触觉 RGB 历史拼图，3 路 |
| 触觉拼图 | 左右触觉相机各 8 帧，4×4、224×224 RGB；沿用现有仿真 preprocessing |
| 语言 | 三物体原始任务文本，无副词 |
| no-state | VLA 的 `discrete_state_input: false`，不输入关节状态 token |
| RLT proprio | actor/critic 保留归一化的 7 维 EEF/gripper 状态；整个 RLT actor 并非完全无 proprio |
| TacField / 实测力输入 | 不使用 marker-motion/TCN prefix，`tactile_streams: []`；实测力只供环境奖励等逻辑使用 |
| 动作 | chunk 10 × 13 维；其中保留 6 维力输出，`tactile_dim: 6` 不代表输入实测力 |
| Encoder | input/embed 2048，1 RL token，2 层，8 heads，MLP ratio 4，prefix capacity 1024；image_only=false，use_mask=true |
| 归一化 | 基模内 `assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json`，与 encoder 所带资产逐值一致 |

`future_force_loss_weight=0.05` / `tactile_loss_weight: 0.05` 记录基模的 SFT 来源。本次 RLT 使用 BC 正则与 Q 目标，不训练新的 VLA LoRA，也不新增 SFT 力损失。实际训练仅更新 MLP actor 和 twin-Q；VLA、encoder 冻结且在 rollout 端提取特征。

## 残差动作约定

```yaml
actor:
  model:
    action_parameterization: residual
    normalized_action_bound: 4.0
    fixed_std: 0.05
algorithm:
  rlt_route:
    actor_scope: full_task
```

在 `model_normalized` 空间中：`delta = 4 * tanh(raw / 4)`，`action = reference_chunk + delta`，随后执行既有 model→environment 动作转换。4.0 约束的是残差；不对相加后的动作再次 tanh。actor 末层零初始化使确定性初始动作等于参考动作；采样时保留固定标准差 0.05 的探索。

`full_task` 表示三个任务共用一个覆盖完整动作的 actor 路由，与 `action_parameterization` 是独立设置；本实验明确采用 residual，不使用 absolute/direct。actor 输入为 reference chunk 130 + encoder 特征 2048 + proprio 7 = 2185 维，输出 130 维。BC 目标约束组合动作接近 reference，Q 项优化组合动作的价值。

actor、rollout 和导出/恢复元数据统一声明：

```json
{"action_parameterization":"residual","action_composition":"reference_plus_bounded_delta","residual_action_bound":4.0}
```

恢复检查拒绝 absolute、缺失动作约定或残差尺度不一致的 checkpoint。后续导出应使用已有 `export_tabero_rlt_for_t2vla.py` 并保留此约定；本脚本上传完整训练 checkpoint，不额外生成推理 bundle。

## 两阶段 RL 课程

| 阶段 | 累计 rollout 轮数 | 初始化 / 恢复 | 场景与奖励 |
|---|---:|---|---|
| RL Stage1 | 0→200 | 冻结上述基模与 encoder，初始化残差 actor/twin-Q | XY ±1 cm；成功奖励 1；力奖励关闭 |
| RL Stage2 | 200→400 | 完整恢复本实验 `global_step_200` | XY ±1 cm，自适应质量；成功奖励 1 + 归一化力奖励，系数 10 |

直接使用参考实验的 [Stage1 场景](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1/gentle_grasp.json) 和 [Stage2 场景](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2/gentle_grasp.json)，不修改 JSON。

| 目标 | Stage1 质量 kg | Stage2 质量 kg | min_effort / max_effort |
|---|---:|---:|---:|
| Vitasoy | 0.10 | 0.10–0.60 | 2 / 36 |
| Coca-Cola | 0.20 | 0.15–0.25 | 4 / 12 |
| cookie | 0.30 | 0.30–0.70 | 3 / 30 |

多任务调权保持 EMA 0.9、范围 0.5–2.0、sigmoid scale 10。力奖励至少 4 个有效样本，contact epsilon 1；使用当前字段 `max_effort`，对应历史 YAML 的 `mid_effort` 数值。当前工作区累积的法向力为 `abs(left Fz) + abs(right Fz)`，保留双指接触有效性门限；以未来实际构建进 0.10.2 并通过源码校验的实现为准。归一化奖励默认范围 [-1,1]，乘系数后 [-10,10]，max_effort 为零点。

## 训练参数

配置：[stage1.yaml](../../RLinf/examples/embodiment/config/2026-11-12-tacimg-rlt-training/stage1.yaml)、[stage2.yaml](../../RLinf/examples/embodiment/config/2026-11-12-tacimg-rlt-training/stage2.yaml)。

| 参数 | 值 |
|---|---|
| 框架 / 入口 | RLinf 同步 RLT AC；`train_embodied_agent.py`；loss_type=rlt_ac |
| 环境 | `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`；Task820 / gentle_grasp task6 三物体 |
| reset / chunk 边界 | task_config_default_reset / terminal_safe_v1；auto_reset=false，ignore_terminations=false |
| GPU | 单节点 8 卡：actor/rollout 0–7，env 0–5，通过 YAML placement 分配 |
| 环境规模 | train 192（每物体64）；eval 配置96，训练期间周期评估关闭 |
| 每轮 / episode 上限 | 300 环境步；rollout_epoch=1 |
| VLA horizon / chunk / 去噪步 | 50 / 10 / 10 |
| Batch | 每 rank micro48，全局384，8 rank，梯度累积1 |
| Actor / critic optimizer | Adam，lr 1e-4 / 3e-4，clip 3.5 / 10；两者 betas=.9/.999，eps=1e-8，weight decay=0 |
| LR scheduler | torch_constant；无 LR warmup；与下述 BC/Q 权重 warmup 区分 |
| Gamma / tau | .995 / .005；每次更新软更新 target |
| Q 聚合 | target min；actor q1；标准双 Q |
| BC / Q 权重 | 前2000个 learner updates 为7/.05，随后5000个 updates 线性过渡到2.5/.45 |
| Reference dropout | 0 |
| 熵温度 | fixed_alpha=0，无 alpha 优化器 |
| 更新节奏 | critic:actor=4:1；每32个新增 transition 产生更新预算；每轮最多500次 learner updates |
| Warmup | 最少256个 replay 样本；warmup_post_collect_updates=2000；train_actor_steps=1000，实际更新数由 RLT schedule 决定 |
| Replay | 原生 TrajectoryReplayBuffer、transition replay；cache10000，sample_window_size100000（trajectory窗口），min_buffer_size1，auto_save=true，enable_preload=false |
| 精度 / FSDP | MLP actor/Q FP32；no_shard、use_orig_params=false；feature precision=null，沿用 OpenPI loader 的参数转换；actor不offload，rollout offload |
| Gradient checkpointing / seed | disabled / 42 |
| 保存 / 上传 | 每10轮保存；每100轮上传 |

优化器采用当前 `FSDPModelManager.build_optimizers()` 的 filtered Adam 路径，显式记录其默认值；没有复制 DSRL actor 的 beta2=.96。RLT replay 保存编码后的特征、参考动作及 transition，使用自身 trajectory checkpoint 格式，不使用 `compact_dsrl_ring`。

YAML 保留历史的 `max_num_samples: 100000`，但当前 worker 不将此字段传入 TrajectoryReplayBuffer，因此不能把它视作生效的 transition 容量限制；实际采样窗口由 `sample_window_size` 控制。

`max_steps/max_epochs=200/400` 是累计 rollout 轮数，**不是 optimizer steps**。Stage2 恢复 actor/critic、target Q、优化器、LR scheduler、RNG、replay、多任务权重和 RLT learner schedule；BC/Q 调度累计延续，不重置为新实验。

## 启动与监督

[run_stage12.sh](./run_stage12.sh) 是宿主机 Bash 编排器，遵循 [Docker 启动约定](../../docker/README.md)。仅在指定镜像中运行正式预检和训练，不自动 build/pull，不替换镜像版本。

```bash
# 现在可执行；不调用 Docker、不读取 .env、不写运行目录。
bash experiments/2026-11-12-tacimg-rlt-training/run_stage12.sh --dry-run

# 完成镜像与训练约定核对后：校验源码、模型来源、配置、环境、HF身份；不训练、不上传。
bash experiments/2026-11-12-tacimg-rlt-training/run_stage12.sh --check

# 后续实际训练命令，在宿主机命名 tmux 会话中运行。
tmux new -s task820-tacimg-rlt
bash experiments/2026-11-12-tacimg-rlt-training/run_stage12.sh --run
```

GPU 必须空闲；不通过 `CUDA_VISIBLE_DEVICES` 分配 RLinf GPU，不抢占其他进程。支持 Compose 既有 `TABERO_MODELS_HOST`、`TABERO_RESULTS_HOST`、`TABERO_RECORD_HOST`、`TABERO_DATASETS_HOST` 挂载变量。脚本固定镜像ID并保存源码哈希、YAML、场景、Compose 和启动器快照；发现镜像/本地不一致时停止，不自动同步或忽略。

每15秒记录训练进度、GPU、磁盘、有限指标及 checkpoint 状态。30分钟没有新 RLT rollout 日志步、指标非有限、磁盘越界或监测进程退出时停止本次容器。预检要求 `/tmp` 至少2 GiB、results 至少256 GiB，运行下限为1 GiB / 64 GiB；这些是保护下限，不是完整训练容量估算。

阶段自动衔接；已完成阶段经复核后跳过。失败/中断会阻止后续阶段，保留故障日志和 checkpoint；重新运行不会擅自覆盖失败阶段。检查真实恢复点并审核同规格恢复命令后才能继续，不删除失败标记来强行重启。

## HF 自动上传与完成判定

- 运行时 `source .env`，**仅使用 `HF_TOKEN`**；不使用 `LQX_HF_TOKEN`，不要求 token 属于 encoder 的发布者。`whoami` 解析并持久化账号，每次实际上传再次核对账号。
- 自动创建 public 模型仓库 **`<HF_TOKEN对应账号>/2026-11-12-tacimg-rlt-training`**。已有仓库若为 private 则停止，不自动修改可见性。
- 每100轮上传完整 `global_step_100/200/300/400`，包括可恢复训练状态。encoder/基模仍通过 README 链接、配置和来源记录引用，不重复上传冻结基模。
- 等待 runner 原子写入 `multi_task_state.json` 后才校验/上传：检查 actor DCP 的模型、优化器、scheduler、RNG和分片范围；trainable sidecar；rank0 target；每rank的任务状态、replay索引/文件/RNG、`rlt_schedule`；核对残差语义及跨rank计数。固定alpha=0不要求alpha优化器文件。
- 上传失败自动重试；训练期间可继续，阶段完成前必须补齐对应两个百轮checkpoint。上传后核对远端固定commit的文件清单和大小，记录revision/URL与本地fingerprint；已验证且本地未变化的上传可跳过。
- 保留最近两个校验通过的完整恢复点和全部百轮checkpoint；只清理本实验未受保护的完整旧点。不完整或校验失败的目录保留。
- TensorBoard与W&B同时记录。项目`tabero-rlinf`、entity`183842220-hkust`、run name`2026-11-12-tacimg-rlt-training`；首次生成并保存一个run ID，Stage2用同一ID和`WANDB_RESUME=must`。
- 训练进程成功退出、最终checkpoint完整、百轮上传齐全、TensorBoard/W&B在零基step199/399具有有限actor/critic/BC/加权Q指标，才标记阶段完成。不以alpha_loss作为RLT完成要求。在线W&B失败时不自动建立新run或切换日志模式。

checkpoint校验是结构检查，不等于分布式恢复成功，也不检查所有replay张量内容；真实恢复另需运行验证。本脚本不附加独立策略评估矩阵或Stage3。

## 输出路径与验证状态

| 内容 | 路径 |
|---|---|
| 启动控制、配置/场景/源码快照 | `results/2026-11-12-tacimg-rlt-training/_launcher/` |
| 基模及encoder来源 | `_launcher/base_provenance.json` |
| 阶段命令、训练/监督/上传日志 | `_launcher/rlt-stage1/`、`_launcher/rlt-stage2/` |
| W&B ID / URL | `_launcher/rlt.wandb_id`、`_launcher/rlt.wandb_url.txt` |
| HF上传回执 | `_launcher/hf/rlt/global_step_N.json` |
| Checkpoint | `results/2026-11-12-tacimg-rlt-training/rlt/2026-11-12-tacimg-rlt-training/checkpoints/global_step_N/` |
| TensorBoard / W&B本地日志 | `results/2026-11-12-tacimg-rlt-training/rlt/tensorboard/`、同级`wandb/` |
| 正式运行报告 | `Record/experiments/<完整时间>-task820-no_adverb-rlt-tacimg_stage12.md` |
| 静态验证材料 | `results/2026-11-12-tacimg-rlt-training/_validation/` |

`--run`生成正式Record报告，包含真实命令、训练参数与来源、尝试/故障、日志和模型路径、阶段与总训练时长、唯一W&B链接、HF结果。当前没有训练结果或正式运行报告。

文件准备阶段的静态验证汇总见 [validation-summary.json](../../results/2026-11-12-tacimg-rlt-training/_validation/validation-summary.json)，涵盖语法、Hydra展开、阶段/参考差异、模型来源和启动器隔离行为。后续镜像源码、依赖导入与配置检查单独记录在上述镜像验证文件中；CPU配置检查使用模拟多卡 placement，不代表 GPU worker、仿真、真实模型加载或 checkpoint 恢复已经通过。
