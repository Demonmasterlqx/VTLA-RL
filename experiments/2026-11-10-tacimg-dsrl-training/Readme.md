# 2026-11-10-tacimg-dsrl-training

为 [Task820 TacImg no-state SFT 基模](https://huggingface.co/Demomasterlqx/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823) 实现 DSRL Stage 1/2。阶段设计及 DSRL 优化参数沿用 [2026-10-22 参考实验](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md)，适配 TacImg 输入、当前 compact replay 和独立实验输出。

**当前状态：配置和脚本已实现，尚未启动训练。镜像固定为 `ccr.ccs.tencentyun.com/vtla/vtla:0.10.1`，基于 0.10.0 构建；构建与验证状态见 [镜像记录](../../results/2026-11-10-tacimg-dsrl-training/image/release-validation.json)。模型加载、分布式恢复、仿真和实际上传尚未验证。**

## 基模与输入

本地基模：

```text
models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000
```

使用该目录的 `model.safetensors`、`config.json`、`training_contract.json`、`export_meta.json`，以及 `assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json`。这是 SFT step30000 的 EMA、merged-LoRA 导出，用于初始化新 DSRL 实验；Stage 2 的续训源是本实验自己的完整 RL checkpoint。

已核对模型文件 SHA-256 与导出元数据一致。归一化文件存在 JSON 排版差异，恢复为导出时的 JSON 序列化后，SHA-256 与元数据一致；预检接受这一格式差异，同时记录实际文件和导出序列化的两个哈希，拒绝归一化数值改变。

| 接口 | 本实验约定 |
|---|---|
| VLA 图像 | 主相机、腕部相机、触觉 RGB 历史拼图，共三路图像 |
| 触觉历史 | 左右触觉相机各 8 帧，组成 4×4、224×224 RGB 拼图 |
| VLA state | `discrete_state_input: false`；不向语言提示加入状态 token |
| DSRL actor state | `dsrl_actor_use_state: false` |
| critic / 动作转换 | 保留 7 维 EEF/gripper 状态，用于 critic、归一化和动作转换 |
| 旧 TacField 输入 | 关闭 marker-motion/TCN prefix；`tactile_streams: []`，prefix 参数置空 |
| DSRL 图像 | `dsrl_num_images: 3`，顺序 main/wrist/tactile；共享图像编码器后拼接 |
| `dsrl_use_tactile: false` | 关闭的是独立 marker-motion 编码器；第三路触觉 RGB 仍然输入 |
| 六维测得力 | 不作为 VLA/DSRL actor 输入；环境仍可用测得力计算奖励 |
| 动作接口 | 13 维，保留六维力输出；`tactile_dim: 6` 是动作输出部分 |

基模的 `future_force_loss_weight=0.05` 是 SFT 来源信息，配置中保留对应 `tactile_loss_weight: 0.05`；DSRL 使用 SAC 损失，不新增 SFT 力监督。Stage 2 的环境力奖励系数另为 10。VLA 基座冻结，仅训练 DSRL 噪声策略、actor 图像编码器、critic 图像/状态编码器和 Q heads，以及熵温度；不训练新的 VLA LoRA。

## 两阶段训练约定

| 阶段 | 累计 rollout 轮数 | 初始化 / 续训 | 场景与奖励 |
|---|---:|---|---|
| Stage 1 | 0 → 200 | 从上述 SFT 导出初始化 DSRL 模块 | XY ±1 cm；成功奖励 1；关闭力奖励 |
| Stage 2 | 200 → 400 | 完整恢复本实验 `global_step_200` | XY ±1 cm；指定范围质量随机化；成功奖励 1 + normalized effort reward |

场景直接复用，不修改 JSON：

- [Stage 1 场景](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1/gentle_grasp.json)
- [Stage 2 场景](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2/gentle_grasp.json)

| 目标 | Stage 1 质量 kg | Stage 2 质量 kg | min_effort / max_effort |
|---|---:|---:|---:|
| Vitasoy / target_object_1 | 0.10 | 0.10–0.60 | 2 / 36 |
| Coca-Cola / target_object_2 | 0.20 | 0.15–0.25 | 4 / 12 |
| cookie / target_object_3 | 0.30 | 0.30–0.70 | 3 / 30 |

保留三个目标的原始无副词任务文本及多任务调权：EMA 0.9、权重范围 0.5–2.0、sigmoid scale 10。力奖励沿用 coefficient 10、至少 4 个有效样本、contact epsilon 1；使用当前字段 `max_effort` 表达旧 YAML 的 `mid_effort`，数值不变。当前默认归一化上下界为 [-1, 1]，乘系数后的力奖励范围为 [-10, 10]，`max_effort` 对应归一化奖励零点。

力统计遵循实际构建进 0.10.1 的代码。编写时工作区的 `tabero_force_reward.py` 使用 `abs(left Fz) + abs(right Fz)` 累积法向力，并保留原双指接触门限；与较早版本的 `2 * min(abs(left Fz), abs(right Fz))` 不同。因此这里保留的是场景、阈值和奖励参数，不宣称与历史镜像的力统计实现完全相同。脚本会冻结源码哈希，拒绝镜像与本地源码不一致的运行。

## RL 配置

- [stage1.yaml](../../RLinf/examples/embodiment/config/2026-11-10-tacimg-dsrl-training/stage1.yaml)
- [stage2.yaml](../../RLinf/examples/embodiment/config/2026-11-10-tacimg-dsrl-training/stage2.yaml)

| 参数 | 值 |
|---|---|
| 框架 / 入口 | RLinf 同步 embodied SAC/DSRL；`train_embodied_agent.py` |
| 环境 | RLinf 中的 `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`；gentle_grasp task 6 / Task820 三物体 |
| reset / chunk 边界 | `task_config_default_reset` / `terminal_safe_v1`；auto_reset=false、ignore_terminations=false |
| GPU | 单节点 8 卡；actor/rollout 0–7，env 0–5；通过 YAML placement 分配 |
| 环境数量 | train 192，三任务各 64；eval 配置 96，训练期间 eval 关闭 |
| 每轮 / episode 上限 | 300 环境步；rollout_epoch=1 |
| VLA horizon / 执行 chunk / 去噪步 | 50 / 10 / 10 |
| batch | 每 rank micro 48；8 rank 全局 384；每次更新梯度累积 1 |
| SAC 更新 | replay 就绪后每轮 8 次；actor/critic 更新比 1:1 |
| gamma / tau | 0.999 / 0.005；每次更新软更新 target |
| Actor optimizer | lr 1e-4，betas 0.9/0.96，eps 1e-8，clip 3.5，weight_decay 0 |
| Critic optimizer | lr 3e-4，betas 0.9/0.999，eps 1e-8，clip 10，weight_decay 0 |
| 熵温度 | softplus，初值 1，target entropy -16，lr 3e-4，clip 10 |
| Scheduler | 三者均 torch_constant；不新增 warmup |
| 网络 | 噪声维数 32；10 个 Q heads；hidden dims [128,128,128]；image/state latent 64 |
| 精度 | 沿用参考 `precision: null`、AMP 关闭；OpenPI loader 使用既有选定参数 BF16 转换；target shadow 为 FP32 |
| FSDP | no_shard，use_orig_params=true；actor 不 offload，rollout offload |
| gradient checkpointing / seed | disabled / 42 |
| 保存 / 上传 | 每 10 轮 / 每 100 轮 |

`max_steps/max_epochs=200/400` 表示**累计 rollout 轮数**，不是优化器步数。每阶段新增 200 轮，名义环境步数 11,520,000；replay warmup 会影响实际 SAC 更新次数。Stage 2 恢复优化器、调度器、熵温度、target、replay、RNG 和任务权重，不重置为新 run。

TacImg 必须使用当前 `compact_dsrl_ring`：

```yaml
algorithm:
  dsrl_observation_semantics: main_wrist_tactile_shared_encoder_concat_v1
  dsrl_replay_semantics: main_wrist_tactile_bf16_64_transition_ring_v1
  replay_buffer:
    backend: compact_dsrl_ring
    capacity_transitions: 80000
    checkpoint_shard_transitions: 4096
    max_resident_gib: 12.0
    min_buffer_size: 3
    enable_preload: false
    prefetch_size: 2
```

每 rank 的 CPU ring 存储三路 64×64 BF16 图像及 SAC transition，不存原始高分辨率图像或 marker-motion。多任务索引也随 replay 保存。`min_buffer_size: 3` 延续按 trajectory 插入次数计数的 warmup；`enable_preload: false` 禁止后台预取。它替换旧 trajectory buffer，不能声称 replay 容量和采样窗口与旧实现相同。

## 训练脚本

[run_stage12.sh](./run_stage12.sh) 在宿主机运行 Bash 编排，仅在指定镜像中执行检查和训练。镜像固定为 **0.10.1**；缺失时立即退出，不自动 build、pull 或替换版本。正式启动前按项目要求确认：容器执行、no-state、同步 RLinf，以及上述环境和训练参数。

```bash
# 当前可执行：不调用 Docker、不读取 .env、不写运行目录。
bash experiments/2026-11-10-tacimg-dsrl-training/run_stage12.sh --dry-run

# 以下命令需先确认 0.10.1 镜像构建与验证通过，再确认训练约定。
# 预检：校验镜像/本地源码、配置、模型哈希、norm stats、场景及 HF 身份；不训练、不上传。
bash experiments/2026-11-10-tacimg-dsrl-training/run_stage12.sh --check

# 在宿主机命名 tmux 会话中执行。
tmux new -s task820-tacimg-dsrl
bash experiments/2026-11-10-tacimg-dsrl-training/run_stage12.sh --run
```

脚本遵循 [docker/README.md](../../docker/README.md) 的 Compose 挂载与 `rlinf` 服务入口，从 `.env` 读取凭据。允许通过已有 `TABERO_MODELS_HOST`、`TABERO_RESULTS_HOST`、`TABERO_RECORD_HOST`、`TABERO_DATASETS_HOST` 指定挂载目录；不使用 `CUDA_VISIBLE_DEVICES`。镜像 ID、Compose、源码哈希、RL 配置和场景快照一经记录，后续运行必须一致。

Stage 1 → Stage 2 自动衔接；已完成阶段经复核后跳过。中断/失败会停止后续阶段并保留日志，重新执行不会擅自覆盖失败阶段或创建新的 W&B 身份；需要检查真实 checkpoint 后按同一训练约定恢复。脚本不包含额外仿真评估。

### 滚动保存、上传与完成判定

- 每 10 轮由 RLinf 保存完整 checkpoint，等待 runner 原子写入 `multi_task_state.json` 后才校验、上传或参与清理。
- 保留最近两个**校验通过**的恢复点和所有整百 checkpoint。只删除本实验中已验证、未受保护的旧目录；不删除不完整或校验失败的 checkpoint。
- Actor 使用参考配置的 `local_shard`，逐一检查 8 rank 的模型、actor/critic 优化器、调度器及 RNG；另检查 compact trainable sidecar、每 rank target/FP32 shadow、alpha DCP、compact replay 元数据及分片大小、SAC update step 和任务状态。结构校验不等同于真实分布式恢复成功。
- 每 100 轮后台上传完整 `global_step_N/`，目标为 `<HF_TOKEN owner>/2026-11-10-tacimg-dsrl-training` 的 public 模型仓库。容器入口读取挂载的 `.env`，上传前获取 HF 身份；不打印 token。
- 上传失败重试，训练可继续；阶段结束必须补齐本阶段两个整百 checkpoint。上传后核对指定远端 commit 的完整文件清单和大小，保存回执，后续凭回执及本地 fingerprint 避免重复上传。
- W&B 项目 `tabero-rlinf`、entity `183842220-hkust`、run name `2026-11-10-tacimg-dsrl-training`。run ID 首次运行生成并持久化，Stage 2 使用同一 ID 和 `WANDB_RESUME=must`；同时写 TensorBoard。
- 每 15 秒监测训练、GPU、磁盘、有限 SAC 指标和 checkpoint 保留状态；30 分钟无新 SAC step、指标异常或监测进程退出时停止本次训练容器。停止操作只针对本次命名容器和所属子进程。
- 预检要求容器 `/tmp` 至少 2 GiB、results 至少 256 GiB；运行中低于 1 GiB / 64 GiB 时停止。这是保护下限，不是完整训练的容量估算；需为多个完整 checkpoint 留出充足空间。
- 仅当训练进程成功、终点 checkpoint 完整、整百上传齐全，且 TensorBoard/W&B 在零基日志 step199/399 均有有限 actor/critic/alpha loss，才标记阶段完成。在线 W&B 审计失败会保留失败状态，不自动新建 run 或切换日志模式。

## 产物路径

路径均相对于仓库根目录或对应宿主机挂载根目录：

| 内容 | 路径 |
|---|---|
| 控制目录 / 快照 / 训练契约 | `results/2026-11-10-tacimg-dsrl-training/_launcher/` |
| 阶段命令、训练/监控/上传日志 | `_launcher/dsrl-stage1/`、`_launcher/dsrl-stage2/` |
| W&B ID / URL | `_launcher/dsrl.wandb_id`、`_launcher/dsrl.wandb_url.txt` |
| HF 回执 | `_launcher/hf/dsrl/global_step_N.json` |
| checkpoint | `results/2026-11-10-tacimg-dsrl-training/dsrl/2026-11-10-tacimg-dsrl-training/checkpoints/global_step_N/` |
| TensorBoard | `results/2026-11-10-tacimg-dsrl-training/dsrl/tensorboard/` |
| W&B 本地日志 | `results/2026-11-10-tacimg-dsrl-training/dsrl/wandb/` |
| 正式运行报告 | `Record/experiments/<完整时间>-task820-no_adverb-dsrl-tacimg_stage12.md` |
| 本次静态验证材料 | `results/2026-11-10-tacimg-dsrl-training/_validation/` |

运行报告由 `--run` 生成，记录真实启动命令、源码版本、参数、基模、各次尝试与故障、模型和日志路径、每阶段及总训练时长、唯一 W&B 链接与 HF 上传结果。当前没有训练结果或正式运行报告。

## 验证边界

已通过 Bash/内嵌 Python 语法检查、两阶段 Hydra 组合与插值解析、参考配置差异审查、基模文件哈希及归一化来源检查。临时合成 checkpoint / 模拟 HF 接口验证了滚动保留、上传回执、重复上传跳过、本地文件变化和远端大小不符时拒绝完成；模拟缺失镜像验证了仅检查 0.10.1、不 pull/build、不启动容器。验证材料为 [launcher-static-validation.json](../../results/2026-11-10-tacimg-dsrl-training/_validation/launcher-static-validation.json)，另有 resolved 配置、差异清单和 dry-run 输出。

文件准备阶段完成的是静态验证；随后 0.10.0 → 0.10.1 的镜像内容、依赖导入和配置验证单独记录在上述镜像记录中。完整 launcher `--check`、模型加载、8 GPU worker 初始化、训练、Stage 2 恢复、远端上传及策略效果仍需分别验证。宿主机 `RLinf/.venv` 缺少 PyTorch 等训练依赖，运行时检查使用指定镜像。
