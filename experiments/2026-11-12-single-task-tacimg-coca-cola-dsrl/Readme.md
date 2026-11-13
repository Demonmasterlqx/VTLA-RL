# 2026-11-12-single-task-tacimg-coca-cola-dsrl

使用 TacImage Pi0.5 SFT30k 基模进行 Coca-Cola 单任务 DSRL 两阶段训练。
实验条件参考 [可乐单任务 PiRL](../2026-11-11-single-task-tacimg-coca-cola-pirl/Readme.md) 的最新 **4×16 环境、六卡**版本；算法与脚本参考 [TacImage DSRL](../2026-11-10-tacimg-dsrl-training/Readme.md) 及其 [run_stage12.sh](../2026-11-10-tacimg-dsrl-training/run_stage12.sh)。

当前状态（2026-11-12）：已在 node01-10 实际尝试启动，尚无完整 rollout、SAC 更新或 checkpoint。六卡 CUDA、完整 CPU 预检、1,749 个源码文件一致性及基模哈希检查通过。第一次因 W&B 代理 403 退出；换线路后同一 W&B ID 正常上线，四个仿真 worker、actor/target 和同步初始化通过，但首轮 rollout 在物理 GPU6 显存不足：actor 约15.55GiB + rollout约7.98GiB，仅剩7.50MiB，申请242MiB失败。已释放训练 GPU。用户随后批准开启 offload，两阶段现均为 `actor.enable_offload: true`，其他训练参数保持原值；W&B ID 仍为 `5b076120`，恢复启动使用 `WANDB_STAGE1_RESUME=must`。当前节点上的其他实验占用原六卡资源，尚未启动本次恢复。详情见 [运行报告](../../Record/experiments/2026-11-12-06-15-22-task820-no_adverb-dsrl-single_task_tacimg_coca_cola-stage12.md) 和 [已应用的配置差异](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_node01_setup/proposed-offload/config.diff)。

本次用户选择六卡直接启动：物理 0/3/4/5 卡运行环境，6/7 卡运行 actor/rollout；0/3/4 与原有进程共置。显式使用 `GPU_COLOCATE_ENV_IDS=0,3,4`，仅允许指定环境卡共置，且每张需至少 16 GiB 空闲；actor 卡仍须空闲。未设置该参数时仍执行原独占检查。训练 YAML、batch、seed 和梯度检查点设置未改变。

节点使用专用 rootless Docker `node01-vtla-data`，镜像仍为 0.10.1。Docker 24 通过 `TABERO_GPU_RUNTIME=nvidia` 和显式 UUID 注入 GPU，专用客户端使用 Compose 2.35.1；`TABERO_MEMLOCK_LIMIT=50539450368` 对应该 daemon 的硬上限。模型与输出挂载到 `/home/lsf_node01/rashare/qixuan/vtla-dsrl/{models,results}`，避免节点上不可用的共享软链接。日志每 15 秒同步回本机相同 `results/` 实验目录，完整 checkpoint 另行同步。启动参数保存在 [_node01_setup/run-on-node01.sh](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_node01_setup/run-on-node01.sh)。

## 文件与基模

```text
experiments/2026-11-12-single-task-tacimg-coca-cola-dsrl/
  Readme.md
  run_stage12.sh

RLinf/examples/embodiment/config/2026-11-12-single-task-tacimg-coca-cola-dsrl/
  stage1.yaml
  stage2.yaml
```

基模与 PiRL 一致：

```text
models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000
```

使用其 `model.safetensors`、`config.json`、`training_contract.json`、`export_meta.json` 和 `assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json`。这是 SFT step30000 的 EMA、merged-LoRA 导出；Stage1 初始化新的 DSRL 模块，Stage2 只恢复本实验完整的 `global_step_200`。

## 保持一致的实验条件

| 项目 | 本实验约定 |
|---|---|
| 框架 / 镜像 | RLinf 同步训练，Docker `ccr.ccs.tencentyun.com/vtla/vtla:0.10.1` |
| 环境 | `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`，Task6 / gentle_grasp |
| 任务接口 | `algorithm.multi_task.enabled: true`，train/eval 列表仅 `coca_cola` |
| 目标 / 提示词 | `target_object_2`；`Pick up the Coca-Cola and put it into the basket`，无副词 |
| VLA 输入 | 主相机、腕部相机、TacImage 历史拼图、语言；no-state |
| 触觉图像 | 左右各8帧，4×4、224×224 RGB 历史拼图；共3路图像输入 |
| 动作 | horizon50，执行chunk10，13维动作，flow-SDE推理10步，noise level0.5 |
| 训练环境 | 总计64；四个环境worker，每worker16；`pipeline_stage_num: 1` |
| 单轮采样 | 最多300控制步；名义64×300=19,200环境步、1,920个chunk槽位 |
| Batch | micro8、global384、actor world size2；每个SAC更新累积24个microbatch |
| 周期评估 | `val_check_interval: -1`；eval配置保留64环境，但不启动周期评估 |
| Seed / 激活检查点 | 42 / `gradient_checkpointing: false`，disabled |
| 场景 | 原adaptive Stage1/2，XY±1cm；其余物体、篮子、摩擦、reset和成功容差保持原场景 |
| 成功判定 | 可乐成功分支，连续8步，成功奖励1 |
| 日志 | TensorBoard + W&B；跨阶段使用同一逻辑run的name/ID |

六张选定物理卡按PCI顺序映射为逻辑0–5：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0-3
    rollout: 4-5
    actor: 4-5
```

`GPU_IDS` 仅用于宿主机Docker设备选择。训练资源分配仍通过上述RLinf placement，脚本不设置 `CUDA_VISIBLE_DEVICES`。训练前保存物理index/UUID/PCI地址与逻辑编号，容器内校验CUDA恰好可见六卡且UUID顺序一致。普通Docker使用明确的GPU UUID列表；rootless Docker使用已有的NVIDIA CDI设备，不修改守护进程。

## 两阶段环境与奖励

| 阶段 | 累计rollout目标 | 可乐质量 | 奖励 |
|---|---:|---|---|
| Stage1 | 0→200 | 0.20 kg | 成功奖励1，关闭力奖励 |
| Stage2 | 200→400 | 0.15–0.25 kg | 成功奖励1 + normalized effort reward |

直接复用场景：

```text
Tabero_X/benchmarks/datasets/realworld/config/
  gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1
  gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2
```

Stage2保留 coefficient=10、min_effort=4、max_effort=12、截断[-1,1]、min_valid_samples=4、contact_epsilon=1；`force_bonus.enabled: false`。力定义为 `abs(Lz)+abs(Rz)`，保留paired-contact有效性门控。归一化分量为 `10 * clip((12 - mean_force) / 8, -1, 1)`，仅按原实现的有效性与终止条件触发。

200/400表示累计rollout轮数，每阶段新增200轮；整个实验名义采样7,680,000环境步。终止mask、replay warmup会影响有效transition及优化器更新次数，不将其当作400个优化器step。

## DSRL算法配置及与PiRL的差异

采用现有 `embodied_sac` DSRL实现：冻结VLA，训练DSRL噪声策略、actor图像编码器、critic图像/状态编码器、Q heads及熵温度；不训练新的VLA LoRA或PPO value head。

| 项目 | DSRL设置 |
|---|---|
| SAC | gamma0.999、tau0.005、`update_epoch: 8`、critic/actor更新比1 |
| Q聚合 / target | mean；10个Q heads；target每次更新，FP32 shadow |
| 熵项 | `backup_entropy: false`；softplus alpha，初值1，target entropy=-16 |
| Actor优化器 | lr1e-4，betas(0.9,0.96)，eps1e-8，clip3.5，weight_decay0 |
| Critic优化器 | lr3e-4，betas(0.9,0.999)，eps1e-8，clip10，weight_decay0 |
| Alpha优化器 | lr3e-4，clip10；其余沿用既有DSRL实现 |
| 调度 | 三者均 `torch_constant`，不新增warmup |
| 网络 | noise dim32，image/state latent64，hidden dims[128,128,128] |
| FSDP / checkpoint | `no_shard`、`use_orig_params: true`、`local_shard`，完整保存两个actor rank |
| 精度 / offload | `precision: null`、AMP关闭、既有OpenPI选定参数BF16转换；actor与rollout的`enable_offload`均为true（actor开关由用户在首轮OOM后批准开启，target model仍驻留GPU） |
| 采样与同步 | 收集SAC transitions；只同步DSRL噪声策略和actor图像编码器；关闭bootstrap overlap |

上述损失、可训练模块、优化器、精度、FSDP和恢复格式属于算法实现差异；任务、场景、奖励、基模、采样规模与六卡布局沿用单任务PiRL。保留DSRL的精度/优化器路径，不把PiRL的LoRA/value-head更新直接改名为DSRL。

`discrete_state_input: false`、`dsrl_actor_use_state: false`；critic仍使用既有7维EEF/gripper状态，动作转换也保留该信息。`dsrl_use_tactile: false`关闭独立marker-motion编码器，**不关闭第三路TacImage**。六维测得力不输入VLA/DSRL actor，环境可用它计算奖励；13维动作仍包含六维力输出。

`tactile_streams: []`，触觉prefix参数置空；`tactile_loss_weight: 0.05`保留SFT力损失来源。DSRL使用SAC损失，本次不新增SFT力监督。

Replay采用：

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

每个actor rank保留80,000个transition上限及12GiB CPU驻留上限；本实验为两个rank。三图以64×64 BF16形式存储。warmup阈值3沿用trajectory插入次数语义，关闭后台预取以支持恢复。Stage2恢复优化器、调度器、target/FP32 shadow、alpha、replay、SAC更新计数、各类RNG及单任务控制状态。

## 启动脚本

按本次要求，将可执行脚本放在本实验 `experiments/` 目录；运行生成的检查器、快照和验证材料放在对应 `results/`。

```bash
# 无Docker、无凭据读取、无运行目录写入。
bash experiments/2026-11-12-single-task-tacimg-coca-cola-dsrl/run_stage12.sh --dry-run

# CPU预检：镜像/源码、Hydra、基模/norm来源、场景、磁盘、HF身份。
# 不初始化模型、仿真、训练或W&B run，不上传。
bash experiments/2026-11-12-single-task-tacimg-coca-cola-dsrl/run_stage12.sh --check

# 正式训练入口示例：实际使用前确认六张卡空闲及训练约定。
tmux new -s cola-tacimg-dsrl
GPU_IDS=0,1,2,3,4,5 \
  bash experiments/2026-11-12-single-task-tacimg-coca-cola-dsrl/run_stage12.sh --run
```

`--run`要求宿主机命名tmux。脚本使用本地0.10.1镜像，缺失时退出，不自动构建、拉取或换版本；源码不一致时停止并报告。检查、上传、维护容器均不申请GPU，训练容器只暴露选定六卡。GPU选择要求PCI顺序、无现存计算进程且每卡显存使用小于1GiB；不停止其他任务。

沿用参考脚本的生命周期：

- 每10轮保存完整checkpoint。逐一审计两个actor rank的模型、两优化器/调度器、RNG、target、alpha、compact replay及任务状态；完整性审计不等于实际分布式恢复成功。
- 保留最近两个完整且验证通过的恢复点，以及所有整百checkpoint；不删除不完整、校验失败或其他实验的目录。
- 每100轮上传整个可续训checkpoint到 `<HF_TOKEN owner>/2026-11-12-single-task-tacimg-coca-cola-dsrl` public模型仓库。只在正式 `--run`执行，先source `.env`，不打印凭据；核对远端commit的文件清单及大小，记录回执。
- W&B project=`tabero-rlinf`、entity=`183842220-hkust`、name=`2026-11-12-single-task-tacimg-coca-cola-dsrl`。首次正式启动生成并保存run ID，Stage2使用同一ID及`WANDB_RESUME=must`。沿用DSRL参考脚本的online模式和最终在线审计，不自动切换日志模式。
- 每15秒监督进程、GPU、磁盘、有限SAC指标及checkpoint。失败停止后续阶段；重试不会自动覆盖失败阶段或新建W&B身份。
- CPU预检要求容器`/tmp`至少2GiB、results至少256GiB；运行中低于1GiB/64GiB即停止。磁盘不足时不绕过检查。
- 仅当训练成功、终点checkpoint完整、阶段整百上传齐全，且TB/W&B零基step199/399具有有限actor/critic/alpha loss时标记完成。

## 输出与记录

以下路径相对仓库根目录：

| 内容 | 路径 |
|---|---|
| 启动器快照/配置/场景/GPU映射 | `results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_launcher/` |
| 阶段命令与日志 | 上述目录内`dsrl-stage1/`、`dsrl-stage2/` |
| checkpoint | `results/2026-11-12-single-task-tacimg-coca-cola-dsrl/dsrl/2026-11-12-single-task-tacimg-coca-cola-dsrl/checkpoints/global_step_N/` |
| TensorBoard / W&B本地数据 | `results/2026-11-12-single-task-tacimg-coca-cola-dsrl/dsrl/{tensorboard,wandb}/` |
| W&B身份 / HF回执 | `_launcher/dsrl.wandb_id`、`_launcher/dsrl.wandb_url.txt`、`_launcher/hf/dsrl/` |
| 正式运行报告 | `Record/experiments/<完整时间>-task820-no_adverb-dsrl-single_task_tacimg_coca_cola-stage12.md` |
| 本轮验证材料 | `results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_validation/` |

正式报告由`--run`生成，包含真实参数、命令、GPU映射、模型/输出位置、源码、阶段事件、总时长以及唯一W&B链接。当前仅准备文件，不生成虚构的训练结果或训练完成报告。

## 验证边界

2026-11-12完成的验证：

- Bash及两个内嵌Python块语法通过；`--dry-run`没有调用Docker或创建运行目录。
- 两阶段Hydra组合/插值通过。环境配置与PiRL单任务版本逐项完全一致；与DSRL参考的差异仅为单任务环境、布局、micro batch和实验标识/路径。
- Docker 0.10.1的CPU检查中，1,749个相关源码文件与本地一致；现有DSRL输入/动作契约、SAC任务、rollout同步及任务力奖励校验全部通过。镜像ID为`sha256:337be55a06a5da4715ef5312650a13584a2e484c61512016ad37541647ca2e7d`。
- 使用合成设备信息验证普通Docker六卡选择及rootless CDI的Compose合并、重复/占用/不存在/顺序错误/映射改变时拒绝运行，以及容器CUDA数量和UUID顺序检查。未实际申请GPU。
- 使用临时checkpoint文件验证保留最近两个、整百点、损坏点、未完成点及未来阶段目录；只删除允许清理的旧合成恢复点。
- 缺失镜像和缺少tmux时正确停止；未pull/build镜像、未读取上传凭据。

汇总：[validation-summary.json](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_validation/validation-summary.json)；细节：[宿主机配置审计](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_validation/host-static-validation.json)、[镜像内CPU校验](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_validation/container-static-validation.json)、[脚本模拟检查](../../results/2026-11-12-single-task-tacimg-coca-cola-dsrl/_validation/launcher-behavior-validation.json)。

未执行完整`--check`（包含基模权重哈希、磁盘门禁及HF身份访问），未初始化模型、仿真或训练worker。运行时尚需验证六卡初始化、完整rollout、有限SAC更新、实际Stage2恢复、checkpoint上传和训练稳定性。本脚本不自动启动额外的九条件评测，也不以静态检查代替策略质量评估。

交付时的本机状态：上述镜像内CPU校验成功结束后，收尾查询发现`docker.service`为`inactive`，Docker socket无法连接。本次没有操作Docker服务。实际执行`--check`或`--run`前需恢复目标主机的Docker服务，并重新通过磁盘等预检；这不撤销已经保存的CPU校验记录。
