# 2026-11-11-single-task-tacimg-coca-cola-pirl

状态：用户已授权在DOIT完成正式训练，2026-11-11 21:25:45 CST已启动0→200→400轮。Docker 0.10.1，四张环境卡各16环境+两张actor/rollout卡；W&B ID `6eb6db01`。当前进度见`results/2026-11-11-single-task-tacimg-coca-cola-pirl/train/supervision-state.json`。下方5轮smoke已通过，正式训练仍在执行。

目标：使用 TacImage Pi0.5 基模训练只包含 Coca-Cola 的 PiRL。保留多任务接口，但任务列表仅有 `coca_cola`。总计使用六张 GPU，64 个训练环境分布在四张环境卡，每卡16个；另外两卡承载 rollout 和 actor。

## 参考与已检查信息

- 环境、奖励、PiRL算法和两阶段目标参考：[2026-10-22 adaptive compare](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md) 及其 RLinf PiRL stage1/stage2 YAML。
- TacImage模型、数据变换和归一化参考：`RLinf/examples/embodiment/config/2026-10-29-melt/tacimg/stage1.yaml`、`stage2.yaml`。只采用其中的TacImage适配，不继承三任务/八卡布局。
- 基模：`models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000`。
- 归一化：基模目录下 `assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json`。本机已确认存在，DOIT已确认基模目录存在；实际启动前继续校验权重、归一化哈希和严格加载。
- 2026-11-11只读检查：DOIT为8张RTX 4090（24 GB），当时GPU 2/4/5/6/7空闲，0/1/3被其他任务占用。正式选择前重新检查，不预占或停止其他任务。
- 初次盘点DOIT只有0.9.6/0.10.0；执行时已补齐并核验0.10.1，1,749个相关源码文件与本机一致。

## 本轮执行选择

1. Docker 0.10.1，镜像ID `sha256:337be55a06a5da4715ef5312650a13584a2e484c61512016ad37541647ca2e7d`。本机与DOIT镜像、1,749个训练相关源码文件、基模权重/config/norm SHA-256一致。DOIT使用rootless Docker CDI挂载选定GPU，不加载DOIT宿主机旧源码。
2. rollout和actor共同使用两卡，各两个worker，沿用offload。本次DOIT物理卡：env=0,2,3,4；actor/rollout=5,6；映射到容器逻辑0–5。旧的2×32试启动使用env=2,4和actor/rollout=5,6。
3. 使用当前 `abs(Lz) + abs(Rz)` 控制器/奖励定义，保留参考奖励参数及paired-contact有效性门控。历史0.9.4为 `2 * min(abs(Lz), abs(Rz))`，本次不宣称数值上完全复现历史控制器。

`mid_effort`在当前代码中仍有兼容映射，会警告并作为`max_effort`使用；新配置使用`max_effort`，保持可乐的4/12边界。该字段改名本身不改变归一化曲线。

## 已创建文件

独立RLinf配置目录：

```text
RLinf/examples/embodiment/config/2026-11-11-single-task-tacimg-coca-cola-pirl/
  stage1.yaml
  stage2.yaml
```

遵循根目录AGENTS.md，实验专用启动和检查脚本放到结果目录：

```text
results/2026-11-11-single-task-tacimg-coca-cola-pirl/
  run_stage12.sh
  _tools/
  _launcher/
  smoke/<attempt>/
  train/
```

脚本参考既有host Bash → Docker流程，提供 `--dry-run`、`--check`、`--smoke`、`--run`，使用当前工作区解析ROOT；正式训练和试启动使用不同输出目录及W&B身份。

## 执行训练约定

| 项目 | 计划值 |
|---|---|
| 方法 | RLinf PiRL/PPO，FSDP，terminal-safe OpenPI同步on-policy训练 |
| 任务接口 | `algorithm.multi_task.enabled: true`；train/eval任务列表仅`coca_cola` |
| 目标 | `target_object_2`，`Pick up the Coca-Cola and put it into the basket`，无副词 |
| 环境 | `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`，Task6/gentle_grasp |
| 观测 | 主相机、腕部相机、8帧TacImage历史拼图和语言；no-state；无TacField/数值测量力token；保留EEF/gripper用于动作转换 |
| 模型配置 | `pi05_lora_tacimg_realworld_replayed_task820_force`，3图输入，`tactile_streams: []` |
| 动作 | action horizon 50，执行chunk 10，环境动作13维，flow-SDE推理10步、noise level 0.5 |
| 训练参数 | action-expert LoRA rank32及value head可训练，其余冻结 |
| 精度 | 冻结参数BF16、训练参数FP32、BF16 autocast |
| 训练环境 | 总计64，四个env worker各16；`pipeline_stage_num: 1` |
| 周期评估 | 关闭，`val_check_interval: -1`，不额外常驻eval环境 |
| 每轮采样 | 300控制步，64×300=19,200名义环境步，64×30=1,920个chunk槽位 |
| 批量 | micro batch=8、global batch=384；与参考micro=48不同，用梯度累积适配24GB卡 |
| 梯度累积 | 两actor时384/(8×2)=24；一actor时384/(8×1)=48 |
| 更新次数 | `update_epoch: 1`；每轮名义1920/384=5个优化器batch；terminal mask按原实现处理 |
| 优化器 | Adam，lr=1e-6、value_lr=1e-4、betas=(0.9,0.95)、eps=1e-8、weight_decay=0.01、clip_grad=1.0 |
| 调度 | constant；lr/critic warmup均0；seed42 |
| PPO | gamma0.99、GAE lambda0.95、clip上下限0.2、value_clip0.2、KL beta0.05、entropy bonus0，其他沿用参考 |
| 激活检查点 | `gradient_checkpointing: false`，明确disabled |
| 显存策略 | 沿用actor/rollout enable_offload；不使用CUDA_VISIBLE_DEVICES代替RLinf placement |
| 保存 | 每10轮保存完整可续训checkpoint，保留DCP、优化器、调度器、RNG和multi_task状态 |
| 导出 | 最终step400导出推理bundle；生产输出与试启动输出隔离 |
| 日志 | TensorBoard与W&B，项目`tabero-rlinf`，每个逻辑正式训练固定一个run ID |

四个环境worker、两个模型worker共置拓扑（六张物理卡映射到容器逻辑卡后）：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0-3
    rollout: 4-5
    actor: 4-5
```

启动时记录物理GPU UUID/序号到容器逻辑卡的对应，并确认容器只看到这六张卡。

单任务保留原权重控制器；实际actor会对任务权重归一化，只有可乐时归一化权重为1。检查任务日志、轨迹task index、成功判定和reward全部路由到可乐。环境中原有其他物体、篮子、位置/质量随机化、摩擦和成功容差保留；RLinf现有Task6适配器根据`target_object_2`选择可乐成功分支。

## 两阶段环境及奖励

| 阶段 | 累计训练目标 | 可乐质量 | 场景 | 奖励 |
|---|---:|---|---|---|
| Stage1 | 200 rollout轮 | 0.20 kg | 原adaptive stage1，XY±1cm | 成功奖励1，关闭力奖励 |
| Stage2 | 400 rollout轮 | 0.15–0.25 kg | 原adaptive stage2，XY±1cm | 成功奖励1，启用normalized effort reward |

直接复用以下场景，不修改IsaacLab：

```text
Tabero_X/benchmarks/datasets/realworld/config/
  gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1
  gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2
```

Stage2力奖励保留：coefficient=10，min_effort=4，max_effort=12，奖励截断[-1,1]，min_valid_samples=4，contact_epsilon=1；成功连续8步；`force_bonus.enabled: false`。有效轨迹力使用上述左右法向力绝对值之和，归一化分量为 `10 * clip((12 - mean_force) / (12 - 4), -1, 1)`，有效性门控和终止奖励触发保留原实现。

Stage1从TacImage SFT30k全新初始化；Stage2只从本实验完整`global_step_200`续训。旧实验的TacField/三任务checkpoint不可充当本次单任务TacImage的Stage1。200/400表示累计rollout轮数，不是优化器step。完整400轮名义采样7,680,000环境步；这保持了原三任务实验中可乐每轮64环境的采样规模。

## 脚本与DOIT试启动流程

1. 创建两个独立YAML，解析Hydra，检查task列表、奖励和场景差异白名单、batch整除、4-env-worker路由及模型适配。使用现有验证器，不新增依赖本地模型/数据的测试。
2. 完成host Bash脚本：冻结配置/源码/基模哈希、显式GPU映射、tmux、唯一run ID、stage顺序、完整checkpoint门禁、日志和失败停止。公共HF发布不在本次试启动范围内。
3. 在DOIT `/data/home/sim6g/code/VTLA-RL` 下检查空闲卡、CPU/RAM、共享内存、磁盘、Docker、权重及USD；按最终镜像选择拉取或传输镜像，执行完整源码一致性核对。有差异时先报告，不自动使用不一致代码。
4. 使用独立smoke配置和目录、完整64环境及六卡拓扑，Stage1跑2个rollout轮并保存完整checkpoint；从该checkpoint切换Stage2奖励/场景再跑2轮，验证跨阶段恢复与实际force reward；再次恢复Stage2跑1轮，验证恢复后的步数和日志连续性。smoke累计5轮，阶段目标与目录显式标为smoke，不作为正式step200/400。
5. 每5–30秒检查owned进程、GPU归属、实际环境数、模型观测、有限loss/reward、参数更新、任务路由、显存峰值和耗时；检查checkpoint和TB/W&B写入。OOM时保留日志，停止本次试启动，不静默减少环境、global batch或改变gradient checkpointing。
6. 试启动结束释放本次资源，将日志、配置、资源曲线和checkpoint核验记录同步回本机，写入`Record/experiments/`，给出可行性和正式训练耗时估计。正式0→400脚本准备好，但试启动成功不自动代表正式训练完成。

正式W&B名称：`2026-11-11-single-task-tacimg-coca-cola-pirl`，随机run ID首次正式启动时生成并持久化；Stage2与同配置恢复保持相同name/ID。Smoke使用单独的`...-smoke`身份。正式TensorBoard/W&B/checkpoint置于`results/2026-11-11-single-task-tacimg-coca-cola-pirl/train/`下，实际完整路径和run链接写入Record。

## 验收边界

本次请求的交付是：独立配置、训练脚本、DOIT真实小规模步数试启动以及审计记录。试启动使用完整64环境，只缩短轮数；它验证初始化、采样、优化、两阶段奖励切换与checkpoint恢复，不证明400轮训练稳定性或策略质量。

## 启动入口

从VTLA-RL工作区根目录运行。`--dry-run`无副作用，`--check`只做CPU预检，训练必须在命名tmux中。首次生成的完整Hydra配置和环境快照位于结果目录的`_launcher/config`、`_launcher/env`。

```bash
bash results/2026-11-11-single-task-tacimg-coca-cola-pirl/run_stage12.sh --dry-run
bash results/2026-11-11-single-task-tacimg-coca-cola-pirl/run_stage12.sh --check
# 已执行的短程启动；失败的启动尝试及W&B ID保持在同一smoke目录。
GPU_IDS=2,4,5,6 SMOKE_ID=20261111-190300 WANDB_RUN_MODE=offline \
  bash results/2026-11-11-single-task-tacimg-coca-cola-pirl/run_stage12.sh --smoke
```

DOIT启用rootless Docker，因此启动脚本通过现有CDI挂载选定GPU；普通Docker使用`--gpus`。不修改Docker守护进程，不手工设置`CUDA_VISIBLE_DEVICES`。W&B在线接口403后，使用离线W&B和TensorBoard，保持ID `9bc5f511`，本轮尚未上传云端。

原2×32试启动Record：`Record/experiments/20261111-190300-task820-no_adverb-pirl-single_task_tacimg_coca_cola-smoke.md`。正式`--run`入口已实现200→400及最终bundle导出，尚未执行。

## 试启动结果（2026-11-11）

- 已验证：Docker 0.10.1源码/基模一致、四卡映射、两个worker各32环境、actor/rollout权重加载、LoRA/value head精度与冻结范围、禁用激活检查点。
- 阻塞点：首个环境步更新触觉RGB时，`gpu_taxim/sim/taxim_torch.py:276`分配临时张量OOM；两worker分别申请104MiB/616MiB失败。其后的相机weak-reference异常是清理阶段二次错误。
- 采样峰值：环境GPU2/4为23,977/23,945MiB，模型GPU5/6为15,200/15,298MiB。环境数、场景、图像分辨率、global batch和激活检查点均未因OOM而改变。
- 未验证：完整rollout、反向传播/参数更新、checkpoint、Stage2恢复与力奖励、最终bundle导出。当前完成轮数0，不能据此估算400轮训练时间。
- W&B离线ID仍为`9bc5f511`；TensorBoard与W&B初始化文件已保存，尚无训练标量，云端未同步。
- 19:40:17 CST清理结束，四卡显存回落至22/23/23/24MiB。

本机结果：`results/2026-11-11-single-task-tacimg-coca-cola-pirl/smoke_summary.json`；原始错误在`smoke/20261111-190300/diagnostics/`，各次尝试、GPU曲线、IsaacSim/Ray日志均保留在该smoke目录。完整记录见[Record](../../Record/experiments/20261111-190300-task820-no_adverb-pirl-single_task_tacimg_coca_cola-smoke.md)。

最终启动脚本补充了环境worker致命错误检测和容器日志保留；各次已执行脚本另存于对应attempt。下一次完整64环境试启动前，需要处理环境侧Taxim峰值显存或使用更大显存GPU。当前没有修改IsaacLab或训练框架源代码。

## 4×16环境重试（用户更新配置）

保持64环境、micro/global=8/384、两actor、累积24、模型/任务/奖励/精度/seed不变；仅将环境worker由2增加到4，每卡环境由32降到16，模型worker映射到逻辑4–5。激活检查点仍为disabled。由于分布式拓扑改变，使用独立的`env4x16-smoke` W&B名称和新输出目录，旧2×32日志与配置不覆盖。验证范围仍为Stage1两轮、Stage2恢复两轮、再次恢复一轮。

## 4×16试启动结果

- 运行：`smoke/20261111-201500-env4x16`；20:15:52–21:07:48 CST，总耗时51分55.5秒，退出码0。
- Stage1 0→2、切换Stage2恢复2→4、再次恢复4→5全部完成；完整checkpoint1–5保留，step2/4/5均通过DCP、优化器、调度器、RNG、任务状态及有限权重审计。
- TensorBoard/W&B步数0–4完全一致，146类标量无NaN/Inf；每对相邻checkpoint的260个可训练张量均改变（其中252个LoRA张量）。
- Stage2三轮力奖励轨迹数为71/73/75，缺失均为0；force_bonus_mean分别2.3530/2.4867/2.4520，属于奖励统计，不是牛顿力指标。
- 10秒采样显存峰值（MiB）：env GPU0/2/3/4 = 16086/15818/15722/15809；actor/rollout GPU5/6 = 15764/15862。未出现OOM；结束后六卡释放，其他任务未改动。
- 采样约258–265秒/轮、actor更新约88–90秒/轮；smoke每轮保存时总轮耗时约378–389秒。正式每10轮保存，短程耗时只能作为估算依据，不代表400轮稳定性验证。
- W&B固定名称`2026-11-11-single-task-tacimg-coca-cola-pirl-env4x16-smoke`，ID `e7919a98`，离线记录完整；未上传云端。
- 本机证据：`results/2026-11-11-single-task-tacimg-coca-cola-pirl/smoke/20261111-201500-env4x16/summary.json`；最终同步核验见同目录`local-transfer-audit.json`。
- Record：`Record/experiments/20261111-201500-task820-no_adverb-pirl-single_task_tacimg_coca_cola-env4x16-smoke.md`。

当前启动配置使用六卡；原2×32失败记录保留。已完成smoke目录不能覆盖重跑。正式启动入口（需在命名tmux中）为：

```bash
GPU_IDS=0,2,3,4,5,6 WANDB_RUN_MODE=offline \
  bash results/2026-11-11-single-task-tacimg-coca-cola-pirl/run_stage12.sh --run
```

本次没有执行正式`--run`，也没有导出正式step400模型。

## 正式训练（2026-11-11 21:25:45启动）

用户目标：在DOIT完成训练。沿用上述已验证配置，Stage1从SFT基模重新开始，至200轮后完整恢复到Stage2累计400轮；每10轮保存、保留全部checkpoint，最终导出merged模型与bundle。当前未宣称正式训练完成。

- DOIT tmux：`cola-pirl-formal-400`；本机同步tmux：`cola-pirl-formal-mirror`。
- 输出：`results/2026-11-11-single-task-tacimg-coca-cola-pirl/train/`。
- W&B名称：`2026-11-11-single-task-tacimg-coca-cola-pirl`；ID `6eb6db01`，离线+TensorBoard，跨阶段保持同一身份。
- 物理卡：env=0,2,3,4，actor/rollout=5,6；micro/global=8/384，gradient_checkpointing=disabled。
- 15秒监督、10秒显存遥测；本机每5分钟同步日志，完整checkpoint写完后同步。
- 按smoke粗估40–43小时，正式耗时以实际指标更新。
- 记录：`Record/experiments/20261111-212500-task820-no_adverb-pirl-single_task_tacimg_coca_cola-train.md`。


## 2026-11-12 用户停止训练并发布step190

用户要求停止；16:19 CST已停止全部本次训练进程并释放6张训练卡，监督goal已paused。正式运行18小时53分20秒，完成Stage1的190轮；未进入Stage2，未达到400轮目标。最新完整checkpoint为global_step_190，全部10–190保存点和日志保留本机。

step190完整checkpoint审计和DOIT/本机逐文件SHA256一致；无副词/no-state TacImage配置不变。merged模型及bundle已导出；T2-VLA消费者在匹配PEFT0.21.0的隔离环境中通过LOAD_OK与重建张量哈希检查。发布包为results/2026-11-11-single-task-tacimg-coca-cola-pirl/train/publish/global_step_190/。

用户指定使用LQX_HF_TOKEN账号Demomasterlqx创建公开仓库，上传与最终校验状态见[正式训练Record](../../Record/experiments/20261111-212500-task820-no_adverb-pirl-single_task_tacimg_coca_cola-train.md)。已记录导出输入task_id元数据兼容处理与PEFT版本要求。

公开发布已完成（2026-11-12 17:10 CST）：[Demomasterlqx/2026-11-11-single-task-tacimg-coca-cola-pirl](https://huggingface.co/Demomasterlqx/2026-11-11-single-task-tacimg-coca-cola-pirl)，commit `c461a13a6082dfc42724e1ce812e29758fa99703`；29个文件的远端大小/哈希全部通过，匿名访问已确认。
