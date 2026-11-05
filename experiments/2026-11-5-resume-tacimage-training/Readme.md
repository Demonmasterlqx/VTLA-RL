# 2026-11-5-resume-tacimage-training

继续 [2026-10-29-melt](../2026-10-29-melt/Readme.md) 的 **TacImage PiRL**：从原训练机器的完整 `global_step_400` 恢复，训练到**累计 800 rollout 轮**，即再训练 400 轮。保持原 stage2 仿真环境、质量随机化、奖励和训练参数。这里的 step 是 RLinf runner 的 rollout 轮数，不是 optimizer update 次数。

本次已完成训练配置、宿主机 Bash 启动脚本和静态/模拟检查。镜像采用 **0.9.7 → 0.9.8** 增量构建，镜像 ID、构建检查和远端发布状态见 [release-validation.json](../../results/2026-11-5-resume-tacimage-training/image/release-validation.json)。Git 提交与远端分支核验记录保存在同目录的 `git-push-validation.json`。未启动真实续训、仿真、W&B run 或 HF 上传。

## 续训契约

| 项目 | 设置 |
| --- | --- |
| 方法 / 入口 | PiRL PPO；`train_embodied_agent.py`，同步 RLinf |
| 输入 | no-state：不输入机械臂关节角度；主相机、腕相机、224×224 双指 8 帧触觉历史拼图 |
| state / 力 | `discrete_state_input: false`；保留 EEF/夹爪 proprio 用于归一化和动作转换；`tactile_streams: []`，不建立 TacField encoder；13D 动作包含 6D 目标力 |
| 原基模 | `models/task820_firm_mixed_pi05_tacimg_nostate_force005_sft_2a800_30k_20261017-221823/step_30000` |
| 归一化 | 基模的 `assets/pi05_horizon50_tacimg_task820_firm_mixed/norm_stats.json` |
| 恢复点 / 终点 | 完整 `global_step_400` → `global_step_800`；`max_epochs = max_steps = 800` |
| 仿真环境 | RLinf IsaacLab realworld Tabero，`gentle_grasp` task 6；沿用 melt stage2 |
| 任务 | Vitasoy / Coca-Cola / cookie，原无副词指令和 target_object_1/2/3 |
| 随机化 | X/Y ±1 cm；沿用 stage2 自适应质量范围：Vitasoy 0.10–0.60 kg、Coca-Cola 0.15–0.25 kg、cookie 0.30–0.70 kg |
| 奖励 | success 1 + normalized effort ×10；三物体 min/max effort 分别 2/36、4/12、3/30；reward bounds −1/1；旧 force_bonus 关闭 |
| GPU / 环境数 | 单机 8 GPU；actor/rollout 0–7，env 0–5；train 192、eval 96；每轮/episode 最多 300 环境步 |
| Batch / seed | micro 48，global 384，actor world size 8；每个 global minibatch 1 次累积；seed 42 |
| 模型 / 训练模块 | Pi0.5，horizon 50，执行 chunk 10，flow SDE 10 步；action-expert LoRA rank 32 + value head，VLM 和非 LoRA 基模参数冻结 |
| 优化器 | AdamW；actor/value LR 1e-6/1e-4；betas 0.9/0.95，eps 1e-8，weight decay 0.01，clip grad 1.0 |
| 调度 / 精度 | constant，warmup 0；冻结参数 bf16、训练参数 fp32、bf16 AMP；gradient checkpointing **disabled** |
| PPO | gamma 0.99，GAE lambda 0.95，update_epoch 1；其余全部沿用原 stage2 YAML |
| 保存 / 评估 | 每 10 轮保存完整 DCP + trainable sidecar；每 100 轮上传；训练期间 eval 关闭，不自动启动训练后仿真评估 |
| 日志身份 | 原 project `tabero-rlinf`、entity `183842220-hkust`、run name `2026-10-29-melt-tacimg-pirl`；TensorBoard + W&B |

配置：[tacimg/resume800.yaml](../../RLinf/examples/embodiment/config/2026-11-5-resume-tacimage-training/tacimg/resume800.yaml)。完整复制原 [tacimg/stage2.yaml](../../RLinf/examples/embodiment/config/2026-10-29-melt/tacimg/stage2.yaml)，解析后的差异仅为两个累计上限 `400 → 800`，以及 `resume_dir` 的 `global_step_200 → global_step_400`。

场景直接使用原 [stage2](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2/gentle_grasp.json)，不新增 stage3，也不修改 IsaacLab、Tabero_X 源码、USD 或依赖。脚本保存该目录的快照，实际训练读取快照，避免运行中场景被改动。

## 原机器上的恢复点与身份

用户确认原实验在另一台机器完成。默认从参考实验的原输出树恢复，保留该输出树和单一 W&B 身份：

```text
results/2026-10-29-melt/
├── _launcher/tacimg.wandb_id
└── tacimg/
    ├── tensorboard/
    ├── wandb/
    └── 2026-10-29-melt-tacimg-pirl/checkpoints/global_step_400/
        ├── actor/dcp_checkpoint/.metadata
        ├── actor/dcp_checkpoint/*.distcp
        ├── actor/model_state_dict/trainable_weights.pt
        └── multi_task_state.json
```

必须保留 DCP 模型、优化器、scheduler、RNG，以及多任务控制器状态。脚本检查分片索引、文件大小、步数、终点标记、world size、模态和归一化身份。检查通过只证明保存产物结构满足要求；真正的分布式恢复要在原机器启动 actor 后验证，仿真现场状态也不保证逐轨迹连续。

当前开发机器下载的 `models/2026-10-29-melt-tacimg-pirl/global_step_400/` 来自 [XiangXiao128 的 step400](https://huggingface.co/XiangXiao128/2026-10-29-melt-tacimg-pirl/tree/2f4cf842b692724e114e880c1441d59695f9207a/global_step_400)，其下载清单注明 `optimizer_or_dcp_downloaded: false`。它只有用于导出/推理的 sidecar 和控制器状态，**不能直接用于本次完整续训**；本轮没有下载大体积 DCP，也没有验证另一台机器上的实际 checkpoint。

优先在原训练机器执行，或将完整原输出树连同 `_launcher/tacimg.wandb_id`、TensorBoard/W&B 日志迁移到新的 results 挂载目录。脚本自动读取原 ID；仅在原 ID 文件不在当前挂载时，用 `TACIMG_WANDB_RUN_ID` 显式传入同一个 ID。没有 ID、与原文件冲突、远端 run 名不匹配或原 run 仍在运行时停止，不生成替代 run。

## 训练脚本

[run-resume-tacimage.sh](run-resume-tacimage.sh) 基于 melt 启动器实现，按本次要求放在本实验的 `experiments` 目录；旧 `run-melt.sh` 保持不变。脚本从自身位置定位工作区，支持用 `VTLA_WORKSPACE` 覆盖。请在**宿主机**运行。脚本默认只展示计划：

```bash
bash experiments/2026-11-5-resume-tacimage-training/run-resume-tacimage.sh --dry-run

# 在原训练机器配置现有目录；results 应包含上面的原 melt 输出树。
# 默认使用当前仓库的 models、datasets、results、Record 和 .env。
# 如路径不同，先 export TABERO_RESULTS_HOST=/absolute/path/to/results 等挂载变量。
bash experiments/2026-11-5-resume-tacimage-training/run-resume-tacimage.sh --check

# 确认使用 container、上述 no-state / 同步 RLinf / 原 stage2 环境和训练契约后：
tmux new -s task820-resume-tacimage
bash experiments/2026-11-5-resume-tacimage-training/run-resume-tacimage.sh --run
```

`--check` 检查本地 0.9.8 镜像、源码/配置/场景一致性、模型与 norm stats、完整 step400、原 W&B run 和 HF 身份；保存快照，但不启动训练、仿真、W&B run 或上传。镜像缺失或内容不一致时停止，不自动 pull、build 或同步。实际 `--run` 要求宿主机 tmux、GPU 0–7 空闲；GPU 分配使用 YAML，不使用 `CUDA_VISIBLE_DEVICES`。

如果完整恢复点挂在不同位置，可设置 `TACIMG_RESUME_DIR` 为**容器内** `/root/VTLA-RL/.../global_step_400` 路径；该变量只替换恢复来源，仍写入原 melt 输出树。已有源路径/W&B 身份快照后不允许静默更换。所有容器调用都使用同一个检查过的镜像 ID。

脚本在本次 `_launcher` 和原 melt `_launcher` 上持有锁，避免两个启动器同时写入。同一 run 使用 `WANDB_RESUME=must`。每 15 秒检查本次进程启动后的训练指标、GPU/磁盘状态和维护进程；非有限指标或监督进程退出会停止本次容器。失败后保留日志和失败状态，禁止直接从 400 自动重跑；需要先诊断并审阅新的恢复点。首次启动应人工核对 actor 的真实恢复回执及最初 100 个新增 rollout 轮，不能把旧日志 step400 当作本次进展。

每完成 500、600、700、800 轮，后台向 `.env` 的 `HF_TOKEN` 所属用户名下 `2026-10-29-melt-tacimg-pirl` public 仓库上传**完整** `global_step_N/`。若原 `_launcher/hf_owner` 存在，用户名必须一致；在原机器使用原 token 就会继续原仓库。每次上传按固定 commit 核对远端文件清单和大小，失败重试；终点必须核对四个整百点后才标记完成。没有进行真实 HF 上传验证。

每 15 秒校验新保存点，保留本次最近两个完整可恢复点及所有整百点；原 `≤400` 的历史目录保持不动。不完整、正在写入、校验失败或含符号链接的 checkpoint 不删除。成功完成后再次执行，会复核 step800、TensorBoard/W&B step799 和远端上传凭据，然后跳过训练。

## 产物与完成标准

| 内容 | 路径 |
| --- | --- |
| 新 checkpoint | `results/2026-10-29-melt/tacimg/2026-10-29-melt-tacimg-pirl/checkpoints/global_step_{410,…,800}/`，受保留规则约束 |
| TensorBoard / W&B 本地日志 | `results/2026-10-29-melt/tacimg/{tensorboard,wandb}/` |
| 本次配置、源码、启动器和镜像快照 | `results/2026-11-5-resume-tacimage-training/_launcher/` |
| 本次启动命令、训练/GPU/维护/上传日志 | `_launcher/tacimg-stage2/`；stage2 表示沿用的场景 |
| HF 上传凭据 | `_launcher/hf/tacimg/global_step_{500,600,700,800}.json` |
| 正式训练记录 | `Record/experiments/<完整时间>-task820-no_adverb-pirl-resume-tacimage.md`，由 `--run` 创建 |

完成标准：训练进程成功退出、完整 step800 带 `is_final: true` 和 `target_global_step: 800`、TensorBoard/W&B 同一 run 到零基日志步 799 且指标有限、四个整百 checkpoint 上传核验通过。Record 记录原恢复点、参数/源码快照、W&B/HF 链接、精确命令、起止时间、容器训练时长、含上传的总耗时与失败事件。本轮未运行训练，不预填训练结果或创建虚假的训练报告。

## 0.9.7 → 0.9.8 增量镜像

基底：`ccr.ccs.tencentyun.com/vtla/vtla:0.9.7`。

```text
base image ID: sha256:9c5ceef7a23060bfd8ee00a8e4f074656ca950e6e256220a6f6c3efb209e9eb2
base registry digest: sha256:2b4e8a7d19ed2370dccbc2a5081c2ab23b2a771233a4e443ce105cffb49bdf3b
target: ccr.ccs.tencentyun.com/vtla/vtla:0.9.8
```

冻结构建上下文在 [image/context](../../results/2026-11-5-resume-tacimage-training/image/context/)，包含 Dockerfile、原 manifest 加本次增量的源码清单、provenance 和以下文件：

1. 新 `resume800.yaml` 和本 README，覆盖到镜像内原仓库对应路径。
2. 本次启动器同时保存在镜像的原 `experiments` 路径和 `/opt/vtla-build/run-resume-tacimage.sh`，后者用于启动前哈希核对；运行仍在宿主机。

现有 4,587 个 manifest 文件与工作区一致。增量不修改依赖、Python `.venv`、CUDA、IsaacSim、IsaacLab、Tabero_X、USD、原训练算法或原 melt 文件。凭据、基模、训练 checkpoint 和日志不写入镜像，仍按 [docker/README.md](../../docker/README.md) 挂载。

构建前核对基底 ID、上下文哈希和 Docker data-root 空间。此次增量仅包含少量文件，构建日志保存在 `image/build.log`；复现命令如下：

```bash
# 在仓库根目录；只接受上面已核验的基底。
test "$(docker image inspect ccr.ccs.tencentyun.com/vtla/vtla:0.9.7 --format '{{.Id}}')" = \
  sha256:9c5ceef7a23060bfd8ee00a8e4f074656ca950e6e256220a6f6c3efb209e9eb2
docker tag ccr.ccs.tencentyun.com/vtla/vtla:0.9.7 vtla:resume-tacimage-098-base-9c5ceef7a230
(cd results/2026-11-5-resume-tacimage-training/image/context && sha256sum -c SHA256SUMS)
docker build --pull=false --network=none \
  -t ccr.ccs.tencentyun.com/vtla/vtla:0.9.8 \
  results/2026-11-5-resume-tacimage-training/image/context
```

镜像检查包括：父层前缀与 0.9.7 一致、manifest 的全部文件哈希、场景和运行时依赖保持一致、配置解析和 Bash 语法检查；然后在原训练机器执行 `--check` 核验真实 checkpoint/W&B/HF 身份。发布时先 `source .env`，推送后核对远端 manifest config digest 等于本地 image ID；实际发布状态、镜像 ID、远端 digest 和检查结果见 `image/release-validation.json`。

脚本和 README 位于根仓库的本实验目录，配置位于嵌套 RLinf 仓库，跨机器需同步两者。`results` 下的构建包和验证证据是外部产物，不随普通 Git checkout 自动带走。仅分发镜像时，也可将其内保存的脚本导出到宿主机：

```bash
mkdir -p experiments/2026-11-5-resume-tacimage-training
docker run --rm --entrypoint cat ccr.ccs.tencentyun.com/vtla/vtla:0.9.8 \
  /opt/vtla-build/run-resume-tacimage.sh \
  > experiments/2026-11-5-resume-tacimage-training/run-resume-tacimage.sh
```

## 本次验证边界

记录位于 [\_validation](../../results/2026-11-5-resume-tacimage-training/_validation/)：

- Bash 语法、默认 dry-run 和内嵌 Python 编译通过。
- 借用现有 0.9.7 镜像的 RLinf Python、只读挂载源码、无 GPU/网络，完成 Hydra 解析、PiRL 接口、多任务/逐物体力奖励检查；`config-diff.json` 确认只有三处配置差异，`resolved-config.json` 保存解析结果。
- stage2 场景 SHA-256 为 `9108ac46ebb04da772ae812b6a9bff132152feaf133748902129d74bc552bdc5`，与 0.9.7 相同；`base-source-diff.json` 确认原 4,587 个文件无变化或缺失。
- 模拟 Docker 编排通过：单次 400→800、原 W&B ID、四个整百上传、完成后跳过、失败停止、缺失/冲突 ID 拒绝启动。
- 11 项仅使用合成数据的启动器 helper 检查通过：DCP 完整性/截断、优化器必需、多任务状态、保留/并发变动保护、上传清单与凭据、非有限指标、旧 TensorBoard 记录过滤及宿主机/镜像启动器一致性。未为配置、真实模型或数据集新增测试。

这些是静态和模拟证据，不代表原机器的完整 DCP 恢复、worker 初始化、真实训练、仿真或策略效果通过。
