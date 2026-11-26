# TacCoRL PiRL + SFT, 400 rollout rounds

在目标机器的宿主机启动脚本，由 Docker `ccr.ccs.tencentyun.com/vtla/vtla:0.11.1`
运行 RLinf。0.11.1 从固定的 0.11.0 增量构建，包含本次审核的源码。
多任务目标只有 Coca-Cola、Vitasoy；输入为 no-state、双 RGB 和触觉 marker history，
8 prefix + 8 suffix tactile tokens。同步 `EmbodiedRunner`，无副词 Task6。

## 训练合同

| 参数 | 值 |
| --- | --- |
| 基模 | `results/2026-11-24-task927-firm-taccorl-sft-30k/step_30000`，合并 EMA |
| 基模 SHA256 | `8fb715689772433d7a321e14d03fbfc266cbc11cc6636a84fab7fa29db9778c7` |
| norm SHA256 | `c944fb4951d7968b6602edc4cf7ce0bbe8f098e40fed8bcbfb29358dbd25da61` |
| GPU | 8 卡；env/actor/rollout 均为 0–7；RLinf placement 分配 |
| 环境 | `Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0`；128 个训练环境，两任务各 64，每个 env worker 16 个；不创建评估环境 |
| rollout | 每轮 300 控制步；horizon 50、执行 chunk 10、去噪 10 步 |
| batch | micro 48 × actor world 8 × 累积 1 = global 384 |
| 可训练模块 | action expert LoRA rank 32、marker encoder、tactile cross-attention、value head |
| 优化 | AdamW，actor LR `1e-6`、value LR `1e-4`，betas `0.9/0.95`，eps `1e-8` |
| 调度 | constant，无 warmup，weight decay `0.01`，gradient clip `1.0` |
| PPO | gamma `0.99`，GAE `0.95`，clip `0.2`，update_epoch `1` |
| 精度 | frozen BF16、trainable FP32、BF16 autocast；FSDP full_shard |
| 梯度检查点 | 开启，non-reentrant；actor/rollout offload 开启 |
| EMA / seed | 初始化使用 SFT EMA，RL 无模型 EMA；任务成功率 EMA `0.9`；seed `42` |
| SFT | 真机 Task927 两任务共 60 episodes / 8,676 frames；PPO + `1.0 × SFT` |
| SFT loss | weighted_full；action `1.0`、future force `0.05`、padding `1.0` |
| 日志 | SwanLab [`Demonmaster/VTLA-RL`](https://swanlab.cn/@Demonmaster/VTLA-RL/quick-start) + TensorBoard；两阶段保持同一 run ID |

真机数据沿用此前 SFT 的 `realworld_task927_firm_4x4_model_coordinates` 数据视图，
原始六维力和 5 Hz 帧序列保持不变，使用基模配套 norm。episode 0–29 为维他奶，
30–59 为可乐；单域 loader 在分布式采样前筛选，不使用 cookie 示范。
PPO 动态任务权重不作用于独立的 SFT 项。

| 阶段 | 累计 rollout 轮数 | 场景 / 奖励 |
| --- | --- | --- |
| stage1 | 0→200 | XY ±1 cm；维他奶 100 g、可乐 200 g；成功奖励 1 |
| stage2 | 200→400 | XY ±1 cm；维他奶 100–600 g、可乐 150–250 g；成功 + normalized effort reward |

stage2 力奖励系数 10，min/max effort 分别为维他奶 2/36 N、可乐 4/12 N。
两阶段篮子质量均为 200–500 g。配置见
[stage1](../../RLinf/examples/embodiment/config/2026-11-25-taccorl-pirl-step400/stage1.yaml)、
[stage2](../../RLinf/examples/embodiment/config/2026-11-25-taccorl-pirl-step400/stage2.yaml)。
`stage1.yaml` 包含完整训练、环境、FSDP 和权重同步参数；`stage2.yaml` 仅继承同目录的
`stage1.yaml`。除 Hydra 内置日志配置外，不依赖其他配置目录。
除用户指定的 env 八卡和每任务 64 环境外，其余参数沿用之前审核的 PiRL 设置，参考
[原实验](../2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/Readme.md)。
400 是 rollout 轮数，每轮名义采样 38,400 环境步，400 轮共 15,360,000 步；
每轮通常 10 次优化器更新，实际有效样本与跳过空 batch 的次数以日志为准。
仅执行训练：`val_check_interval=-1`、`only_eval=false`、`env.eval=null`。
不运行周期评估、训练结束后的成功率评测或推理测试。

## 启动与恢复

从宿主机仓库根目录运行。`.env` 提供 `SWANLAB_API_KEY`，凭据不会写进镜像。
指定 `TACCORL_GPU_UUIDS` 为目标机器的八张卡，按 PCI 顺序填写逗号分隔 UUID。
目标机器需有基模、真机数据视图及与镜像匹配的源码 checkout。辅助脚本已纳入本目录的
`tools/`，启动时自动复制到结果目录；首次启动从镜像读取源码哈希和提交清单。
直接使用已发布的镜像：

```bash
docker --context default pull ccr.ccs.tencentyun.com/vtla/vtla:0.11.1
```

如需重新构建，构建机器还需有 `results/docker-0.11.0-20261124/source-manifest.json`。
从当前源码生成八卡运行快照，再以固定的 0.11.0 digest 构建 0.11.1：

```bash
python3 experiments/2026-11-25-taccorl-pirl-step400/tools/prepare_image.py
OUT=results/2026-11-25-taccorl-pirl-step400/8gpu
docker --context default build --pull=false -t ccr.ccs.tencentyun.com/vtla/vtla:0.11.1 "$OUT/image/context"
docker --context default image inspect ccr.ccs.tencentyun.com/vtla/vtla:0.11.1 --format '{{.Id}}' > "$OUT/image/image_id"
```

```bash
SCRIPT=experiments/2026-11-25-taccorl-pirl-step400/run_stage12.sh
bash "$SCRIPT" --dry-run
bash "$SCRIPT" --check
tmux new -s taccorl-pirl-step400
bash "$SCRIPT" --smoke
bash "$SCRIPT" --run
```

`--smoke` 使用独立目录和 SwanLab ID，保持完整八卡拓扑、128 env、300 控制步和 micro48/global384：
stage1 完成两轮、保存完整 checkpoint，再按 stage2 奖励配置恢复到第三轮。
检查 PPO/SFT 更新、恢复、两任务样本、权重和日志后退出容器；不启动正式 400 轮训练。
正式训练前必须通过上述验证。每 10 轮保存完整 checkpoint；stage2 完整恢复
本实验 step200。再次 `--run` 从最新完整 checkpoint 恢复，不覆盖已有日志和配置。
模型、优化器、调度器、RNG、`multi_task_state.json`、SFT 合同及数据游标均需保留。

八卡结果根目录：`results/2026-11-25-taccorl-pirl-step400/8gpu/`。
此前 DOIT 四卡 smoke 已按用户要求停止；其记录留在父目录，不复用其验收标记或 checkpoint。
配置、环境、源码和镜像身份保存在 `provenance/`、`config/`、`env/`、`image/`；
日志在 `train/`，checkpoint 在 `train/2026-11-25-taccorl-pirl-step400/checkpoints/`，
最终部署导出在 `exports/step_400/`。报告写入 `Record/experiments/`。
SwanLab run ID、容器名、退出码和阶段状态保存在 `_launcher/`。

前 100 轮每 5–30 秒监督，之后持续检查至终点。验收包括 step400 完整 checkpoint、
SwanLab/TensorBoard 步轴一致、合并权重的 CPU 严格加载与文件哈希核验，以及本次 GPU 进程退出。
日志中的成功率来自训练 rollout。
