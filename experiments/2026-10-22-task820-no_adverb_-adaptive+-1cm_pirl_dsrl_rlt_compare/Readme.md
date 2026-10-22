# 2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare

这里需要完成三种 rl 方式的对比以选出最合适的rl方式。训练到 stage2 然后做出选择。

所有模型的统一输入为：

* no-state (指的是不输入机械臂关节角度)
* tacfield

两阶段统一为

* stage1 +- 1cm 随机化 0-200 step
* stage2 +- 1cm 随机化 + 根据力大小手动设定的质量随机化 + 力 reward 200-400 step

两阶段使用的场景分别为：

* [gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage1)
* [gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_adaptive_stage_1_3_+-1cm/stage2)

镜像使用 ccr.ccs.tencentyun.com/vtla/vtla:0.9.3 镜像

双阶段配置在 [2026-10-22-adaptive-pirl-dsrl-rlt-compare](../../RLinf/examples/embodiment/config/2026-10-22-adaptive-pirl-dsrl-rlt-compare) 中

对于基模/训练结果的位置约束如下：

* sft 基模在 `models/VTLA-RL-sft-lora-franka-no-adverb-05-effort/step_30000` 下
* 训练结果放到 'results' 下
* rlt stage1 encoder 放到 `models/VTLA-RL-sft-lora-franka-no-adverb-05-effort-rlt-stage1/rlt_encoder.safetensors` 下

注意， pirl 的 stage1 已经在 [2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare](../2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare/Readme.md) 中完成训练，本次训练不训练 pirl stage1，pirl stage2 直接使用已经训练好的 stage1 开始续训

PiRL stage2 的 `runner.resume_dir` 指向旧实验的完整 checkpoint（路径相对于仓库根目录）：

```text
results/2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare/pirl/2026-10-20-task820-no_adverb_xy_1cm_pirl_compare/checkpoints/global_step_200
```

该目录必须包含 actor 的 DCP 模型、优化器、调度器、RNG 状态及 `multi_task_state.json`；推理 bundle 或单独的 `trainable_weights.pt` 不能替代。`--check` 和 stage2 启动前均校验实际续训目录。

三种方法的共同训练规模为：8 GPU（actor/rollout 0–7，env 0–5），192 个训练环境、96 个 eval 环境，每轮 300 个环境步，action chunk 10，micro batch 48、global batch 384、seed 42。阶段目标是累计 rollout 轮数 200/400，不是优化器更新次数；每阶段新增 200 轮，每轮名义采样 57,600 个环境步。训练中 eval 关闭，每 10 轮保存 checkpoint，每 100 轮上传。

同一阶段的环境奖励一致：成功奖励 1；stage1 关闭力奖励，stage2 启用 normalized effort reward，系数 10，Vitasoy/Coca-Cola/cookie 的 min/mid effort 分别为 2/36、4/12、3/30。保留各算法原有优化设置：PiRL/DSRL/RLT 的 gamma 分别为 0.99/0.999/0.995；PiRL 每轮遍历数据 1 次，DSRL 在 replay 就绪后每轮更新 8 次，RLT 按 transition 调度、每轮最多 500 次。因此采样预算一致，但优化计算量和折扣回报不相同。

复用 PiRL stage1 也有环境历史差异：旧 stage1 未显式覆盖物体质量，新 adaptive stage1 为三个目标设定固定质量 0.1/0.2/0.3 kg，并将篮子质量设为 0.2–0.5 kg。不能据此声称三种方法的 stage1 环境完全相同；本次按上述要求复用已训练的 PiRL stage1。

拉取镜像

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:0.9.3
```

使用[脚本](./run_stage12.sh)直接自动化完成所有训练，顺序为 pirl/dsrl/rlt

在宿主机的仓库根目录运行，顺序为 **PiRL stage2 → DSRL stage1 → DSRL stage2 → RLT stage1 → RLT stage2**。本次 adaptive 实验每种方法使用独立的新 W&B run ID 和名称；DSRL/RLT 的 stage2 复用各自 stage1 的 ID，PiRL stage2 从旧 checkpoint 恢复训练状态并新建本实验的 W&B run。同时记录 TensorBoard。

脚本从仓库根目录 `.env` 获取 `HF_TOKEN`，通过 token 查询用户名，为每个实验创建一个 public 模型仓库，仓库名称为 `experiment_name`。每完成 100 步，后台上传完整的 `checkpoints/global_step_N/` 到该仓库的 `global_step_N/`，包括 actor、优化器、replay、调度器和多任务状态，下载后可直接作为 `resume_dir` 使用。

脚本等待 checkpoint 完整落盘，再上传并核对远端文件清单及大小；成功记录保存在 `_launcher/hf/<method>/`，不会重复上传。失败会重试，当前训练继续，但进入下一阶段前必须上传完本阶段的整百 checkpoint。上传日志为各阶段目录中的 `hf.log`。`--check` 只验证 HF_TOKEN 身份，不创建仓库或上传。需要代理时，在宿主机导出 `HTTPS_PROXY` / `HTTP_PROXY`（或对应小写变量），脚本会传入检查和上传容器。

```bash
SCRIPT=experiments/2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/run_stage12.sh
bash "$SCRIPT" --dry-run  # 只显示顺序
bash "$SCRIPT" --check    # 检查镜像、源码、模型路径和配置，不训练
tmux new -s task820-compare
bash experiments/2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/run_stage12.sh --run
```

配置快照、各阶段日志和 W&B 链接保存在 `results/2026-10-22-task820-no_adverb_-adaptive+-1cm_pirl_dsrl_rlt_compare/_launcher/`，训练报告写入 `Record/experiments/`。阶段失败会停止后续训练；再次运行会检查并跳过已完成阶段。

各方法的 `experiment_name` 为 `2026-10-22-task820-no_adverb_adaptive_xy_1cm_<method>_compare`。`log_path` 为上述 results 实验目录下的 `<method>/`；TensorBoard 和 W&B 本地日志分别在 `log_path/tensorboard/`、`log_path/wandb/`，checkpoint 在 `log_path/experiment_name/checkpoints/`。
