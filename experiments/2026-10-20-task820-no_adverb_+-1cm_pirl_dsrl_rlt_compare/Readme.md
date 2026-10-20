# 2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare

这里需要完成三种 rl 方式的对比以选出最合适的rl方式。训练到 stage2 然后做出选择。

所有模型的统一输入为：

* no-state (指的是不输入机械臂关节角度)
* tacfield

两阶段统一为

* stage1 +- 1cm 随机化 0-200 step
* stage2 +- 1cm 随机化 + +500g 随机化 + 力 reward 200-400 step

两阶段使用的场景分别为：

* [gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm/stage1](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm/stage1)
* [gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm/stage2](../../Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5_griper_pos_random_delta_500g_stage_1_3_+-1cm/stage2)

镜像使用 ccr.ccs.tencentyun.com/vtla/vtla:0.9.1 镜像

双阶段配置在 [2026-10-20-pirl-dsrl-rlt-compare](../../RLinf/examples/embodiment/config/2026-10-20-pirl-dsrl-rlt-compare) 中

对于基模/训练结果的位置约束如下：

* sft 基模在 `models/VTLA-RL-sft-lora-franka-no-adverb-05-effort/step_30000` 下
* 训练结果放到 'results' 下
* rlt stage1 encoder 放到 `models/VTLA-RL-sft-lora-franka-no-adverb-05-effort-rlt-stage1/rlt_encoder.safetensors` 下

拉取镜像

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:0.9.1
```

使用[脚本](./run_stage12.sh)直接自动化完成所有训练，顺序为 pirl/dsrl/rlt

在宿主机的仓库根目录运行；每种方法依次完成 stage1、stage2，并复用同一个 W&B run ID 和链接，同时记录 TensorBoard。

```bash
SCRIPT=experiments/2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare/run_stage12.sh
bash "$SCRIPT" --dry-run  # 只显示顺序
bash "$SCRIPT" --check    # 检查镜像、源码、模型路径和配置，不训练
tmux new -s task820-compare
bash experiments/2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare/run_stage12.sh --run
```

配置快照、各阶段日志和 W&B 链接保存在 `results/2026-10-20-task820-no_adverb_+-1cm_pirl_dsrl_rlt_compare/_launcher/`，训练报告写入 `Record/experiments/`。阶段失败会停止后续训练；再次运行会检查并跳过已完成阶段。
