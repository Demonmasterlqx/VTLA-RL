# 2026-10-17-z-task820-no_adverb_+-2cm_delta_500_pirl_dsrl_rlt_compare

这里需要完成三种 rl 方式的对比以选出最合适的rl方式。训练只训练到 stage2 然后做出选择。

所有模型的统一输入为：

* no-state
* tacfield

三阶段统一为

* stage1 +- 1cm 随机化
* stage2 +- 1cm 随机化 + +500g 随机化
* stage3 +- 1cm 随机化 + +500g 随机化 + 力 reward

镜像使用 ccr.ccs.tencentyun.com/vtla/vtla:0.8.9 镜像

基模统一为 [Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort](https://huggingface.co/Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort) 的 `step_30000`

镜像启动方式

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:0.8.9"
docker compose run --rm rlinf
```

## PIRL

参考 [2026-10-10 实验](../2026-10-10-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-normalized-reward-6-gpu-env/Readme.md)，从相同 Stage 1 `global_step_200` 起点训练至累计 step 400。

* stage2 使用配置 [isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_2_10_times_normalized_reward_6_env_gpu_delta_500g_xy_1cm](RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_2_10_times_normalized_reward_6_env_gpu_delta_500g_xy_1cm.yaml)

## DSRL

* stage1 使用配置 [isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage1](../../RLinf/examples/embodiment/config/isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage1.yaml)
* stage2 使用配置 [isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage2](../../RLinf/examples/embodiment/config/isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage2.yaml)

启动方式为：

```bash
export WANDB_RESUME=must
python ./examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/RLinf/examples/embodiment/config" \
  --config-name isaaclab_pi05_dsrl_task820_multi_tasks_tacfield_8gpu_benchmark_normalize_reward_stage1
```

## RLT

rlt stage 1 模型在 https://huggingface.co/Demomasterlqx/vtla-rl-sft-tabero-no-state-no-adverb-force-005-80k-2026-10-5

* stage1 使用配置 [isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage1](../../RLinf/examples/embodiment/config/isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage1.yaml)
* stage2 使用配置 [isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage2](../../RLinf/examples/embodiment/config/isaaclab_pi05_rlt_task820_multi_tasks_tacfield_8gpu_benchmark_stage2.yaml)
