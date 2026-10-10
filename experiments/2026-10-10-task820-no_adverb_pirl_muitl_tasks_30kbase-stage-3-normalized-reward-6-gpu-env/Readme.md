# 2026-10-10-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-3-normalized-reward-6-gpu-env

在 [2026-9-29-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-1-3](../2026-9-29-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-1-3/Readme.md) 的基础上，尝试完成对于 stage2 的续训。

首先拉取 0.8.3 版本的 docker 镜像

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:0.8.3
```

配置文件：[RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_normalized_reward_6_env_gpu.yaml](../../RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_normalized_reward_6_env_gpu.yaml)

续训前，将该配置的 `runner.resume_dir` 设置为 Stage 2 的 `global_step_<400>` checkpoint 目录。

启动

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:0.8.3"
docker compose run --rm rlinf
```

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark_stage_3_normalized_reward_6_env_gpu
```

训练从 400 训练到 800