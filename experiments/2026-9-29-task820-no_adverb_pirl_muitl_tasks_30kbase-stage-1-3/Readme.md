# 2026-9-29-task820-no_adverb_pirl_muitl_tasks_30kbase-stage-1-3

三个目标物体（Vitasoy、Coca-Cola、cookie）的多任务 PiRL 训练，当前仅配置 stage 1（0–200 step）。

使用 8 卡共置、192 个训练环境，每卡三个环境 stage，每个目标共 64 个环境；actor micro/global batch 为 48/384。环境 stage 与训练阶段 stage 1/2/3 是不同概念。

## 数据准备

* 拉取基模

```bash
source .env
HF_TOKEN="$HF_TOKEN" hf download Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort --include "step_30000/**" --local-dir "./models/VTLA-RL-sft-lora-franka-no-adverb-05-effort/"
```

* 准备好 `.env` 环境

在 `.env` 中设置好 `WANDB_API_KEY`。

* 在仓库根目录拉取包含本次多任务代码与配置的新版 0.6 镜像

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:0.6
```

## 训练

启动 docker：

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:0.6"
docker compose run --rm rlinf
```

只启动 stage 1

step 0-200

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_multi_tasks_tacfield_8gpu_benchmark
```
