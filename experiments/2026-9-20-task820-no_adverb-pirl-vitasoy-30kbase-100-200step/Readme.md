# 2026-9-20-task820-no_adverb-pirl-vitasoy-30kbase-100-200step

完成 [训练报告](https://zanpw3z2hb6.feishu.cn/docx/XWEpd2jCZovDAqxJOC8cpATAnwe) 中的 Demomasterlqx/VTLA-RL-sft-lora-xarm-no-adverb-pirl-vitasoy-2026-9-17 的 100-200 训练

## 数据准备

* 拉取基模

```bash
hf download Demomasterlqx/VTLA-RL-sft-lora-xarm-no-adverb --include "global_step_30000/model/**" --local-dir "./models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model"
```

* 拉去续训的 100 step 模型

```bash
hf download Demomasterlqx/2026-09-17_16-17-task820-no_adverb-pirl-vitasoy-30kbase-300step --include "vitasoy/resume_checkpoints/step_100/**" --local-dir "results/2026-9-20-task820-no_adverb-pirl-vitasoy-30kbase-100-200step/checkpoints/global_step_100"
```

* 准备好 .env 环境

在 .env 中设置好 WANDB_API_KEY

* pull 新的镜像

新的镜像为增量构建，应该在1min以内就可以拉下来

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:2026-9-20-task820-no-adverb-pirl-vitasoy-30kbase-100-200step-local
```

## 训练

启动docker

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:2026-9-20-task820-no-adverb-pirl-vitasoy-30kbase-100-200step-local"

docker compose run --rm rlinf

```

启动训练

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_benchmark_vitasoy_force_reward 
```