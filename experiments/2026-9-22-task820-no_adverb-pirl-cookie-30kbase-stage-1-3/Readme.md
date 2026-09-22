# 2026-9-22-task820-no_adverb-pirl-cookie-30kbase-stage-1-3

仿照 `2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3`，使用相同的 8 GPU、64 train env、32 eval env 和 0-100/100-200/200-300 step 训练规模；夹取目标改为 Coca-Cola，基础镜像改为 `vtla:0.5`。

环境快照沿用 v0.5 中的三阶段 Task820 配置，仅在 RLinf YAML 中将目标切换为 `target_object_2`。

## 数据准备

* 拉取基模

```bash
source .env
hf download Demomasterlqx/VTLA-RL-sft-lora-xarm-no-adverb --include "global_step_30000/model/**" --local-dir "./models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model"
```

* 准备好 `.env` 环境

在 `.env` 中设置好 `WANDB_API_KEY`。

* pull 新的镜像

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:2026-9-22-task820-no_adverb-pirl-cookie-30kbase-stage-1-3
```

如需从 v0.5 基础镜像本地增量构建：

```bash
./experiments/2026-9-22-task820-no_adverb-pirl-cookie-30kbase-stage-1-3/docker-build.sh
```

## 训练

启动 docker：

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:2026-9-22-task820-no_adverb-pirl-cookie-30kbase-stage-1-3"
docker compose run --rm rlinf
```

依次启动这三个训练：

step 0-100

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_1
```

step 100-200

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_2
```

step 200-300

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_3
```
