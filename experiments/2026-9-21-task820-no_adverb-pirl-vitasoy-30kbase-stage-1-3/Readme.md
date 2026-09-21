# 2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3

完成 [训练报告](https://zanpw3z2hb6.feishu.cn/docx/XWEpd2jCZovDAqxJOC8cpATAnwe) 中的 Demomasterlqx/VTLA-RL-sft-lora-xarm-no-adverb-pirl-vitasoy-2026-9-17 的 100-200 训练

## 数据准备

* 拉取基模

```bash
hf download Demomasterlqx/VTLA-RL-sft-lora-xarm-no-adverb --include "global_step_30000/model/**" --local-dir "./models/VTLA-RL-sft-lora-xarm-no-adverb/global_step_30000/model"
```

* 准备好 .env 环境

在 .env 中设置好 WANDB_API_KEY

* pull 新的镜像

新的镜像为增量构建，应该在1min以内就可以拉下来

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3
```

## 训练

启动docker

```bash
export TABERO_IMAGE="ccr.ccs.tencentyun.com/vtla/vtla:2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3"

docker compose run --rm rlinf

```

依次启动训练这三个训练

step 0-100

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_1 
```

step 100-200

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_2
```


step 200-300

```bash
python ./examples/embodiment/train_embodied_agent.py --config-path "/root/VTLA-RL/RLinf/examples/embodiment/config" --config-name isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_3
```