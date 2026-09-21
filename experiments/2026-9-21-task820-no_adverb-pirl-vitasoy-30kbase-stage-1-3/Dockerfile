ARG BASE_IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.4
FROM ${BASE_IMAGE}

# Build from the workspace root. Reuse the base runtime and overlay only
# the RLinf no-state implementation and this experiment's configurations.
WORKDIR /root/VTLA-RL

COPY RLinf/rlinf/config.py /root/VTLA-RL/RLinf/rlinf/config.py
COPY RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py /root/VTLA-RL/RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py
COPY RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py /root/VTLA-RL/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
COPY RLinf/rlinf/utils/tabero_ppo_boundary.py /root/VTLA-RL/RLinf/rlinf/utils/tabero_ppo_boundary.py

COPY RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_1.yaml \
     RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_2.yaml \
     RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_vitasoy_2026_9_21_stage_3.yaml \
     /root/VTLA-RL/RLinf/examples/embodiment/config/

COPY Tabero_X/benchmarks/datasets/realworld/config/2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3/ \
     /root/VTLA-RL/Tabero_X/benchmarks/datasets/realworld/config/2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3/
