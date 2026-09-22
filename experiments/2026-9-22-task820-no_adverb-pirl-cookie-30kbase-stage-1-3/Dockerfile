ARG BASE_IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.5
FROM ${BASE_IMAGE}

# The v0.5 base already contains the runtime, environments, assets, and the
# validated Task820 no-state implementation. Overlay only this experiment's
# three staged training configurations.
WORKDIR /root/VTLA-RL

COPY RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_1.yaml \
     RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_2.yaml \
     RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_6000pro_cookie_2026_9_22_stage_3.yaml \
     /root/VTLA-RL/RLinf/examples/embodiment/config/

