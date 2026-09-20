FROM ccr.ccs.tencentyun.com/vtla/vtla:0.4

# Build with the repository root as the context.  The base image already
# provides the runtime environments and the full repositories; this layer
# only overlays files that are present in the local checkout but absent or
# stale in the published image.
WORKDIR /root/VTLA-RL

COPY RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_benchmark_vitasoy_force_reward.yaml \
     /root/VTLA-RL/RLinf/examples/embodiment/config/isaaclab_pi05_pirl_task820_tacfield_8gpu_vtla_rl_benchmark_vitasoy_force_reward.yaml

COPY Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5griper.json \
     /root/VTLA-RL/Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5griper.json

COPY Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5griper \
     /root/VTLA-RL/Tabero_X/benchmarks/datasets/realworld/config/gentle_grasp_0.5griper
