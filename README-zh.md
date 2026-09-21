# VTLA-RL

本工作区把 IsaacLab、Tabero_X、RLinf 和 T2-VLA 作为一个可复现的研究环境管理。
宿主机和 Docker 镜像使用相同的仓库分支约定；NVIDIA 驱动始终由宿主机提供。

## 数据与模型

在 [VTLA-RL 训练记录](https://zanpw3z2hb6.feishu.cn/docx/XWEpd2jCZovDAqxJOC8cpATAnwe) 中记录了所有数据与模型链接。

## 部署

### uv

> 本地 uv 部署部分未完善

本节说明宿主机本地使用 uv 配置和进入开发环境。三个 uv 环境分别是：

| 用途 | Python |
| --- | --- |
| IsaacLab + Tabero_X（共享） | `IsaacLab/.venv/bin/python` |
| RLinf | `RLinf/.venv/bin/python` |
| T2-VLA | `T2-VLA/.venv/bin/python` |

Isaac Sim 二进制位于 `isaacsim/`，IsaacLab 和 RLinf 通过各自的链接使用同一份二进制。
Isaac Sim 5.1.0 standalone 安装包下载地址：

```text
https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

进入 IsaacLab/Tabero_X 开发 Shell 时执行：

```bash
export ROOT="$(pwd)"
source "$ROOT/IsaacLab/.venv/bin/activate"
source "$ROOT/isaacsim/setup_python_env.sh"
export ISAACSIM_PATH="$ROOT/isaacsim"
export ISAACLAB_PATH="$ROOT/IsaacLab"
export PYTHONPATH="$ROOT/isaacsim/extscache/omni.warp.core-1.8.2+lx64:$ROOT/isaacsim/extscache/omni.warp-1.8.2:${PYTHONPATH:-}"
cd "$ROOT/Tabero_X"
```

`RLinf/.venv` 和 `T2-VLA/.venv` 不切换到共享环境，直接调用对应解释器：

```bash
RLinf/.venv/bin/python -u RLinf/examples/embodiment/train_embodied_agent.py ...
T2-VLA/.venv/bin/python T2-VLA/scripts/serve_policy.py ...
```

Tabero_X 的 Taxim GPU 后端使用共享环境中的 JAX CUDA 12（0.5.3）。首次配置或依赖
修复时安装 Tabero_X requirements 和扩展：

```bash
uv venv --seed --python 3.11 IsaacLab/.venv
uv pip install --python IsaacLab/.venv/bin/python -r Tabero_X/requirements.txt
uv pip install --python IsaacLab/.venv/bin/python -e Tabero_X/source/tac_manip
```

验证三个解释器和共享环境的关键包：

```bash
env -u PYTHONPATH IsaacLab/.venv/bin/python -c \
  'import torch, jax, torch_scatter; print(torch.__version__, jax.__version__, torch_scatter.__version__)'
RLinf/.venv/bin/python -c 'import rlinf; print(rlinf.__file__)'
T2-VLA/.venv/bin/python -c 'import openpi; print(openpi.__file__)'
```

### docker

**拉取仓库**

> 由于 Tabero_X 仓库是 private 的，所以拉去之前请保证当前用户 ~/.ssh/config 中连接 github.com 的凭证是有权限的。
>
> 如果是公用服务器导致不好直接设置 github.com 可以通过设置环境变量 `GIT_SSH_COMMAND` 解决
> 
> ```
> export GIT_SSH_COMMAND="ssh -i /path/to/github_private_key" 
> ```

```bash
git clone --recursive https://github.com/Demonmasterlqx/VTLA-RL.git

# 如果是开发者
# git clone git@github.com:Demonmasterlqx/VTLA-RL.git

cd VTLA-RL

```

如果已经完成仓库拉去，但是没有克隆子模块

```bash
cd VTLA-RL
git submodule update --init --recursive
```

**初始化文件目录与密钥**

```bash

cd VTLA-RL
mkdir results
mkdir models
mkdir Record
mkdir datasets
cp .env.example .env

```

在 [.env](.env) 中完善 huggingface token 以及 wandb token

```
export HF_TOKEN="XXX"
export WANDB_API_KEY="XXX"
```

> results/models/Record/datasets/.env 会在运行时被挂载到镜像中。
>
> 一定要在运行镜像之前创建这四个目录，如果让运行镜像时创建会导致宿主机没有足够的权限控制这几个目录

**拉取镜像**

```bash
docker pull ccr.ccs.tencentyun.com/vtla/vtla:0.5
```

尝试运行

```bash
docker compose run --rm shell
```

能进入 shell 即表示完成

更具体的使用查看 [docker/README.md](docker/README.md) 中的 `训练、导出与评测` 章节

**拉取数据**

在本地安装 hf 客户端，[Command Line Interface (CLI)](https://huggingface.co/docs/huggingface_hub/guides/cli)

> 如果有网络问题，可以设置
>
> ```
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> 或者设置网络代理
>
> ```bash
> export http_proxy="http://xxxxx"
> export https_proxy="http://xxxxx"
> export HTTP_PROXY="http://xxxxx"
> export HTTPS_PROXY="http://xxxxx"
> ```

下载数据样例命令

```bash
hf download xiangxin0923/realworld_replay_task820_firm_gentle_mixed_current --repo-type dataset --local-dir "./datasets/path/to/target/datasets"
```

下载模型样例命令

```bash
hf download Demomasterlqx/VTLA-RL-sft-lora-franka-no-adverb-05-effort --include "step_30000/**" --local-dir "./models/path/to/target/models"
```

> 注意下载模型的时候需要上 hf check 一下模型是否为我们期望的部分，因为 hf 仓库中会有一些可以续训的检查点

## docker 开发与构建

### 构建镜像

默认镜像为 `ccr.ccs.tencentyun.com/vtla/vtla:0.5`，基础镜像为 Isaac Sim 5.1，
工作区位于 `/root/VTLA-RL`。镜像使用基础镜像自带的 `/isaac-sim`，并创建
`/root/VTLA-RL/isaacsim -> /isaac-sim`，不会复制第二份 Isaac Sim。

apt、uv 和 pip 使用 Docker builder 内的 BuildKit 持久缓存，不再把宿主机
`$HOME/.cache/uv` 和 `$HOME/.cache/pip` 作为额外构建上下文上传。Compose 中只有
`shell` 服务声明 `build`，其余服务复用相同镜像，因此以下命令只构建一次：

```bash
export HTTP_PROXY=http://127.0.0.1:10080
export HTTPS_PROXY=http://127.0.0.1:10080
export ALL_PROXY=
export PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
export HF_ENDPOINT=https://hf-mirror.com
export TABERO_IMAGE=ccr.ccs.tencentyun.com/vtla/vtla:0.5

BUILDKIT_PROGRESS=plain docker compose build
```

其中 `PYPI_INDEX_URL` 用于普通 Python 包和 IsaacLab 内部的 pip 安装；
`PYTORCH_INDEX_URL` 用于 CUDA 12.8 的 PyTorch wheel；`HF_ENDPOINT` 用于
Tabero_X 固定版本资产。资产下载使用独立 Docker 层和 BuildKit 缓存，偶发断线后
可继续补齐文件；构建时禁用会绕过镜像的 Hugging Face Xet/CAS 传输，重建不会
重新安装 IsaacLab。PyG wheel与 GitHub 仓库继续使用各自专用来源。需要
切回官方源时可分别设置为 `https://pypi.org/simple`、
`https://download.pytorch.org/whl/cu128` 和 `https://huggingface.co`。

第一次构建需要下载依赖，后续构建会复用 BuildKit 缓存。使用独立本地标签验证时：

```bash
export TABERO_IMAGE=vtla:env-fix
BUILDKIT_PROGRESS=plain docker compose build
```

镜像中的 IsaacLab/Tabero_X 使用 `/root/VTLA-RL/IsaacLab/.venv`，RLinf 和
T2-VLA 各自使用仓库下的 `.venv`。容器入口会自动 source 只读挂载的
`/root/VTLA-RL/.env`；不要将 HF 或 W&B token 写入 Dockerfile、镜像层或 Git。

当前经过训练和 bundle 加载 smoke 验证的兼容组合是：T2-VLA 使用
`peft==0.20.0`，RLinf 使用 `peft==0.21.0`，两边均使用
`torch==2.7.1+cu128`。两套 PEFT 版本是导出/部署合同的一部分，不要在现有
`.venv` 中就地升级。完整的版本核对、SFT/PIRL/DSRL/RLT Stage 1/Stage 2、bundle
导出和 mode-6 视频评测命令见 `docker/README.md`。

`.env` 至少包含以下两个键（文件应放在宿主机并限制为 `chmod 600`）：

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

### 启动容器

```bash
docker compose run --pull never --rm shell
docker compose run --pull never --rm tabero_x
docker compose run --pull never --rm rlinf
docker compose run --pull never --rm t2-vla
```

Compose 将宿主机的 `datasets`、`models`、`results`、`Record` 分别挂载到
`/root/VTLA-RL` 下对应目录；前两者只读，后两者可写。也可以显式指定凭据文件：

可用 `TABERO_DATASETS_HOST`、`TABERO_MODELS_HOST`、`TABERO_RESULTS_HOST` 和
`TABERO_RECORD_HOST` 分别覆盖四个宿主机目录的完整路径。训练前应检查输出盘
剩余空间；设置这些变量不会迁移已有结果。具体命令见 `docker/README.md`。

```bash
export TABERO_ENV_FILE=/secure/path/vtla.env
docker compose run --rm shell
```

- `shell` 是不自动激活 Python 环境的通用 Shell。
- `tabero_x` 自动执行 `docker/env_setup/tabero_x.sh`。
- `rlinf` 自动执行 `docker/env_setup/rlinf.sh`，加载 Isaac Sim、IsaacLab 和
  Tabero_X 所需运行路径。
- `t2-vla` 自动执行 `docker/env_setup/t2-vla.sh`。

直接 `docker run` 时需使用 `--gpus all`、`--ipc host`，并外挂 `.env` 和上述四个目录。
镜像不包含宿主机 NVIDIA 驱动，运行前需安装 NVIDIA Container Toolkit。

### 启动训练与测试

四个服务共享同一镜像，但入口会按服务加载不同环境：

```bash
docker compose run --pull never --rm shell
docker compose run --pull never --rm tabero_x
docker compose run --pull never --rm rlinf
docker compose run --pull never --rm t2-vla
```

- `shell` 不自动激活 Python 环境。
- `tabero_x` 使用 `IsaacLab/.venv`，并加载 Isaac Sim/IsaacLab 运行变量。
- `rlinf` 使用 `RLinf/.venv`，并加载 Isaac Sim、IsaacLab 和 Tabero_X 路径。
- `t2-vla` 使用 `T2-VLA/.venv`。

在同一宿主机启动端口 `10010` 的 T2-VLA 服务器：

```bash
docker compose run --pull never --rm -T t2-vla t2-vla -lc '\
exec "$ROOT/T2-VLA/.venv/bin/python" -u \
"$ROOT/T2-VLA/scripts/serve_policy.py" \
    --port 10010 \
    policy:checkpoint \
    --policy.config="<policy-config>" \
    --policy.dir="$ROOT/models/<checkpoint-dir>"'
```

运行 Tabero_X 客户端时，使用同一 host network 下的 `127.0.0.1:10010`。例如：

```bash
docker compose run --pull never --rm -T tabero_x tabero_x -lc '\
exec "$ROOT/IsaacLab/.venv/bin/python" -u \
"$ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py" \
    --server-host 127.0.0.1 \
    --server-port 10010 \
    --task Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0 \
    --task-domain realworld \
    --task-suite gentle_grasp \
    --task-id 6 \
    --control-mode hybrid_tactile \
    --hdf5-folder "$HDF5_TRAJ_SOURCE_DIR" \
    --num-total-experiments 50 \
    --max-inference-steps 30 \
    --language-instruction "pick up the Vitasoy and put it into the basket" \
    --replan-steps 10 \
    --debug-mode 6 \
    --debug-path "$ROOT/results/openpi_video" \
    --debug-run-dir "$ROOT/results/openpi_video/artifacts" \
    --smooth-action-chunks \
    --device cuda:0 \
    --headless'
```

`tabero_x` 入口会设置默认 `HDF5_TRAJ_SOURCE_DIR`。视频和调试产物写入
`results/openpi_video`，该目录通过 Compose 映射回宿主机。更完整的启动、验证和
参数示例见 `docker/README.md`。

RLinf 训练的 GPU 只通过配置中的 `cluster.component_placement` 分配，不使用
`CUDA_VISIBLE_DEVICES`。T2-VLA/Tabero_X 推理可显式设置物理 GPU，但需区分物理
Vulkan 编号和进程内 `cuda:0`。PIRL、DSRL、RLT bundle 分别使用
`--lora-bundle`、`--dsrl-bundle`、`--rlt-bundle`，且三者互斥。RLT 导出时必须填写
T2 部署配置 `pi05_lora_tacfield_realworld_replayed_task820_firm_mixed_no_state`，不能
误用 RLinf training config 名称。

保存可审计视频时使用 `--debug-mode 6`，随后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  .codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py \
  /absolute/path/to/result-run
```

镜像不要求某个固定的 FFmpeg 小版本，但必须支持 H.264 `libx264` 编码、
`yuv420p` 像素格式和帧解码；本轮镜像已用实际 mode-6 视频完成帧数一致性校验。

### 更新和验证

仓库源码在构建时写入镜像；代码更新后应从宿主机工作区重新构建，不在容器内执行
自动 `git pull`。构建前可以检查 Dockerfile 和 Compose：

```bash
docker buildx build --check -f Dockerfile .
docker compose config --quiet
```

完整的构建、启动、目录挂载和解释器验证命令见 `docker/README.md`。

推送镜像前执行：

```bash
docker login ccr.ccs.tencentyun.com
docker push ccr.ccs.tencentyun.com/vtla/vtla:0.5
```
