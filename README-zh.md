# VTLA-RL

本工作区把 IsaacLab、Tabero_X、RLinf 和 T2-VLA 作为一个可复现的研究环境管理。
宿主机和 Docker 镜像使用相同的仓库分支约定；NVIDIA 驱动始终由宿主机提供。

## 部署

### uv

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

进入 IsaacLab/Tabero_X 开发 shell 时执行：

```bash
source activate_uv.sh
python -c 'import torch; print(torch.__version__, torch.__file__)'
```

脚本会加载工作区 `.env`（如果存在）、设置 Isaac Sim 的 Kit 运行时变量，并把
`VIRTUAL_ENV` 指向 `IsaacLab/.venv`。`RLinf/.venv` 和
`T2-VLA/.venv` 不需要切换到共享环境，直接调用对应的绝对路径即可：

```bash
RLinf/.venv/bin/python -u RLinf/examples/embodiment/train_embodied_agent.py ...
T2-VLA/.venv/bin/python T2-VLA/scripts/serve_policy.py ...
```

Tabero_X 的 Taxim GPU 后端使用共享环境中的 JAX CUDA 12（0.5.3）。已有环境只需要
执行 `source activate_uv.sh`；首次配置或依赖修复优先复用本地 uv 缓存：

```bash
uv venv --seed --python 3.11 IsaacLab/.venv
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

#### 构建镜像

默认镜像为 `ccr.ccs.tencentyun.com/vtla/vtla:0.3`，基础镜像为 Isaac Sim 5.1，
并在 `/root/Tabero` 中放置 IsaacLab、T2-VLA、Tabero_X、RLinf 及共享的 Isaac Sim 链接。
构建时可以把宿主机 uv 缓存作为 BuildKit additional context：

```bash
UV_CACHE_HOST="${UV_CACHE_HOST:-$HOME/.cache/uv}" docker compose build
```

镜像中的环境布局与宿主机一致：IsaacLab/Tabero_X 使用
`/root/Tabero/IsaacLab/.venv`，RLinf 和 T2-VLA 各自使用仓库下的 `.venv`。
容器入口会自动 source 外挂的 `/root/Tabero/.env`；不要将 HF 或 W&B token 写入
Dockerfile、镜像层或 Git。

`.env` 至少包含以下两个键（文件应放在宿主机并限制为 `chmod 600`）：

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

#### 启动容器

```bash
docker compose run --rm shell
docker compose run --rm t2-vla
docker compose run --rm rlinf
```

Compose 将宿主机的 `datasets`、`models`、`results`、`Record` 分别挂载到
`/root/Tabero` 下对应目录；前两者只读，后两者可写。也可以显式指定凭据文件：

```bash
TABERO_ENV_FILE=/secure/path/vtla.env docker compose run --rm shell
```

`docker compose build` 需要 BuildKit；`UV_CACHE_HOST` 指向宿主机 uv 缓存目录，
不设置时默认使用 `$HOME/.cache/uv`。如果只想检查 Dockerfile
语法而不构建镜像，可运行：

```bash
docker buildx build --check \
  --build-context uv-cache="$HOME/.cache/uv" \
  -f Dockerfile .
```

直接 `docker run` 时需使用 `--gpus all`、`--ipc host`，并外挂 `.env` 和上述四个目录。
镜像不包含宿主机 NVIDIA 驱动，运行前需安装 NVIDIA Container Toolkit。

#### 仓库更新和验证

容器内更新脚本会校验远程仓库和分支，只进行 fast-forward：

```bash
/root/Tabero/docker/pull_repos.sh --dry-run
/root/Tabero/docker/pull_repos.sh
docker compose run --rm shell validate
```

当前固定分支为 IsaacLab `release/2.3.0`、Tabero_X/T2-VLA `rl`、RLinf `tabero`。
推送镜像前执行：

```bash
docker login ccr.ccs.tencentyun.com
docker push ccr.ccs.tencentyun.com/vtla/vtla:0.3
```
