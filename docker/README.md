# VTLA Docker 环境

镜像目标名：

```text
ccr.ccs.tencentyun.com/vtla/vtla:0.3
```

镜像包含 `/root/Tabero` 下的 IsaacLab、T2-VLA、Tabero_X、RLinf 和
Isaac Sim 二进制链接。IsaacLab 与 Tabero_X 共用 `IsaacLab/.venv`，其他
Python 环境由 uv 管理。GPU 驱动由宿主机 NVIDIA Container Toolkit 提供，
不会写入镜像。

## 外挂凭据

不要把真实 token 写进 Dockerfile、镜像层或 Git。复制
`docker/.env.example` 到宿主机安全位置，例如：

```bash
cp docker/.env.example /secure/path/vtla.env
chmod 600 /secure/path/vtla.env
```

填写：

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

入口会在每次容器启动时自动 source `/root/Tabero/.env`。Compose 默认挂载
当前目录的 `.env`；使用其他文件时设置 `TABERO_ENV_FILE`：

```bash
TABERO_ENV_FILE=/secure/path/vtla.env docker compose run --rm shell
```

该挂载是只读的，token 不会进入镜像。

## 构建和推送

构建时可复用宿主机 uv cache：

```bash
UV_CACHE_HOST="${UV_CACHE_HOST:-$HOME/.cache/uv}" \
docker compose build
```

推送前确认登录了腾讯云容器镜像服务：

```bash
docker login ccr.ccs.tencentyun.com
docker push ccr.ccs.tencentyun.com/vtla/vtla:0.3
```

也可以覆盖镜像名进行本地测试：

```bash
TABERO_IMAGE=tabero:local docker compose run --rm shell
```

## 启动

```bash
docker compose run --rm shell
docker compose run --rm t2-vla
docker compose run --rm rlinf
```

Compose 会把宿主机的 `datasets`、`models`、`results` 和 `Record` 挂载到
容器对应目录。`datasets`、`models` 为只读，`results`、`Record` 可写。
交互式 `shell` 会自动加载 `/root/Tabero/activate_uv.sh`，因此可直接使用
共享的 `IsaacLab/.venv`；`rlinf`、`t2-vla` 命令则显式使用各自仓库的 `.venv`。

直接使用 `docker run` 时需要显式挂载 `.env` 和四个数据目录：

```bash
docker run --rm -it --gpus all --network host --ipc host \
  --env-file /secure/path/vtla.env \
  -v /secure/path/vtla.env:/root/Tabero/.env:ro \
  -v "$PWD/datasets:/root/Tabero/datasets:ro" \
  -v "$PWD/models:/root/Tabero/models:ro" \
  -v "$PWD/results:/root/Tabero/results" \
  -v "$PWD/Record:/root/Tabero/Record" \
  ccr.ccs.tencentyun.com/vtla/vtla:0.3 shell
```

## 更新 Git 仓库

镜像内的 `docker/pull_repos.sh` 只更新以下分支：

| 仓库 | 分支 |
| --- | --- |
| T2-VLA | `rl` |
| Tabero_X | `rl` |
| RLinf | `tabero` |

IsaacLab 不由该脚本更新；Isaac Sim 是基础镜像中的二进制，也不是 Git
仓库。更新前脚本会拒绝脏工作区、错误分支或错误 origin：

```bash
/root/Tabero/docker/pull_repos.sh --dry-run
/root/Tabero/docker/pull_repos.sh
# The image may contain local environment patches. Explicitly opt in if those
# worktrees are intentionally retained:
/root/Tabero/docker/pull_repos.sh --allow-dirty
```

拉取需要对应 Git 远程的网络和认证；脚本只执行 fast-forward 更新，不会
强制覆盖本地修改。默认拒绝脏工作区；`--allow-dirty` 只在明确确认本地
修改可以保留时使用。

## 验证

```bash
docker compose run --rm shell validate
```

验证会检查仓库分支和 origin、Isaac Sim 链接、uv Python 环境以及 GPU
可见性。
