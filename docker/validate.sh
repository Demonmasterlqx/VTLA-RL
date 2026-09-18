#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/Tabero

echo "[validate] repository branches and origins"
for spec in \
  "IsaacLab|release/2.3.0|git@github.com:isaac-sim/IsaacLab.git" \
  "T2-VLA|rl|git@github.com:NathanWu7/T2-VLA.git" \
  "Tabero_X|rl|git@github.com:xiangixn/Tabero_X.git" \
  "RLinf|tabero|git@github.com:Demonmasterlqx/RLinf.git"; do
  IFS='|' read -r repo branch remote <<<"$spec"
  test "$(git -C "$ROOT/$repo" branch --show-current)" = "$branch"
  test "$(git -C "$ROOT/$repo" remote get-url origin)" = "$remote"
  echo "  $repo: $branch ($remote)"
done

test -x "$ROOT/isaacsim/python.sh"
test -x "$ROOT/activate_uv.sh"
test "$(readlink -f "$ROOT/IsaacLab/_isaac_sim")" = "$(readlink -f "$ROOT/isaacsim")"
test "$(readlink -f "$ROOT/RLinf/isaac_sim")" = "$(readlink -f "$ROOT/isaacsim")"

echo "[validate] uv environments"
for python in "$ROOT/IsaacLab/.venv/bin/python" "$ROOT/RLinf/.venv/bin/python" "$ROOT/T2-VLA/.venv/bin/python"; do
  test -x "$python"
  "$python" -c 'import sys; print(sys.executable)'
done
"$ROOT/IsaacLab/.venv/bin/python" - <<'PY'
import importlib.util
import jax
import torch
assert torch.__version__.startswith("2.7."), torch.__version__
assert jax.__version__.startswith("0.5.3"), jax.__version__
for package in ("isaaclab", "isaaclab_tasks", "tac_manip", "torch_scatter"):
    assert importlib.util.find_spec(package), package
print("IsaacLab/Tabero_X uv imports: ok")
PY

echo "[validate] CUDA visibility"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[validate] static checks passed"
