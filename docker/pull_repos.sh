#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: pull_repos.sh [--dry-run] [--allow-dirty]

Validate and fast-forward the pinned runtime repositories from origin.
IsaacLab and Isaac Sim are intentionally not updated by this script.
By default, dirty worktrees are rejected. --allow-dirty permits a normal
fast-forward pull while still refusing force-reset or non-fast-forward merges.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[pull-repos] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

declare -a REPO_SPECS=(
  "T2-VLA|rl|git@github.com:NathanWu7/T2-VLA.git"
  "Tabero_X|rl|git@github.com:xiangixn/Tabero_X.git"
  "RLinf|tabero|git@github.com:Demonmasterlqx/RLinf.git"
)

echo "[pull-repos] root: $ROOT"

for spec in "${REPO_SPECS[@]}"; do
  IFS='|' read -r name branch remote <<<"$spec"
  repo="$ROOT/$name"

  [[ -d "$repo/.git" || -f "$repo/.git" ]] || {
    echo "[pull-repos] missing Git repository: $repo" >&2
    exit 1
  }

  actual_remote="$(git -C "$repo" remote get-url origin)"
  [[ "$actual_remote" == "$remote" ]] || {
    echo "[pull-repos] $name origin mismatch: expected $remote, got $actual_remote" >&2
    exit 1
  }

  actual_branch="$(git -C "$repo" branch --show-current)"
  [[ "$actual_branch" == "$branch" ]] || {
    echo "[pull-repos] $name branch mismatch: expected $branch, got $actual_branch" >&2
    exit 1
  }

  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    if ((ALLOW_DIRTY)); then
      echo "[pull-repos] warning: allowing dirty worktree: $name"
    else
      echo "[pull-repos] refusing to update dirty worktree: $name" >&2
      echo "[pull-repos] rerun with --allow-dirty only if local changes are preserved" >&2
      exit 1
    fi
  fi

  echo "[pull-repos] $name: origin=$actual_remote branch=$actual_branch clean"
done

if ((DRY_RUN)); then
  echo "[pull-repos] dry-run complete; no fetch or checkout was performed"
  exit 0
fi

for spec in "${REPO_SPECS[@]}"; do
  IFS='|' read -r name branch _ <<<"$spec"
  repo="$ROOT/$name"
  echo "[pull-repos] fetching $name/$branch"
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" pull --ff-only origin "$branch"
done

echo "[pull-repos] all repositories updated successfully"
