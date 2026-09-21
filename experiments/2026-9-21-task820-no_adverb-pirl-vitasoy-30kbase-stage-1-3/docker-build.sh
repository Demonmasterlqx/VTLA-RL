#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
DOCKERFILE="${EXPERIMENT_DIR}/Dockerfile"
IMAGE_NAME="${TASK820_IMAGE:-ccr.ccs.tencentyun.com/vtla/vtla:2026-9-21-task820-no_adverb-pirl-vitasoy-30kbase-stage-1-3}"
BASE_IMAGE="${TASK820_BASE_IMAGE:-ccr.ccs.tencentyun.com/vtla/vtla:0.4}"

cd "${WORKSPACE_ROOT}"
echo "Building ${IMAGE_NAME}"
echo "Base:       ${BASE_IMAGE}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Context:    ${WORKSPACE_ROOT}"

docker build \
  --pull=false \
  --progress=plain \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}" \
  "$@" \
  "${WORKSPACE_ROOT}"
