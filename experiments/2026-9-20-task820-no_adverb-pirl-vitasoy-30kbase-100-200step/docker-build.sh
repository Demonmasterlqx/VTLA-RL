#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
DOCKERFILE="${EXPERIMENT_DIR}/Dockerfile"
IMAGE_NAME="${TASK820_IMAGE:-ccr.ccs.tencentyun.com/vtla/vtla:2026-9-20-task820-no-adverb-pirl-vitasoy-30kbase-100-200step-local}"

cd "${WORKSPACE_ROOT}"

echo "Building ${IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Context:    ${WORKSPACE_ROOT}"

docker build \
  --pull=false \
  --progress=plain \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}" \
  "${WORKSPACE_ROOT}"
