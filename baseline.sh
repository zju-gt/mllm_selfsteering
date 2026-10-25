#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering}"
MODEL_DIR="${MODEL_DIR:-/sda/llm_weights/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_DIR}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8770/v1}"
EVAL_LIMIT="${EVAL_LIMIT:-1500}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-8}"
VLLM_VERSION="${VLLM_VERSION:-0.12.0}"
MODEL_REVISION="${MODEL_REVISION:-main}"
DATASETS_YAML="${DATASETS_YAML:-}"

BASELINE_ARGS=()
if [ -n "${1:-}" ]; then
  BASELINE_ARGS+=(--generation-dir "$1")
fi
if [ -n "$DATASETS_YAML" ]; then
  BASELINE_ARGS+=(--override "data.enabled_datasets=${DATASETS_YAML}")
fi

python scripts/run_baseline.py \
  --eval-limit "$EVAL_LIMIT" \
  "${BASELINE_ARGS[@]}" \
  --override "model.name=${SERVED_MODEL_NAME}" \
  --override "model.source=${MODEL_DIR}" \
  --override "model.revision=${MODEL_REVISION}" \
  --override experiment.generation.backend=vllm_hook \
  --override "experiment.vllm_hook.base_url=${VLLM_BASE_URL}" \
  --override "experiment.vllm_hook.server_media_root=${PROJECT_ROOT}/data" \
  --override "experiment.vllm_hook.request_concurrency=${REQUEST_CONCURRENCY}" \
  --override "experiment.vllm_hook.server_provenance.vllm_version=${VLLM_VERSION}" \
  --override "experiment.vllm_hook.server_provenance.served_model_revision=${MODEL_REVISION}"
