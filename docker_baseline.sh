#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering}"
MODEL_DIR="${MODEL_DIR:-/sdb/duyuetian/models/VLM-CapCurriculum-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_DIR}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/sdb/duyuetian/.cache/huggingface}"

GPU_IDS="${GPU_IDS:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PROXY_PORT="${PROXY_PORT:-27890}"
VLLM_PORT="${VLLM_PORT:-8770}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.12.0}"

if [ ! -d "$PROJECT_ROOT/data" ]; then
  echo "Project data directory not found: $PROJECT_ROOT/data" >&2
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TENSOR_PARALLEL_SIZE must be a positive integer" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in GPU_IDS: $gpu_id" >&2
    exit 1
  fi
done
if [ "${#GPU_ID_ARRAY[@]}" -ne "$TENSOR_PARALLEL_SIZE" ]; then
  echo "GPU_IDS contains ${#GPU_ID_ARRAY[@]} GPUs, but TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE" >&2
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "GPU_IDS=$GPU_IDS"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "VLLM_PORT=$VLLM_PORT"
echo "VLLM_IMAGE=$VLLM_IMAGE"

docker run --rm \
  --name mllm-vllm-baseline \
  --gpus "\"device=${GPU_IDS}\"" \
  --ipc=host \
  --network=host \
  -e HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}" \
  -e HTTPS_PROXY="http://127.0.0.1:${PROXY_PORT}" \
  -e NO_PROXY=localhost,127.0.0.1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}:ro" \
  -v "${MODEL_DIR}:${MODEL_DIR}:ro" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --entrypoint vllm \
  "${VLLM_IMAGE}" \
  serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enforce-eager \
    --allowed-local-media-path "${PROJECT_ROOT}/data"
