#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering
MODEL_DIR=/sdc/wangyucheng/models/self-steering-7B
# MODEL_DIR="${MODEL_DIR:-/sda/llm_weights/Qwen3-VL-8B-Instruct}"
HF_CACHE_DIR=/sdb/duyuetian/.cache/huggingface

GPU_IDS="${GPU_IDS:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PROXY_PORT="${PROXY_PORT:-27890}"
LLM_PORT="${LLM_PORT:-8772}"
# Docker 容器名在同一台服务器上必须唯一；同一镜像可运行多个独立实例。
CONTAINER_NAME="${CONTAINER_NAME:-mllm-vllm-hook-table2-${LLM_PORT}}"

if [ ! -d "$PROJECT_ROOT/data" ]; then
  echo "Project data directory not found: $PROJECT_ROOT/data" >&2
  exit 1
fi

if [ ! -d "$PROJECT_ROOT/outputs/vectors" ]; then
  echo "Vector directory not found: $PROJECT_ROOT/outputs/vectors" >&2
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
echo "HF_CACHE_DIR=$HF_CACHE_DIR"
echo "GPU_IDS=$GPU_IDS"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "PROXY_PORT=$PROXY_PORT"
echo "LLM_PORT=$LLM_PORT"
echo "CONTAINER_NAME=$CONTAINER_NAME"

docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=${GPU_IDS}\"" \
  --ipc=host \
  --network=host \
  -e VLLM_HOOK_WORKER=steer \
  -e HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}" \
  -e HTTPS_PROXY="http://127.0.0.1:${PROXY_PORT}" \
  -e NO_PROXY=localhost,127.0.0.1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}:ro" \
  -v "${MODEL_DIR}:${MODEL_DIR}:ro" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  mllm-vllm-hook:v0.2-vllm0.12 \
  vllm serve "${MODEL_DIR}" \
    --served-model-name "${MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "${LLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --allowed-local-media-path "${PROJECT_ROOT}/data"
