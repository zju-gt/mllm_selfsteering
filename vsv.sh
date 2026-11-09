#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash scripts/run_vsv_full.sh mathvision
# bash scripts/run_vsv_full.sh mathvista
# bash scripts/run_vsv_full.sh mmstar
# bash scripts/run_vsv_full.sh mmmu
#
# "mmlu" 会自动按 "mmmu" 处理。

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {mathvision|mathvista|mmstar|mmmu}"
  exit 1
fi

DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')

if [[ "$DATASET" == "mmlu" ]]; then
  DATASET="mmmu"
fi

case "$DATASET" in
  mathvision)
    GPU_ID=1
    ;;
  mmstar)
    GPU_ID=2
    ;;
  mathvista)
    GPU_ID=3
    ;;
  mmmu)
    GPU_ID=4
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Supported datasets: mathvision, mathvista, mmstar, mmmu"
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"
mkdir -p outputs/baselines

RUN_DIR="outputs/baselines/vista_vsv_${DATASET}"

echo "Dataset : $DATASET"
echo "GPU     : $GPU_ID"
echo "Output  : $RUN_DIR"
echo

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python mllm_selfsteering/scripts/run_vista_vsv_baseline.py \
    --vsv-lambda 0.1 \
    --override "data.enabled_datasets=[$DATASET]" \
    --run-dir "$RUN_DIR"