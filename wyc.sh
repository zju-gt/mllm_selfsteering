#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2

MODE="${1:-}"

case "$MODE" in
  prepare)
    PREPARATION_ARGS=()
    if [ -n "${2:-}" ]; then
      PREPARATION_ARGS+=(--preparation-dir "$2")
    fi
    python scripts/prepare_experiment.py \
      --pair-limit 128 \
      "${PREPARATION_ARGS[@]}"
    ;;

  generate)
    if [ -z "${2:-}" ]; then
      echo "Usage: bash wyc.sh generate <preparation-dir> [generation-dir]" >&2
      exit 2
    fi
    GENERATION_ARGS=(--preparation-dir "$2")
    if [ -n "${3:-}" ]; then
      GENERATION_ARGS+=(--generation-dir "$3")
    fi
    python scripts/07_run_adaptive_steering.py \
      --baseline-no-answer-only \
      "${GENERATION_ARGS[@]}" \
      --override experiment.generation.backend=vllm_hook \
      --override experiment.vllm_hook.protocol=vllm_hook_v0_2 \
      --override experiment.vllm_hook.base_url=http://127.0.0.1:8770/v1 \
      --override experiment.vllm_hook.server_media_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/data \
      --override experiment.vllm_hook.server_vector_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/outputs/vectors \
      --override experiment.vllm_hook.server_provenance.vllm_version=0.12.0 \
      --override experiment.vllm_hook.server_provenance.mia_revision=0e34fddf2c234a50c9c820c2b7852472f8f4c51c \
      --override experiment.adaptive_steering.selector_concurrency=8 \
      --override experiment.adaptive_steering.generation_concurrency=8
    ;;

  *)
    cat >&2 <<'EOF'
Usage:
  bash wyc.sh prepare [preparation-dir]
  bash wyc.sh generate <preparation-dir> [generation-dir]

Omit generation-dir to start a new timestamped generation run.
Pass an existing generation-dir to resume it.
EOF
    exit 2
    ;;
esac
