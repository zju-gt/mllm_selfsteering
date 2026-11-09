#!/usr/bin/env bash
set -euo pipefail

# 在 tmux 中运行：bash table2.sh [single|all|random|text|visual|self|none] [Python脚本参数...]
# 常用默认值集中在这里；也可使用同名环境变量覆盖。
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARATION_DIR="${PREPARATION_DIR:-outputs/preparation_1105_0414}"
SUBSET_FRACTION="${SUBSET_FRACTION:-0.3}"
SUBSET_SEED="${SUBSET_SEED:-42}"
RANDOM_SEED="${RANDOM_SEED:-36}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8773/v1}"
CONCURRENCY="${CONCURRENCY:-8}"
GLOBAL_SCALE="${GLOBAL_SCALE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
ROUTING_CACHE_DIR="${ROUTING_CACHE_DIR:-outputs/table2_routing}"

# 留空时只运行子集；指定基准文件才生成30%实测 + 70%复用的混合结果。
MERGE_BASE="${MERGE_BASE:-}"
# 留空时新建带时间戳目录；续跑时指定已有目录，并保持实验参数一致。
RUN_DIR="${RUN_DIR:-}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash table2.sh [single|all|random|text|visual|self|none] [extra arguments...]

Defaults: single, MathVision test (3040), subset 0.3, subset seed 42,
          random seed 36, concurrency 8, global scale 1.0, max tokens 8192.
Activate conda environment mllm-steer before running; start the trained model server first.

Examples:
  bash table2.sh single --dry-run
  bash table2.sh visual --merge-base /path/to/base.jsonl
  MERGE_BASE=/path/to/base.jsonl bash table2.sh random
  SUBSET_FRACTION=1.0 bash table2.sh self
  bash table2.sh single --run-dir outputs/table2_single_MMDD_HHMM

Defaults can be changed above, through environment variables, or through extra
arguments (passed last). MathVision defaults to test; use the Python script's
--override data.sources.mathvision.* options to explicitly select testmini.
EOF
  exit 0
fi

MODE="${MODE:-single}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  MODE="$1"
  shift
fi
case "$MODE" in
  single|all|random|text|visual|self|none) ;;
  *) echo "Unknown mode: $MODE (see bash table2.sh --help)" >&2; exit 2 ;;
esac

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

ARGS=(
  "$MODE"
  --preparation-dir "$PREPARATION_DIR"
  --subset-fraction "$SUBSET_FRACTION"
  --subset-seed "$SUBSET_SEED"
  --random-seed "$RANDOM_SEED"
  --base-url "$BASE_URL"
  --concurrency "$CONCURRENCY"
  --routing-cache-dir "$ROUTING_CACHE_DIR"
  --override "model.max_new_tokens=$MAX_NEW_TOKENS"
  --override "experiment.adaptive_steering.global_scale=$GLOBAL_SCALE"
  --override "experiment.vllm_hook.protocol=vllm_hook_v0_2"
  --override "experiment.vllm_hook.server_media_root=$PROJECT_ROOT/data"
  --override "experiment.vllm_hook.server_vector_root=$PROJECT_ROOT/outputs/vectors"
)
if [[ -n "$MERGE_BASE" ]]; then
  ARGS+=(--merge-base "$MERGE_BASE")
fi
if [[ -n "$RUN_DIR" ]]; then
  ARGS+=(--run-dir "$RUN_DIR")
fi

exec "$PYTHON_BIN" scripts/08_various_steering.py "${ARGS[@]}" "$@"
