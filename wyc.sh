export CUDA_VISIBLE_DEVICES=1,2

python scripts/01_validate_pairs.py \
  --pair-limit 128 \
  --eval-limit 1500 \

python scripts/02_capture_contrasts.py \
  --pair-limit 128 \
  --eval-limit 1500 \

python scripts/03_extract_vectors.py \
  --pair-limit 128 \
  --eval-limit 1500 \

python scripts/04_analyze_vectors.py \
  --pair-limit 128 \
  --eval-limit 1500 \

# python scripts/05_run_steering.py \
#   --pair-limit 128 \
#   --eval-limit 1500 \
#   --override experiment.generation.backend=vllm_hook \
#   --override experiment.vllm_hook.protocol=vllm_hook_v0_2 \
#   --override experiment.vllm_hook.base_url=http://127.0.0.1:8770/v1 \
#   --override experiment.vllm_hook.request_concurrency=1 \
#   --override experiment.vllm_hook.server_media_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/data \
#   --override experiment.vllm_hook.server_vector_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/outputs/vectors \
#   --override experiment.vllm_hook.server_provenance.vllm_version=0.12.0 \
#   --override experiment.vllm_hook.server_provenance.mia_revision=0e34fddf2c234a50c9c820c2b7852472f8f4c51c

# python scripts/06_score_generations.py \
#   --pair-limit 128 \
#   --eval-limit 1500 \
#   --override experiment.generation.backend=vllm_hook \
#   --override experiment.vllm_hook.protocol=vllm_hook_v0_2 \
#   --override experiment.vllm_hook.base_url=http://127.0.0.1:8770/v1 \
#   --override experiment.vllm_hook.request_concurrency=1 \
#   --override experiment.vllm_hook.server_media_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/data \
#   --override experiment.vllm_hook.server_vector_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/outputs/vectors \
#   --override experiment.vllm_hook.server_provenance.vllm_version=0.12.0 \
#   --override experiment.vllm_hook.server_provenance.mia_revision=0e34fddf2c234a50c9c820c2b7852472f8f4c51c

python scripts/07_run_adaptive_steering.py \
  --pair-limit 128 \
  --eval-limit 1500 \
  --baseline-no-answer-only \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.protocol=vllm_hook_v0_2 \
  --override experiment.vllm_hook.base_url=http://127.0.0.1:8770/v1 \
  --override experiment.vllm_hook.server_media_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/data \
  --override experiment.vllm_hook.server_vector_root=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering/outputs/vectors \
  --override experiment.vllm_hook.server_provenance.vllm_version=0.12.0 \
  --override experiment.vllm_hook.server_provenance.mia_revision=0e34fddf2c234a50c9c820c2b7852472f8f4c51c \
  --override experiment.adaptive_steering.selector_concurrency=8 \
  --override experiment.adaptive_steering.generation_concurrency=8