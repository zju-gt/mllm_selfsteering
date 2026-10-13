# Legacy vLLM-Hook on CUDA 12.x

This image pins the last pre-MIA protocol used by this project:

- IBM vLLM-Hook `v0.2.0` (`0e34fddf2c234a50c9c820c2b7852472f8f4c51c`)
- vLLM `0.12.0`
- PyTorch `2.9.0`
- CUDA `12.8` user-space runtime from the vLLM image

It is intended for an A100 host whose driver cannot run CUDA 13. CUDA 12 minor
compatibility makes driver 550 a plausible target, but the final result must be
proved on the target host because JIT-compiled kernels can still require newer
driver functionality.

Build from the project root:

```bash
docker build \
  -f docker/vllm_hook_v0_2/Dockerfile \
  -t mllm-vllm-hook:v0.2-vllm0.12 .
```

Check that the image sees an A100 before downloading the model:

```bash
docker run --rm --gpus '"device=1"' \
  --entrypoint python3 \
  mllm-vllm-hook:v0.2-vllm0.12 \
  -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); print(torch.cuda.get_device_name(0))'
```

Start the server:

```bash
export PROJECT_ROOT=/sdb/duyuetian/iclr_selfsteering/mllm_selfsteering
export HF_CACHE_DIR=/sdb/duyuetian/.cache/huggingface

docker run --rm --name mllm-vllm-hook \
  --gpus '"device=1"' \
  --ipc=host \
  -p 8770:8770 \
  -v "$PROJECT_ROOT:/workspace/mllm_selfsteering" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  mllm-vllm-hook:v0.2-vllm0.12 \
  vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --served-model-name Qwen/Qwen3-VL-4B-Instruct \
    --host 0.0.0.0 \
    --port 8770 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --allowed-local-media-path /workspace/mllm_selfsteering/data
```

The server log must say that steering hooks were installed on decoder layers.
Then run a one-item end-to-end effect probe from the host environment:

```bash
conda activate mllm-steer
python scripts/05_run_steering.py \
  --eval-limit 1 \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.protocol=vllm_hook_v0_2 \
  --override experiment.vllm_hook.request_concurrency=1 \
  --override experiment.vllm_hook.server_media_root=/workspace/mllm_selfsteering/data \
  --override experiment.vllm_hook.server_vector_root=/workspace/mllm_selfsteering/outputs/vectors \
  --override experiment.vllm_hook.server_provenance.vllm_version=0.12.0 \
  --override experiment.vllm_hook.server_provenance.mia_revision=0e34fddf2c234a50c9c820c2b7852472f8f4c51c
```

Do not disable `verify_steering_effect`: it compares baseline and steered
first-token log-probabilities and fails if the old plugin accepted but ignored
the request. If extraction used `--pair-limit N`, add the same option here.
