# vLLM-Hook integration status

The project has two generation backends:

- `transformers`: reference implementation, used for extraction and parity checks.
- `vllm_hook`: OpenAI-compatible client for an IBM vLLM-Hook/MIA server.

Current vLLM-Hook documentation supports decoder names of the form
`language_model.model.layers.N`, explicitly covering multimodal wrappers, and supports
`phase=both` plus `positions=last_token`. This matches the reference intervention:
modify only the final position during prefill and every one-token cached decode step.

It is not yet evidence that every Qwen3-VL release works. Run a GPU parity smoke test
for each pinned Qwen3-VL and vLLM-Hook revision. Compare `alpha=0` output first, then a
small non-zero alpha, before launching a benchmark.

The upstream project currently validates its MIA branch with vLLM 0.29.0, PyTorch
2.13.0, Python 3.11-3.14, CUDA 13, and H100 hardware. Keep it in a dedicated
environment rather than installing it into the Transformers extraction environment.

## CUDA 12 legacy compatibility mode

Hosts that cannot run CUDA 13 can try the pinned legacy stack in
`docker/vllm_hook_v0_2`: IBM vLLM-Hook v0.2.0 on vLLM 0.12.0 and its CUDA 12.8
runtime. Select it on the client with:

```bash
--override experiment.vllm_hook.protocol=vllm_hook_v0_2
```

The compatibility mode translates the current `phase=both, positions=last_token`
semantics into v0.2's `apply_at_all_positions=true`: modify the last prompt token,
then each single-token cached decode step. It also supports `phase=prefill`. The old
worker cannot express decode-only or all-token steering, and configuration validation
rejects those combinations instead of silently changing the experiment.

This route is intentionally isolated in Docker. Do not install vLLM 0.12 into the
Transformers extraction environment. Driver 550 falls inside CUDA 12.x minor-version
compatibility, but JIT kernels can still expose driver/toolchain incompatibilities;
the Stage 05 log-probability effect probe remains mandatory. See the image-specific
build, launch, and smoke-test commands in
`docker/vllm_hook_v0_2/README.md`.

vLLM-Hook expects one PyTorch file per direction:

```python
{"dir": hidden_size_vector}
```

Stage 03 exports these files under `outputs/vectors/<capture_id>/vllm_hook/`.

## Linux server launch

For a same-host Linux deployment, launch the server with access to the canonical
image directory. `file://` media avoids base64-encoding every image for every alpha.
Do not add `--enforce-eager`: the steering hook supports CUDA graph execution, and
disabling it gives up part of the intended acceleration.

```bash
export PROJECT_ROOT=/absolute/path/to/mllm-self-steering

VLLM_WORKER_MULTIPROC_METHOD=spawn \
MIA_WORKER=steer \
  vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 32768 \
  --port 8770 \
  --gpu-memory-utilization 0.8 \
  --allowed-local-media-path "$PROJECT_ROOT/data"
```

The command above is for current MIA only. The legacy image uses
`VLLM_HOOK_WORKER=steer`, which is already set in its Dockerfile.

Then run the client from the project directory:

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.request_concurrency=16
```

The client keeps a rolling window of asynchronous requests, which allows vLLM's
continuous batching to combine different items, alpha values, and steering vectors.
It also reuses one baseline generation across all six capabilities. Tune concurrency
to the GPU and request lengths; 8, 16, and 32 are sensible measurements for a pilot.
`model.max_new_tokens` remains one global setting for every dataset and answer type.

MIA caches loaded direction tensors on the server GPU. One run uses six vector files
for the selected scaling. Restart a long-lived server between unrelated vector
libraries if GPU memory usage grows unexpectedly.

Before scheduling the full matrix, the client makes one baseline and one steered
single-token request and compares their first-token log probabilities. Identical
results fail fast because they usually mean that MIA accepted but ignored the hook
configuration (for example, no decoder layer matched). This diagnostic does not
change the global `model.max_new_tokens` used for benchmark generation. It can be
disabled with `verify_steering_effect=false` only when the server cannot return chat
logprobs and steering has been verified independently.

For a formal run, record the exact server implementation in configuration so it is
part of the generation run identity:

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.server_provenance.vllm_version=0.x.y \
  --override experiment.vllm_hook.server_provenance.mia_revision=GIT_SHA \
  --override experiment.vllm_hook.server_provenance.served_model_revision=MODEL_SHA
```

## Container or remote path mapping

The steering vector path is interpreted by the server process, not by the client. If
the client and server see different mount points, map both roots explicitly:

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.server_media_root=/workspace/data \
  --override experiment.vllm_hook.server_vector_root=/workspace/outputs/vectors
```

Here the local `data_dir` maps to `/workspace/data`, and the local
`outputs/vectors` directory maps to `/workspace/outputs/vectors`. Mount those exact
directories into the vLLM container and set `--allowed-local-media-path
/workspace/data` on the server. The mapper rejects paths outside the configured local
roots instead of silently sending unusable host paths.

Prepared benchmark images are always copied below `data/images`, including images
referenced by an already-canonical local JSONL file. Each request also carries the
image SHA-256 as its media UUID so vLLM can reuse its multimodal input cache instead
of hashing the same image again for every capability and alpha.

If the server cannot access a shared image filesystem, use
`experiment.vllm_hook.media_transport=data_url`. This is compatible with a remote
server but increases client CPU work, request size, and network traffic.

Pipeline parallelism is not supported by vLLM-Hook; tensor parallelism is supported
by its steering worker. Before a long run, the client checks the `/models` endpoint
and performs the steering-effect probe described above. During generation it
checkpoints JSONL in batches and aborts after repeated request failures, so a bad
mount or endpoint does not produce hundreds of thousands of doomed calls. On resume,
only an unterminated trailing JSONL fragment left by an abrupt process kill is
discarded; corruption in completed records remains a hard error.
