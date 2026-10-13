# MLLM Self-Steering

This project migrates `vector-make` from a text-only causal LM to multimodal LLMs,
initially Qwen3-VL-4B/8B-Instruct. It extracts six user-defined residual-stream
directions from controlled text or image contrasts and evaluates continuous steering
on MMStar, MMMU, MathVista, and MathVision.

There is no annotation API and no demand-scoring stage. Vector extraction consumes
only controlled contrast inputs; benchmark answers are used only after generation
for local evaluation.

## Benchmark configuration

| Project name | Hugging Face source | Requested/logical split | Actual source split |
|---|---|---|---|
| MMStar | `Lin-Chen/MMStar` | `test` | `val` (the only public official split) |
| MMMU | `MMMU/MMMU` | `val` | all 30 subject configs, `validation` |
| MathVista | `AI4Math/MathVista` | `testmini` | `testmini` |
| MathVision | `MathLLMs/MathVision` | `test` | `test` |

`mmvista` and `mmvision` are interpreted as MathVista and MathVision. Images are
materialized as ordinary PNG files, and canonical JSONL records contain one or more
absolute image paths.

## Pipeline

```text
00 benchmark download/canonicalization
  ↓
00b generate six evaluation-independent controlled contrast-pair files
  ↓
01 validate the six contrast-pair files
  ↓
02 capture positive - negative activation deltas with Transformers
  ↓
03 aggregate raw/unit/mean-norm vectors and export vLLM-Hook files
  ↓
04 measure within-capability coherence and cross-capability similarity
  ↓
05 run steered generation through Transformers or vLLM-Hook
  ↓
06 require a complete paired population, then score accuracy and flips
```

Run from this directory:

```bash
python -m pip install -e ".[dev]"
python scripts/00_prepare_data.py
python scripts/00_prepare_pairs.py
python scripts/01_validate_pairs.py
python scripts/02_capture_contrasts.py
python scripts/03_extract_vectors.py
python scripts/04_analyze_vectors.py
python scripts/05_run_steering.py
python scripts/06_score_generations.py
```

For a stage-by-stage Chinese operations guide, including inputs, outputs, smoke tests,
resume behavior, and acceptance checks, see
[docs/pipeline_guide_zh.md](docs/pipeline_guide_zh.md).

Use `--pair-limit N` to limit contrast pairs in stages 02-04 and `--eval-limit N`
to limit evaluation items per dataset in stages 05-06. Pass `--pair-limit N` to
stages 05-06 only when selecting vectors that were captured with that pair limit.
The legacy `--limit N` still applies one value to every scope. Limits are part of
artifact identities, so smoke artifacts cannot overwrite a formal run.

## Contrast pairs

Generate the six configured files with `python scripts/00_prepare_pairs.py`, or
replace them with reviewed manual pairs using the schemas in
[pairs/README.md](pairs/README.md).
The provisional taxonomy and proposed visual controls are described in
[docs/capability_design.md](docs/capability_design.md). The taxonomy lives entirely in
YAML and can be renamed or replaced without changing Python code, provided it remains
three text and three visual capabilities.

Text contrasts hold images fixed. Visual contrasts hold prompt text fixed. This is
validated before GPU work starts.

## Qwen3-VL

The default model is `Qwen/Qwen3-VL-4B-Instruct`; switch to the 8B model with:

```bash
python scripts/02_capture_contrasts.py \
  --override model.name=Qwen/Qwen3-VL-8B-Instruct
```

Pin `model.revision` to a commit SHA for formal experiments. `target_layer: 23` is a
provisional starting point, not a scientifically selected layer. Run a layer sweep or
pilot before fixing it for the paper.

## vLLM-Hook acceleration

Extraction uses Transformers because controlled paired capture is simpler and easier
to audit there. Generation can use IBM vLLM-Hook/MIA:

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook
```

See [docs/vllm_hook.md](docs/vllm_hook.md) for server setup and compatibility limits.
The generated `.pt` files follow the current MIA contract: `{"dir": vector}`. The
configured intervention uses `phase=both`, `positions=last_token`, and 0-based decoder
block indices, matching the Transformers reference hook.

The vLLM client is asynchronous and uses a configurable rolling request window so the
server can continuously batch generation. It sends Linux `file://` image URLs by
default, supplies content-derived media UUIDs for vLLM's multimodal cache, reuses each
unsteered baseline across capabilities, keeps vectors in the server-side MIA cache,
and writes resumable output checkpoints without an `fsync` per row. It also rejects a
server whose baseline and steered probe logprobs are identical. `model.max_new_tokens`
is still a single global value.

## Important evaluation boundaries

- MathVista and MathVision contain open-ended math answers. The included evaluator is
  deterministic normalized/numeric matching, not a complete symbolic equivalence
  checker or official submission evaluator.
- MMStar's official public dataset calls its split `val`; the logical `test` label in
  this project reflects the requested experimental role, not the upstream split name.
- Stage 06 refuses to score missing datasets, capabilities, items, or alpha values.
- A vLLM-Hook/Qwen3-VL GPU parity smoke test is required before treating the high-speed
  backend as equivalent to the reference implementation.
