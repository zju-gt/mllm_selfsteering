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

07 optional main experiment: route six capability weights, compose one vector,
   then run paired baseline/adaptive inference through vLLM-Hook
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

The adaptive main experiment uses separate, immutable preparation and resumable
generation directories:

```bash
python scripts/prepare_experiment.py --pair-limit 128

python scripts/07_run_adaptive_steering.py \
  --preparation-dir outputs/preparation_1021_1654 \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.protocol=vllm_hook_v0_2
```

Omitting `--generation-dir` always creates a new `outputs/generation_MMDD_HHMM`
directory. Pass an existing directory to resume it:

```bash
python scripts/07_run_adaptive_steering.py \
  --preparation-dir outputs/preparation_1021_1654 \
  --generation-dir outputs/generation_1021_1730 \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.protocol=vllm_hook_v0_2
```

Both directories contain a `config.json` snapshot. Resume validates the preparation,
semantic generation settings, evaluation data, and tensor-content vector fingerprints
before reusing checkpoints. Operational concurrency may be changed while resuming.

The repository helper exposes the same lifecycle explicitly:

```bash
bash wyc.sh prepare
bash wyc.sh generate outputs/preparation_1021_1654
bash wyc.sh generate outputs/preparation_1021_1654 outputs/generation_1021_1730
```

The second command starts a new generation; the third resumes the named generation.
Existing pre-refactor outputs can still use `--legacy-layout`; it preserves the old
file-hash/run-ID behavior and must not be combined with either directory option.

For an unsteered base-model-only evaluation, start the standalone vLLM service and
run the resumable baseline entry point. It does not load a preparation directory or
any steering vectors:

```bash
MODEL_DIR=/sda/llm_weights/Qwen3-VL-8B-Instruct bash docker_baseline.sh
bash baseline.sh
```

The second command creates `outputs/baseline_MMDD_HHMM/` containing `config.json`,
`baselines.jsonl`, and `metrics.json`. Resume an interrupted run by passing that
directory explicitly:

```bash
bash baseline.sh outputs/baseline_1025_1530
```

After generation completes, calculate paired baseline/steering metrics and standard
dataset subdivisions with:

```bash
python scripts/split_metrics.py outputs/generation_1021_1730
```

This writes `split_metrics.json` inside the generation directory. MathVista and
MathVision report OE/MC/ALL; MMStar reports Perception/Reasoning/ALL; MMMU reports
STEM/HASS/ALL. Every group includes accuracies, delta, paired flips, and counts.

Start with `--eval-limit 1`. Round 1 scores six capabilities independently in
`[0,1]` at temperature `0.6`, then normalizes nonzero scores to sum to 1.
All-zero scores remain unsteered. Round 2 composes the corresponding Stage 03
vectors and performs deterministic paired baseline/adaptive generation. See
[docs/adaptive_two_round_steering_design_zh.md](docs/adaptive_two_round_steering_design_zh.md)
for the prompt contract, configuration, outputs, resumption behavior, and metrics.

To steer only when the baseline has no parseable answer, add
`--baseline-no-answer-only`. This mode checkpoints `baselines.jsonl` first and routes
only unanswered items. `selections.jsonl` and `generations.jsonl` are checkpointed in
the same generation directory.

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
For driver-550/CUDA-12 hosts, the repository also provides the pinned experimental
image and protocol adapter under
[`docker/vllm_hook_v0_2`](docker/vllm_hook_v0_2/README.md). Keep
`verify_steering_effect=true`; a successful HTTP response alone does not prove that
the legacy hook modified model activations.
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
