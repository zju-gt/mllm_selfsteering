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

## Table 2: frozen-model steering strategies

Activate the project environment (`conda activate mllm-steer`) and run from this
directory. Stage 08 defaults to `outputs/preparation_1105_0414` and inherits its
checkpoint (`/sdc/wangyucheng/models/self-steering-7B`), six-vector order, target
layer, and vector scaling. It disables baseline-no-answer gating and evaluates
every item with a paired baseline from the same checkpoint.

Start a matching server in a tmux window:

```bash
MODEL_DIR=/sdc/wangyucheng/models/self-steering-7B bash docker.sh
```

After the server is ready, run one mode in another tmux window:

```bash
python scripts/08_various_steering.py single
```

For defaults collected in one editable shell script, use `table2.sh`:

```bash
conda activate mllm-steer
bash table2.sh single --dry-run
bash table2.sh single
bash table2.sh random --merge-base /path/to/base.jsonl

# Environment variables also override defaults; extra CLI arguments take priority.
MERGE_BASE=/path/to/base.jsonl SUBSET_FRACTION=0.3 bash table2.sh visual
SUBSET_FRACTION=1.0 bash table2.sh self
bash table2.sh single --run-dir outputs/table2_single_MMDD_HHMM
```

The shell defaults to `single`, 30% per dataset, sampling seed 42, random seed
36, concurrency 8, global scale 1.0, max new tokens 8192, and the trained bank
`outputs/preparation_1105_0414`. MathVision defaults to the 3040-item test in
both entry points. Set `MERGE_BASE` or pass `--merge-base` for hybrid outputs;
without a reference, the shell runs only the subset. Edit the defaults at the
top of `table2.sh`, or set their environment variables, including `BASE_URL`,
`PREPARATION_DIR`, `RUN_DIR`, `ROUTING_CACHE_DIR` and `PYTHON_BIN`. Relative paths
are resolved from the project root, even when the shell is invoked elsewhere.
The shell expects an active project Python environment and a running model
server. Direct Python invocation retains its default full evaluation fraction.

| Mode | Weights |
|---|---|
| `none` | All zero; no steering or routing requests. |
| `single` | First vector in the preparation config (`text_logic`): 1; others: 0. |
| `all` | All six weights are exactly `1/6`. |
| `random` | Independent per-item Dirichlet(1,1,1,1,1,1) weights, summing to 1. |
| `text` | Route all six, remove visual weights, renormalize the remainder. |
| `visual` | Route all six, remove textual weights, renormalize the remainder. |
| `self` | Full adaptive six-capability routing. |

Text/visual masks preserve zero-vector bypass when no retained weight is positive.
`self`, `text`, and `visual` share original routing results under
`outputs/table2_routing/`, keyed by configuration, vector contents, and item
identity. Retain this cache when comparing these modes. Fixed and random modes
do not call the router. Random weights are independent of processing order;
`--random-seed` changes weights without changing the solver's seed.

```bash
# Check model, vectors, actual input files and item counts without an API call.
python scripts/08_various_steering.py single --dry-run

# One item per selected dataset, then an independent random replicate.
python scripts/08_various_steering.py all --datasets mmstar mathvista --eval-limit 1
python scripts/08_various_steering.py random --random-seed 37

# Resume: repeat the same mode, seed, dataset and semantic overrides.
python scripts/08_various_steering.py single --run-dir outputs/table2_single_MMDD_HHMM

# Change endpoint or concurrency; --mode single is also accepted.
python scripts/08_various_steering.py --mode single --base-url http://127.0.0.1:8770/v1 --concurrency 8
```

Each run creates `outputs/table2_<mode>_MMDD_HHMM/` with `config.json`,
`selections.jsonl`, `generations.jsonl`, `metrics.json`, `metrics.csv` and
`table2.csv`. The Table 2 CSV uses percentage accuracies and the macro-average
across enabled datasets. Incomplete runs are marked as such, have no final Avg,
and exit with an error so they can be resumed. Changing mode or random seed in
an existing run directory is rejected.

Stage 08 defaults MathVision to the **3040-item test** in `mathvision.jsonl`
before applying explicit CLI overrides. Select the same population for every
Table 2 row. To explicitly use the **304-item testmini** instead:

```bash
python scripts/08_various_steering.py single \
  --override data.sources.mathvision.local_path=data/processed/mathvision_testmini.jsonl \
  --override data.sources.mathvision.split=testmini \
  --override data.sources.mathvision.source_split=testmini
```

`--preparation-dir` and the standard `--override key=value` options remain
available. The server must expose the bank's checkpoint; model mismatches fail
before generation. Stage 08 defaults to the existing vLLM-Hook v0.2 protocol.

### Subset evaluation and hybrid previews

Use `--subset-fraction 0.3` to evaluate approximately 30% of **each** enabled
dataset, and `--merge-base` to fill the complement from an existing result file:

```bash
python scripts/08_various_steering.py single \
  --subset-fraction 0.3 --subset-seed 42 \
  --merge-base /path/to/base.jsonl

# Change only the method to compare on the same sampled questions.
python scripts/08_various_steering.py random \
  --subset-fraction 0.3 --subset-seed 42 --random-seed 36 \
  --merge-base /path/to/base.jsonl

# Check sample counts, reference coverage and identity without inference.
python scripts/08_various_steering.py single \
  --subset-fraction 0.3 --subset-seed 42 \
  --merge-base /path/to/base.jsonl --dry-run
```

Sampling ranks sample identities by a seeded hash, rounds each dataset's count
up, and preserves the original output order. The same `--subset-seed`, data
population and fraction select the same questions across all seven methods,
independently of `--random-seed` and processing order. For the current default
datasets, 30% selects 450 MMStar, 270 MMMU, 300 MathVista and 912 MathVision
test questions. `--eval-limit`, if specified, applies **before** sampling.
Omitting `--merge-base` runs only the subset. The default fraction is `1.0`,
which retains full evaluation.

The reference can be an existing `baseline.jsonl` or `generations.jsonl`.
Each relevant row needs `dataset`, `item_id`, `item_identity`, and either
`raw_output` / `baseline_raw_output` or `predicted_answer` /
`baseline_predicted_answer`. Only successful rows (`status: "ok"`, or no status
field) are reused. A full reference file is recommended: its sampled questions
also allow direct consistency comparisons. A file containing only the
complement is accepted; consistency metrics then report zero compared items.
Different run IDs for the same sample are rejected; retries within one run use
the last successful row. Answers are scored again against the current gold
answers. Missing complementary rows, mismatched sample identities, gold
answers, or answer types stop the run before inference.

The run directory contains separate artifacts:

| File | Meaning |
|---|---|
| `subset_manifest.json` | Sampling parameters, per-dataset counts, every sample's assignment, and reference fingerprint. |
| `generations.jsonl`, `metrics.json`, `table2.csv` | Actual subset inference and subset-only scores, labelled `evaluation_scope: subset`. |
| `hybrid_generations.jsonl` | Full-population preview: new subset outputs plus reused reference outputs; every row records its source. |
| `hybrid_metrics.json`, `hybrid_table2.csv` | Explicitly labelled hybrid scores and per-dataset macro-average. The JSON also reports parsed-answer agreement, paired accuracy delta, and wrong-to-right / right-to-wrong counts on the overlapping subset. |

The hybrid score measures the requested mixture; the subset score measures the
method on the sampled questions. Failed sampled questions remain incomplete in
both outputs and are never replaced by reference answers. The reference file
is read-only and its contents are frozen by SHA-256. Resume with the same
fraction, sampling seed, reference and method (plus random seed for `random`):

```bash
python scripts/08_various_steering.py single \
  --subset-fraction 0.3 --subset-seed 42 \
  --merge-base /path/to/base.jsonl \
  --run-dir outputs/table2_single_MMDD_HHMM
```

Stage 08 resolves migrated `.../data/images/...` paths to existing images under
this project in memory, including legacy Windows paths. Source JSONL files are
not rewritten; missing local images still fail explicitly.

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
