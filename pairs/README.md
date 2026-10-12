# Contrast-pair inputs

Vector extraction never calls an annotation API. Each capability reads a controlled
JSONL file configured in `configs/experiment.yaml`; the file may come from the
reproducible generator below or from a reviewed manual collection.

Text-operation pair (images are optional; when present they must be identical on both
sides):

```json
{"pair_id":"t1","positive_prompt":"Solve by deriving each implication.","negative_prompt":"Give an intuitive guess.","image_paths":[],"metadata":{"source":"manual"}}
```

Visual-operation pair (the prompt is held fixed and the image input changes):

```json
{"pair_id":"v1","prompt":"Read the smallest label in the diagram.","positive_image_paths":["data/source/original.png"],"negative_image_paths":["data/source/blurred.png"],"metadata":{"transform":"downsample_blur"}}
```

Paths may be absolute or relative to the project directory. The validator rejects
text pairs whose positive and negative image lists differ and visual pairs whose
positive and negative prompts differ. These constraints keep each contrast as close
as possible to a single controlled intervention.

## Reproducible controlled library

Run `python scripts/00_prepare_pairs.py` to build the six configured files with 128
evaluation-independent pairs per capability. Text pairs use matched synthetic tasks
with only the reasoning instruction changed. Visual pairs use deterministic diagrams
and controlled detail degradation, aligned patch derangement, or matched/unmatched
grounding panels. The generator records its seed and transform parameters in each row
and writes content hashes to `pairs/manifest.json`.

The four evaluation benchmarks are deliberately not sampled by this generator. This
prevents examples used to estimate a capability vector from reappearing in stage 05.

