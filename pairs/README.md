# Contrast-pair inputs

Vector extraction never calls an annotation API. Each capability reads a user-curated
JSONL file configured in `configs/experiment.yaml`.

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

