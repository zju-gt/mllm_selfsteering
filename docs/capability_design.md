# Provisional six-capability design

The taxonomy is deliberately configuration-driven. The current proposal keeps three
dimensions close to the original DeLeAn experiment and adds three controlled visual
contrasts.

| ID | Modality | Positive side | Negative/control side |
|---|---|---|---|
| `text_logic` | text | explicit premises, dependencies, deductions | intuitive or answer-only instruction |
| `text_quantitative` | text | explicit quantities, equations, checked arithmetic | qualitative estimation or answer-only instruction |
| `text_relevance` | text | identify necessary evidence and ignore distractors | indiscriminate description/use of all information |
| `vision_detail` | visual | original image | downsampled/blurred image preserving global layout |
| `vision_spatial` | visual | original spatial layout | patch-shuffled or layout-disrupted image preserving local appearance |
| `vision_grounding` | visual | question-matched image | mismatched/control image with similar broad content |

## Why these three visual dimensions

They target different failure modes: loss of local detail, loss of global spatial
structure, and failure to bind language to the correct visual evidence. The transforms
must be recorded in pair metadata and should be calibrated so that the negative side
changes the intended property without making the entire image unusable.

The visual direction is always `activation(original/matched) - activation(control)`.
For text dimensions it is `activation(positive instruction) - activation(negative
instruction)`. Both sides keep every uncontrolled input identical.

Before a formal run, inspect cross-capability cosine similarity and add nuisance
controls (random image pairing, equal-length text prompts, transform-severity sweeps).

