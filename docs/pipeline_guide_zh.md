# 00–06 实验流水线运行指南

本文档说明 `scripts/` 下各阶段脚本的功能、输入、输出、关键配置和验收方法。
所有命令都应在仓库根目录执行。

> 自适应主实验现在推荐使用显式运行目录：先用
> `python scripts/prepare_experiment.py --pair-limit N` 创建
> `outputs/preparation_MMDD_HHMM/`，再把该目录传给
> `scripts/07_run_adaptive_steering.py --preparation-dir ...`。不传
> `--generation-dir` 会新建 `outputs/generation_MMDD_HHMM/`；传入已有生成目录才会断点续跑。
> 两类目录均保存 `config.json`，下述旧版 01–06 分阶段命令仍保留。

## 1. 流水线总览

```text
00_prepare_data
  四个评测数据集 → 统一 JSONL 与本地 PNG
          │
          ├── 00_prepare_pairs（可选：重新生成 contrast pairs）
          │
01_validate_pairs
  校验六组 contrast pairs
          │
02_capture_contrasts
  正样本激活 − 负样本激活 → 每对一个 activation delta
          │
03_extract_vectors
  聚合 activation deltas → 六个 capability vectors
          │
04_analyze_vectors
  计算组内一致性与能力间相似度
          │
05_run_steering
  在四个评测集上运行不同 capability × alpha 的生成
          │
06_score_generations
  完整性检查、准确率、相对 baseline 的变化与答案翻转统计
```

默认读取三个配置文件：

- `configs/model.yaml`：模型、精度、设备映射和最大生成长度；
- `configs/data.yaml`：数据集来源、本地 JSONL 和 Hugging Face 缓存；
- `configs/experiment.yaml`：六种能力、目标层、向量缩放、alpha 和生成后端。

除 `00_prepare_pairs.py` 外，各脚本都支持：

```bash
--config PATH                 # 可重复指定，按顺序合并配置
--override dotted.key=value   # 可重复指定，临时覆盖配置
--data-limit N                # 仅限制 Stage 00 写出的数据记录
--pair-limit N                # 仅限制 Stage 02-04 使用的 contrast pairs
--eval-limit N                # 仅限制 Stage 05-06 每个评测集的题目数
--limit N                     # 兼容旧用法：同时应用到以上三个范围
```

例如：

```bash
python scripts/02_capture_contrasts.py \
  --override experiment.target_layer=23 \
  --override model.name=Qwen/Qwen3-VL-4B-Instruct
```

每个正式阶段都会在 `outputs/manifests/` 写入 manifest，其中包含本次完整配置、
关键产物哈希和 `run_id`，用于区分不同模型、数据、pairs 和实验参数。

## 2. 正式运行前检查

建议先固定模型 revision，而不是长期使用浮动的 `main`：

```yaml
model:
  name: Qwen/Qwen3-VL-4B-Instruct
  revision: <Hugging-Face-commit-SHA>
```

确认 CUDA、torchvision 和可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

在当前服务器上，GPU 0 已被其他任务大量占用，应选择实际空闲的 GPU，例如：

```bash
export CUDA_VISIBLE_DEVICES=1
```

设置后，物理 GPU 1 会在当前进程内显示为 `cuda:0`，这是正常行为。

## 3. Stage 00：准备评测数据

### 命令

```bash
python scripts/00_prepare_data.py
```

### 功能

依次处理 `data.enabled_datasets` 中的 MMStar、MMMU、MathVista 和 MathVision：

1. 如果配置了 `local_path`，读取本地 JSONL；否则从 Hugging Face 加载；
2. 使用对应的数据集 adapter 统一题目、答案、选项和 split；
3. 把图片物化为普通 PNG；
4. 写出项目统一格式的 JSONL。

### 输入

- `configs/data.yaml`；
- 本地 `local_path` 或 Hugging Face 数据源；
- `data/images/` 中已部署的图片（使用本地规范化 JSONL 时）。

`local_path` 应指向 JSONL 文件：

```yaml
local_path: data/processed/mmstar.jsonl
```

而不是 `data/` 或 `data/images/` 目录。

### 输出

```text
data/processed/mmstar.jsonl
data/processed/mmmu.jsonl
data/processed/mathvista.jsonl
data/processed/mathvision.jsonl
data/images/<dataset>/*.png
outputs/manifests/00_prepare_data.json
```

当前完整数据的预期规模：

| 数据集 | 记录数 | 图片数 |
|---|---:|---:|
| MMStar | 1,500 | 1,500 |
| MMMU | 900 | 982 |
| MathVista | 1,000 | 1,000 |
| MathVision | 3,040 | 3,040 |
| 合计 | 6,440 | 6,522 |

如果 JSONL 是从 Windows 复制来的，执行本阶段后应再次确认其中的图片路径已经变成
服务器上的 `/sdb/...` 路径。

### 注意事项

不要在唯一一份正式 JSONL 上使用 `00_prepare_data.py --data-limit N`，因为它会把
`data/processed/*.jsonl` 覆盖为截断后的数据。小规模测试可以直接在后续阶段使用
`--pair-limit N` 或 `--eval-limit N`，无需截断正式数据集。

## 4. 可选 Stage 00b：生成 contrast pairs

### 命令

```bash
python scripts/00_prepare_pairs.py --count 128 --seed 20261012
```

### 功能

确定性构造三组文本能力与三组视觉能力的正负对：

- `text_logic`：显式逻辑推理与直觉回答；
- `text_quantitative`：精确计算与定性估算；
- `text_relevance`：筛选相关证据与混入无关信息；
- `vision_detail`：清晰图与受控模糊图；
- `vision_spatial`：原始布局与网格块重排；
- `vision_grounding`：包含目标标记与不包含目标标记的匹配面板。

这些 pairs 不采样四个评测数据集，避免提取向量时泄漏评测题目。

### 输出

```text
pairs/<capability>.jsonl
pairs/images/<visual-capability>/*.png
pairs/manifest.json
```

### 正式实验建议

仓库已经包含并版本控制了正式的 128 对/能力数据，因此服务器上通常跳过此脚本，
直接从 Stage 01 开始校验。重新运行会覆盖现有 pairs；而且 Windows Arial 与 Linux
DejaVu Sans 的字体渲染可能造成视觉图片像素和哈希不同。只有在明确需要生成一套新
pairs 时才运行，并在运行后从 Stage 01、Stage 02 重新开始。

## 5. Stage 01：校验 contrast pairs

### 命令

```bash
python scripts/01_validate_pairs.py
```

### 功能

加载 `experiment.capabilities` 配置的六个 JSONL，并检查：

- 文件和所有引用图片存在；
- 每个能力至少有两对样本；
- `pair_id` 在能力内唯一；
- 文本 pair 的正负两侧使用相同图片，只改变 prompt；
- 视觉 pair 的正负两侧使用相同 prompt，只改变图片；
- modality 和字段 schema 合法。

### 输出

```text
outputs/pair_validation.json
outputs/manifests/01_validate_pairs.json
```

正式数据应显示六种能力各有 128 对。这个阶段不加载大模型，也不需要 GPU；应在
任何 GPU 计算之前执行。

## 6. Stage 02：捕获正负对激活差

### 命令

先进行最小测试：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/02_capture_contrasts.py --pair-limit 1
```

正式运行：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/02_capture_contrasts.py
```

### 功能

1. 使用 Transformers 加载 `model.name` 和 processor；
2. 定位 `experiment.target_layer` 指定的语言解码器层；
3. 对每个 pair 分别执行正向和负向前向传播；
4. 捕获该层最后一个 prompt token 的 residual-stream activation；
5. 计算：

```text
delta = positive_activation - negative_activation
```

6. 每个 pair 保存一个 float32 `safetensors` shard。

文本 pair 固定图片并改变指令；视觉 pair 固定 prompt 并改变图片。因此 delta 的方向
被解释为相应能力从负条件指向正条件的方向。

### 输出

```text
outputs/activations/<capture_id>/<capability>/<pair_id>.safetensors
outputs/activations/<capture_id>/index.json
outputs/activations/<capture_id>/errors.jsonl       # 仅发生错误时出现
outputs/manifests/02_capture_contrasts.json
```

`capture_id` 由模型、revision、dtype、目标层、pairs JSONL 哈希、pair 图片哈希和
`--pair-limit` 等共同决定。修改其中任何一项通常都会产生新的目录，不会覆盖另一组实验。

已经存在的 shard 会被跳过，因此同一配置中断后可以直接重新执行。失败项写入
`errors.jsonl`，脚本仍会继续处理后续 pair。

### 正式验收

不要只看到脚本结束就进入 Stage 03。应确认每种能力都有 128 个成功 shard，并检查
`errors.jsonl`。Stage 03 只要求每种能力至少有一个成功 shard，因此不会自动拒绝
“部分成功”的向量提取。

```bash
find outputs/activations -name '*.safetensors' | wc -l
find outputs/activations -name errors.jsonl -print
```

默认六组各 128 对，完整总数应为 768 个 shard。

## 7. Stage 03：聚合 capability vectors

### 命令

```bash
python scripts/03_extract_vectors.py
```

如果 Stage 02 使用了 `--pair-limit 1`，本阶段也必须使用相同参数：

```bash
python scripts/03_extract_vectors.py --pair-limit 1
```

### 功能

读取当前 `capture_id` 下的所有成功 delta，对每种能力沿 pair 维度取平均，并构造三种
形式：

- `raw`：原始均值向量；
- `unit`：归一化为单位范数；
- `steering`：先单位化，再缩放到六个 raw vector 范数的平均值。

配置中的缩放名称与实际使用形式对应如下：

| `experiment.vector_scaling` | 使用的 vector form |
|---|---|
| `raw` | `raw` |
| `unit` | `unit` |
| `mean_norm` | `steering` |

默认 `mean_norm` 让六个能力向量具有相同范数，同时保留整体的典型强度。

### 输出

```text
outputs/vectors/<capture_id>/capability_vectors.safetensors
outputs/vectors/<capture_id>/capability_vectors.json
outputs/vectors/<capture_id>/vllm_hook/<capability>__raw.pt
outputs/vectors/<capture_id>/vllm_hook/<capability>__unit.pt
outputs/vectors/<capture_id>/vllm_hook/<capability>__steering.pt
outputs/manifests/03_extract_vectors.json
```

`.safetensors` 供 Transformers 后端使用；`.pt` 文件采用 `{"dir": vector}` 格式，
供 vLLM-Hook/MIA 使用。JSON metadata 包含目标层、各能力成功样本数和各形式范数。

## 8. Stage 04：分析向量质量

### 命令

```bash
python scripts/04_analyze_vectors.py
```

如果向量来自受限 pairs，smoke run 同样需要传入相同的 `--pair-limit`。

### 功能

计算两类诊断指标：

1. **Within-capability coherence**：同一能力中所有 pair delta 两两余弦相似度的均值；
2. **Cross-capability similarity**：六个聚合单位向量之间的余弦相似度矩阵。

coherence 越高，说明同一能力的 pair 更倾向于产生一致方向；能力间相似度过高则可能
表示两个能力未被清晰区分。但这些是诊断指标，不应单独作为能力有效性的最终证据，
仍需结合 Stage 05–06 的因果 steering 结果判断。

### 输出

```text
outputs/metrics/<capture_id>_vector_coherence.json
outputs/metrics/<capture_id>_vector_similarity.json
outputs/manifests/04_analyze_vectors.json
```

## 9. Stage 05：运行 steering generation

### Transformers 后端

最小测试：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/05_run_steering.py --eval-limit 1
```

正式运行：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/05_run_steering.py
```

### 功能

读取四个评测集，对每个题目执行完整笛卡尔积：

```text
dataset item × 6 capabilities × configured alphas
```

默认 alpha 为：

```yaml
[-1.0, -0.5, 0.0, 0.5, 1.0]
```

- `alpha = 0`：不注入向量的 baseline；
- `alpha > 0`：沿正向 capability vector 注入；
- `alpha < 0`：沿反方向注入。

Transformers 后端在指定语言层进行干预。`alpha=0` 的模型输出会在六种能力之间复用，
但结果文件仍为每个 capability 写一条 baseline 行，以支持严格配对统计。

当前完整配置预计写出：

```text
6,440 items × 6 capabilities × 5 alphas = 193,200 rows
```

`--eval-limit 1` 是“每个启用的数据集取 1 条”，因此会处理 4 个 item，并写出：

```text
4 × 6 × 5 = 120 rows
```

如果 Stage 02–04 使用全部 128 个 pairs 提取正式向量，但 Stage 05 只想评测 MMStar
前 100 条，可以独立设置评测范围：

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/05_run_steering.py \
  --eval-limit 100 \
  --override 'data.enabled_datasets=[mmstar]'
```

随后使用完全相同的评测参数评分：

```bash
python scripts/06_score_generations.py \
  --eval-limit 100 \
  --override 'data.enabled_datasets=[mmstar]'
```

这里不传 `--pair-limit`，所以 Stage 05 会选择由全部 pairs 得到的正式向量。若
Stage 02 使用的是 `--pair-limit 16`，那么 Stage 03–06 都要传入
`--pair-limit 16` 来选择对应向量；`--eval-limit` 仍可取任意独立值。

`--eval-limit N` 当前采用每个启用数据集 JSONL 的前 N 条，是确定性的前缀子集，
不是随机抽样。

### 输出

```text
outputs/generations/<run_id>.jsonl
outputs/manifests/05_run_steering.json
```

每行包含 dataset、item、capability、alpha、原始输出、抽取后的答案、是否正确、后端和
向量缩放方式。单项失败会写为 `status: error`。

该阶段支持断点续跑：

- 重新运行相同配置时跳过已有的成功组合；
- 错误组合会再次尝试；
- 意外终止留下的末尾不完整 JSONL 行会在恢复时移除；
- `generation.checkpoint_rows` 控制刷新检查点的频率。

`run_id` 包含 capture/vector 指纹、模型、revision、目标层、缩放方式、alphas、后端、
最大生成长度、`--eval-limit`、评测 JSONL 哈希，以及 vLLM 后端 provenance。因此参数变化
会生成新的结果文件。

### vLLM-Hook 后端

如果已经按 `docs/vllm_hook.md` 部署 MIA/vLLM 服务，可使用：

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook
```

正式运行必须在配置中记录服务版本：

```bash
python scripts/05_run_steering.py \
  --override experiment.generation.backend=vllm_hook \
  --override experiment.vllm_hook.server_provenance.vllm_version=<VERSION> \
  --override experiment.vllm_hook.server_provenance.mia_revision=<GIT_SHA> \
  --override experiment.vllm_hook.server_provenance.served_model_revision=<MODEL_SHA>
```

客户端会在大规模运行前执行一次 steering-effect probe。如果 baseline 和 steered 的
首 token logprobs 完全相同，会快速失败，避免在 hook 未生效时生成大量无效结果。

如果宿主机是 550 驱动、不能运行 CUDA 13，可按
`docker/vllm_hook_v0_2/README.md` 尝试固定的 CUDA 12 兼容栈，并额外传入：

```bash
--override experiment.vllm_hook.protocol=vllm_hook_v0_2
```

该模式把当前 `phase=both, positions=last_token` 精确映射为旧插件的
`apply_at_all_positions=true`。旧插件不支持 decode-only 或 all-token 语义，配置阶段会
直接拒绝这两种组合。这个兼容栈必须通过默认开启的 steering-effect probe 后才能用于
实验；仅服务启动成功或返回 HTTP 200 不能证明 hook 已经生效。

## 10. Stage 06：完整性检查与评分

### 命令

```bash
python scripts/06_score_generations.py
```

smoke run 必须与 Stage 05 使用相同的 `--eval-limit`，并使用匹配向量的
`--pair-limit`（如果 Stage 02 曾设置）：

```bash
python scripts/06_score_generations.py --eval-limit 1
```

### 功能

首先构造期望的完整组合并严格检查：

```text
所有 item × 所有 capability × 所有 alpha
```

只要存在缺失、额外组合或没有成功完成的错误行，脚本就拒绝评分。此时应重新运行
Stage 05，让失败组合补齐，而不是手动删除完整性检查。

完整后，按 dataset、capability 和 alpha 计算：

- `count`：成功样本数；
- `accuracy`：该 alpha 的准确率；
- `delta`：相对 `alpha=0` baseline 的准确率变化；
- `wrong_to_right`：baseline 错、steering 后正确；
- `right_to_wrong`：baseline 对、steering 后错误；
- `net_flips = wrong_to_right - right_to_wrong`。

### 输出

```text
outputs/metrics/<run_id>_steering_metrics.json
outputs/metrics/<run_id>_steering_metrics.csv
outputs/manifests/06_score_generations.json
```

需要注意，MathVista 和 MathVision 的开放式数学答案使用项目内的确定性归一化/数值匹配，
不是完整的符号等价证明器，也不是上游官方提交评分器。

## 11. 推荐运行顺序

### 已部署正式数据与仓库内 pairs

这是当前服务器最适合的正式流程：

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

python scripts/00_prepare_data.py
# 不重新运行 00_prepare_pairs.py，使用 Git 中已经固定的正式 pairs。
python scripts/01_validate_pairs.py
python scripts/02_capture_contrasts.py
python scripts/03_extract_vectors.py
python scripts/04_analyze_vectors.py
python scripts/05_run_steering.py
python scripts/06_score_generations.py
```

### 正式运行前 smoke test

保持完整评测 JSONL 不变，只限制计算阶段：

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

python scripts/01_validate_pairs.py
python scripts/02_capture_contrasts.py --pair-limit 1
python scripts/03_extract_vectors.py --pair-limit 1
python scripts/04_analyze_vectors.py --pair-limit 1
python scripts/05_run_steering.py --pair-limit 1 --eval-limit 1
python scripts/06_score_generations.py --pair-limit 1 --eval-limit 1
```

`--pair-limit` 是 capture ID 的组成部分；Stage 02 使用后，Stage 03–06 必须用相同值
选择该向量库。`--eval-limit` 只控制 Stage 05–06 每个启用数据集取多少条，并进入
generation run ID。两者彼此独立。正式运行时移除两个限制，会产生独立于 smoke test
的新 artifact ID，不会覆盖 smoke 结果。

## 12. 哪些配置变化要求重跑哪些阶段

| 修改内容 | 最早需要重跑的阶段 |
|---|---|
| 数据源、local path 或评测 JSONL | 00；之后至少重跑 05–06 |
| pairs JSONL 或 pair 图片 | 01，然后重跑 02–06 |
| 模型、revision、dtype、target layer | 02–06 |
| vector scaling | 05–06（Stage 03 已保存三种形式） |
| alpha 列表、最大生成长度、生成后端 | 05–06 |
| 仅分析展示方式 | 04 |

修改配置后不要手工复用旧目录。项目的 `capture_id` 和 `run_id` 会根据内容哈希及配置
自动选择相应产物；如果找不到前置产物，通常意味着某个阶段使用了不同的配置或不同的
对应 scope 的 limit。
