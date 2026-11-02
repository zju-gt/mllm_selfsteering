# VISTA VSV-Only baseline

这是与主实验隔离的 `VISTA (VSV-Only, Qwen3-VL adaptation)`。它复现 VISTA 的
Visual Steering Vector 模块，不包含 Self-Logits Augmentation（SLA）。现有
`scripts/07_run_adaptive_steering.py`、Stage 03 vectors 和 Self-Steering 代码均不会被修改。

## 方法对应关系

对每一个图像—问题输入单独执行：

1. 构造包含图像的正上下文 `Xs + Xv + Xq`；
2. 构造移除图像但保留同一问题的负上下文 `Xs + Xq`；
3. 在每个语言层读取最后一个上下文 token，计算 `VSV_l = h_l(pos) - h_l(neg)`；
4. 将每层 VSV 注入对应 decoder MLP 输出；
5. 使用官方实现的 cosine-adaptive scaling，并将干预后的 token norm 恢复到原值；
6. 使用 greedy decoding 生成配对的 vanilla 与 VSV-Only 结果。

因此 VSV 与本项目 Stage 03 的离线 capability vectors 不同：它是逐输入、逐层动态生成的，
不需要 `--preparation-dir`。现有 vLLM-Hook 只接受单个目标层和普通加法，无法忠实表达
VISTA 的 all-layer norm-preserving injection，所以本 baseline 独立使用 Transformers。

这也意味着其绝对准确率可能与主表中通过 vLLM 得到的 Pass@1 存在少量后端差异。判断
VSV 收益时应优先比较同一 `generations.jsonl` 内的 paired Transformers baseline 与
VSV-Only 结果；若填主表，建议同时保留这条 paired baseline 作为审计依据。

论文和官方代码没有评测 Qwen3-VL，因此这是架构适配，而不是作者报告配置的逐位复现。
结果表中建议写作 `VISTA (VSV-Only, Qwen3-VL adaptation)`。

## 运行

```powershell
python scripts/run_vista_vsv_baseline.py --vsv-lambda 0.1
```

小规模检查：

```powershell
python scripts/run_vista_vsv_baseline.py --eval-limit 5 --vsv-lambda 0.1
```

`lambda` 对架构敏感。论文在其支持的模型上使用过 `0.10`、`0.12` 和 `0.17`，Qwen3-VL
没有官方值。应在验证集上选择强度，然后冻结后评测测试集。

## 输出

默认目录：

```text
outputs/baselines/vista_vsv_MMDD_HHMM/
├── config.json
├── generations.jsonl
├── vector_index.jsonl
├── vectors/
│   └── <dataset>/<item_identity>.safetensors
├── metrics.json
├── metrics.csv
└── manifests/
```

每个 safetensors 文件包含 `layer_000`、`layer_001` 等逐层 VSV。`generations.jsonl`
同时包含 `baseline_correct` 与 `correct`，因此兼容：

```powershell
python scripts/split_metrics.py outputs/baselines/vista_vsv_MMDD_HHMM
```

断点续跑时必须使用相同的配置和 lambda：

```powershell
python scripts/run_vista_vsv_baseline.py `
  --run-dir outputs/baselines/vista_vsv_MMDD_HHMM `
  --vsv-lambda 0.1
```
