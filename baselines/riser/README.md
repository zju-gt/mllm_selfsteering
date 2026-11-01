# RISER fixed-router baseline

这是一个与主实验隔离的 **RISER fixed-router approximation**。它复用 Stage 03
已经生成的六个多模态 capability vectors，但不修改 `scripts/07_run_adaptive_steering.py`
或 Self-Steering 的选择、归一化和生成逻辑。

## 复现边界

保留的 RISER 推理机制：

1. 从冻结的 vector library 读取 primitive vectors；
2. 固定 Router 对所有输入输出同一个 binary mask 与 strength vector；
3. 按 `v_inject = sum_i(mask_i * strength_i * vector_i)` 直接求和，不归一化；
4. 通过现有 vLLM-Hook 在目标层最后一个 token 上注入，并在 decode 阶段持续应用；
5. 使用同一 seed 生成 paired Pass@1 baseline 与 steered output。

这里的 primitive library 是本项目定义的三类文本、三类视觉 capability vectors，属于
RISER 机制的多模态适配，并不是论文原始的纯文本聚类 primitive library。

没有复现论文中的 Router warm-up、GRPO、逐输入动态选择，也没有使用可直接复现实验的
Router checkpoint。因此论文表格中应标为 `RISER (fixed router approximation)`，不能直接
标为完整 RISER。

## 运行

先完成现有 Stage 03，并准备一个状态为 `complete` 的 preparation run。然后运行：

```powershell
python scripts/run_riser_baseline.py `
  --preparation-dir outputs/preparation_MMDD_HHMM `
  --component text_logic=1.0
```

该入口会通过 baseline 专用配置将 generation backend 设为 `vllm_hook`；运行前需要像
`scripts/07_run_adaptive_steering.py` 一样启动并配置好 vLLM-Hook 服务。

也可以固定组合多个向量；这仍然不是动态 Router：

```powershell
python scripts/run_riser_baseline.py `
  --preparation-dir outputs/preparation_MMDD_HHMM `
  --component vision_detail=0.7 `
  --component text_quantitative=0.5
```

若不传 `--component`，使用 `configs/baselines/riser.yaml` 中的固定配置。建议只在验证集上
选择 capability 与 strength，之后冻结配置用于测试集，避免把测试集用于选择 baseline。

默认输出到 `outputs/baselines/riser_MMDD_HHMM/`：

- `vectors/fixed_*.pt`：固定组合后的 vLLM-Hook vector；
- `fixed_router.json`：mask、strength、来源向量指纹与范数诊断；
- `generations.jsonl`：逐样本 baseline/steered 配对结果；
- `metrics.json`、`metrics.csv`：总体及逐数据集指标；
- `config.json`：完整配置快照，可用于断点续跑和 `scripts/split_metrics.py`。

断点续跑：

```powershell
python scripts/run_riser_baseline.py `
  --preparation-dir outputs/preparation_MMDD_HHMM `
  --run-dir outputs/baselines/riser_MMDD_HHMM `
  --component text_logic=1.0
```

细分指标：

```powershell
python scripts/split_metrics.py outputs/baselines/riser_MMDD_HHMM
```
