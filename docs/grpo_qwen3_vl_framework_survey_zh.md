# Qwen3-VL-8B-Instruct GRPO 框架调研

调研日期：2026-11-02。依据为本次读取的官方仓库 main 分支代码与文档；本文未进行安装或 GPU 训练验证，后续选型应固定 commit 和依赖环境。

目标方案见 [GRPO 联合训练 Self-Steering 方案](grpo_self_steering_training_design_zh.md)。本次只进行资料调研，源码片段临时缓存于 `/tmp`，尚未在项目中 clone 或安装候选框架。

## 1. 选型要求

除支持 Qwen3-VL 多模态 GRPO 外，还需要：

1. 一条 rollout 包含两次生成，中间解析权重并合成冻结向量。
2. 两次生成使用独立 prompt，作答上下文不拼接路由文本。
3. 每条样本携带自己的 steering 状态，生成与训练重放都使用该状态。
4. 路由以 `format + accuracy`、作答以 `accuracy` 分别计算优势。
5. 同时训练两段生成 token，支持八条完整轨迹的分组统计。
6. 正确同步训练与生成模型权重，重算旧策略、新策略和采用时的参考策略概率。

支持多个 reward 函数通常只意味着先合成一个序列级分数，不能据此认定框架默认支持本方案的分段优势。支持多轮对话也不能代替两个独立 prompt 的轨迹表示。

## 2. 候选框架概览

| 框架 | 已核验模型证据 | 后端与扩展入口 | 对本项目的判断 |
|---|---|---|---|
| [TRL](https://github.com/huggingface/trl) | 官方 Qwen3-VL GRPO 教程，代码明确列出 8B-Instruct 作为可选模型 | Transformers 训练/生成、可选 vLLM；`rollout_func`；可继承 trainer | 优先用于首版算法实现，便于统一生成与训练侧 Hook |
| [verl](https://github.com/verl-project/verl) | 8B-Instruct 专用 GRPO FSDP2/vLLM 脚本 | FSDP2、Megatron 等；AgentLoop、自定义 advantage estimator | 优先的分布式训练候选，数据流与训练 forward 需要适配 |
| [ms-swift](https://github.com/modelscope/ms-swift) | 目标 8B 型号显式注册，支持多模态 GRPO | Transformers、DeepSpeed 等；vLLM rollout；自定义多轮 scheduler | 候选成熟，但分段奖励不能直接套用默认多轮路径 |
| [EasyR1](https://github.com/hiyouga/EasyR1) | Qwen3-VL GRPO，官方脚本为 4B 与 30B-A3B | FSDP、vLLM；自定义 advantage estimator | 结构相对精简；8B 需补运行验证，两阶段调度需改核心流程 |
| [miles](https://github.com/radixark/miles) | Geo3K GRPO 脚本默认即 8B-Instruct | FSDP/Megatron 训练、SGLang rollout；custom generate | 明确可用的模型候选；现有 vLLM-Hook 迁移和环境成本较大 |
| [ROLL](https://github.com/alibaba/ROLL) | 8B-Instruct Geo3K GRPO 专用配置 | 示例为 Megatron 训练、vLLM rollout；可定制 pipeline | 分布式方案备选，官方示例拓扑需调整 |
| [AReaL](https://github.com/areal-project/AReaL) | 官方模型表支持 Qwen3-VL，提供 VLM GRPO 示例 | FSDP/Megatron、vLLM/SGLang；自定义 workflow | Agent rollout 候选；本次未核验 8B 专用配置 |

## 3. 关键证据与限制

### TRL

- [Qwen3-VL GRPO 教程](https://github.com/huggingface/trl/blob/main/examples/grpo_qwen3_vl/grpo_qwen3_vl.ipynb)：默认 `Qwen/Qwen3-VL-4B-Instruct`，同一模型配置行显式列出 `Qwen/Qwen3-VL-8B-Instruct`。这是官方教程支持证据，不等于本次完成了 8B 实测。
- [GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)：`rollout_func` 可自定义生成并返回 token/logprob 和附加字段；当前接口标注为 experimental。
- 同文件的普通 loss 路径明确兼容子类传入 `[B,T]` advantages，默认仍由序列奖励计算优势，需要自定义分段构造。
- 本项目需要定制两段样本组织、生成/重放 Hook、分段 reward 与 advantage，以及多模态附加字段传递；仅提供 `rollout_func` 不会自动完成这些工作。
- 首版可使用 Transformers 完成两段生成与训练重放，先减少推理引擎迁移因素；速度需要实测。之后再考虑 vLLM 加速。

### verl

- [8B FSDP GRPO 脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh)：`MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct`，`algorithm.adv_estimator=grpo`，训练策略为 `fsdp2`，rollout 为 `vllm`。
- [AgentLoop 文档](https://github.com/verl-project/verl/blob/main/docs/advance/agent_loop.rst)：可实现 `AgentLoopBase.run`，定义生成、环境操作等流程，并返回 token 与 response mask。
- [GRPO estimator](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)：有注册扩展入口；默认 outcome GRPO 将序列分数归一化后广播至 response token，需要替换为分段优势。
- [Qwen3-VL 训练适配](https://github.com/verl-project/verl/blob/main/verl/models/transformers/qwen3_vl.py)：存在专用模型 forward 与多模态位置适配。Hook 需考虑 remove-padding、packing 和样本边界。
- AgentLoop 的普通拼接轨迹表示不能直接假设支持本项目的独立 prompt 重置；需显式保留两段记录及其共同 group/trajectory ID。

### ms-swift

- [目标型号注册](https://github.com/modelscope/ms-swift/blob/main/swift/model/models/qwen.py#L1146)：显式注册 `Qwen/Qwen3-VL-8B-Instruct`。
- [官方 README](https://github.com/modelscope/ms-swift/blob/main/README.md)：支持 Qwen3-VL，以及多模态全参数/LoRA/QLoRA GRPO。
- [多轮 scheduler 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md)：允许覆写 `run` 定义完整流程、重置下一轮请求，并传递 token、loss mask 和多模态数据。
- 同文明确要求拆成多个记录的同一轨迹使用相同 reward；[trainer](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/grpo_trainer.py) 会校验同 `request_id` 奖励一致。因此不能只写 scheduler 就实现当前分段奖励。
- [advantage 实现](https://github.com/modelscope/ms-swift/blob/main/swift/rl_core/advantage.py) 已使用 `[B,T]` 表示，但默认广播序列优势，需定制。当前 Liger fused loss 路径还会将 token advantages 平均回序列值，应禁用或适配该路径。

### EasyR1

- [README](https://github.com/hiyouga/EasyR1/blob/main/README.md) 明确支持 Qwen3-VL、图文/多图输入和 GRPO。
- [4B 专用脚本](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen3_vl_4b_geo3k_grpo.sh) 与 [30B 专用脚本](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen3_vl_30b_geo3k_grpo.sh) 已核验；将 dense 型号改为 8B 是合理候选，但本次未找到 8B 专用示例。
- [rollout](https://github.com/hiyouga/EasyR1/blob/main/verl/workers/rollout/vllm_rollout_spmd.py) 主要围绕一次 vLLM 生成组织；两段流程需要改数据流。
- [GRPO estimator](https://github.com/hiyouga/EasyR1/blob/main/verl/trainer/core_algos.py) 默认为 outcome-only，可注册替代实现。

### miles

- [Geo3K 文档](https://github.com/radixark/miles/blob/main/examples/geo3k_vlm/README.md) 显式列出 8B-Instruct，并提供 FSDP 与 Megatron 两种路线。
- [启动脚本](https://github.com/radixark/miles/blob/main/examples/geo3k_vlm/run_geo3k_vlm.sh) 默认 `Qwen3-VL-8B-Instruct`、`--advantage-estimator grpo`，使用 `--custom-generate-function-path`。
- [FSDP 模型适配](https://github.com/radixark/miles/blob/main/miles/backends/fsdp_utils/adaptations/specs/qwen3_vl.py) 使用 HF 模型路径，具有训练侧改造入口。
- 默认 SGLang rollout 需要另行实现 steering；不能直接复用现有 vLLM-Hook 服务。该示例当前还推荐 CUDA 13 image，与本机现有驱动环境需要单独评估。

### ROLL 与 AReaL

- ROLL 的 [8B Geo3K 配置](https://github.com/alibaba/ROLL/blob/main/examples/qwen3-vl-8B-geo3k-rlvr_megatron/rlvr_megatron_80G.yaml) 明确写有目标模型、`adv_estimator: grpo` 和 vLLM 推理策略。配置默认 8 卡及相应 TP/PP/CP 拓扑，不能直接照搬到本机 6 卡。
- AReaL 的 [模型表](https://github.com/areal-project/AReaL/blob/main/README.md) 明确支持 Qwen3-VL 的 FSDP 与 Megatron 路线，并有 [VLM agent GRPO 示例](https://github.com/areal-project/AReaL/blob/main/examples/vlm/geometry3k_agent_grpo.py)。本次读取的普通 Geometry3K 配置默认是 Qwen2.5-VL-3B，应将家族支持与目标 8B 实测区分。

## 4. 暂不列为目标型号已确认的首选

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：已支持 GRPO 和多模态；本次读取的 [VLM 示例](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_vlm_math_hybrid_engine.sh) 标注测试型号为 Qwen3.5，未直接证明目标 Qwen3-VL-8B 已测试。
- [slime](https://github.com/THUDM/slime)：支持自定义生成和 VLM GRPO；本次 main 的 [Geo3K 示例](https://github.com/THUDM/slime/blob/main/examples/geo3k_vlm/README.md) 主要为 Qwen3.5。存在 Qwen3-VL 权重转换文件不足以单独证明完整 8B GRPO 链路；miles 的目标型号证据更明确。
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)：存在多模态 GRPO，但本次未定位到目标 Qwen3-VL-8B 的专用训练配置，故不列为已确认首选。

这些判断只表示本次证据覆盖范围，不等于断言对应框架无法支持目标模型。

## 5. 环境兼容性

本机只读查询结果为 6 张 A100 80GB PCIe，驱动 `550.54.15`；现有 Hook image 基于 vLLM `0.12.0`。GPU 数量不代表当前可用资源，亦不能据此保证任意默认训练配置可运行。

本次 main 快照的依赖差异包括：TRL 的 vLLM extra 为 `>=0.22.1,<0.33.0`；verl 的 requirements 要求 `transformers>=5.5.3,!=5.6.0,<5.17`。ms-swift 的 Qwen3-VL 最佳实践写有 `transformers>=4.57`、`qwen_vl_utils>=0.0.14`、`ms-swift>=4.0`，vLLM 推理下限为 `0.11.0`；这些下限不等于与现有 Hook 版本经过联合验证。

因此需要固定适配版本并使用独立训练环境。沿用 vLLM 路线时，必须同时处理 Hook、token/logprob 返回和训练权重同步，不能把现有只读推理服务直接视为可训练 rollout 后端。

## 6. 建议

第一批优先 clone TRL 与 verl，固定提交，仅用于实现与对照参考：

1. 首版优先采用 TRL + Transformers 的生成/训练路径，验证两段独立 prompt、逐样本向量状态、分段优势以及干预重放的一致性。
2. 若后续实测吞吐或显存成为主要瓶颈，评估 verl FSDP2 + vLLM 路线。应先确保目标模型原生 GRPO 能运行，再接入 steering。
3. ms-swift 作为具有成熟模型适配和多轮插件的备选；miles/ROLL/AReaL 作为后续分布式方案候选。

优先级依据是本项目的修改成本，不是框架整体性能排名。正式选择前仍需进行小规模模型加载、生成、logprob 重算与反向传播验证。
