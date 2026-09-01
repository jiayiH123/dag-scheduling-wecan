# 第一阶段阶段报告：WeCAN 基线忠实重实现

**报告日期：** 2026-07-29  
**阶段状态：** 工程最小闭环已实现并完成单元/烟雾测试；这不是官方或完全复现，也尚未证明优于 Greedy/HEFT。

## 1. 已完成功能

### 论文分析

- 已交付 [`wecan_analysis.md`](wecan_analysis.md)。
- 已确认原论文训练方法是 **REINFORCE**，不是 PPO；PPO-BiHyb 是论文的对照方法。
- 已记录原论文输入/输出/约束、WeCA、LDDGNN、递减 skip、算法2生成器、损失梯度、采样与 baseline。
- 已区分论文明确描述与缺失细节：当前只能称为**忠实重实现**。

### 数据与调度

- `data/instance.py`：离线单 DAG 实例、异构资源池、兼容性、随机分层 DAG 生成器、JSON 数据集序列化。
- `scheduler/types.py`：独立的 schedule 表达与可行性验证器。
- `scheduler/generator.py`：论文算法2风格的“静态任务—池分数 + 动态 mask + 解析递减 skip”生成器。
- 生成器不会执行神经网络；它最多执行 \(2|V|\) 次决策。
- 生成器的 mask 只允许：未调度、全部父任务已完成、兼容、且当前资源可容纳的任务—池动作。skip 仅在有运行任务时可用。

### WeCAN 模型

- `models/wecan.py`：
  - softmax 外乘兼容性系数的多头 WeCA；
  - LDD-biased DAG attention；
  - 静态任务—池分数；
  - 严格正的 \((\alpha,\beta,\gamma)\) skip 参数输出；
  - 每次 `forward()` 计数，用于验证单次前向要求。
- 因论文未公开精确的 WeCA 交替层数与正参数数值化方式，本工程将 `ldd_layers=2`、`alternating_weca_layers=2` 与 `softplus + epsilon` 明确作为配置假设；不是论文声称的唯一实现。

### 算法、训练与评估

- `baselines/algorithms.py`：Random、Greedy、capacity-aware HEFT。
- `training/reinforce.py`：Adam + REINFORCE；使用论文采用的平均样本 baseline。论文未定义其精确式，本工程显式使用每 batch 的 detached mean makespan，且每实例采样一个 rollout。
- 已支持训练、恢复训练、测试、单实例推理、结果 JSON 和绘图。
- 评价输出包含：makespan 均值/标准差、可行率、网络前向耗时、完整生成耗时、相对 Greedy/HEFT 改善比例，以及相对 Greedy 的逐实例胜率。

## 2. 已通过测试

执行命令：

```bash
python -m compileall -q baselines data environment evaluation models scheduler scripts tests training
pytest -q
```

结果：**5 passed**。

覆盖内容：

1. 随机生成 DAG 合法且可拓扑排序。
2. Random、Greedy、HEFT 的输出满足依赖、兼容性和累积资源容量约束。
3. skip-extended 生成器满足基础约束，且决策步数不超过 \(2n\)。
4. 模型只执行一次前向；之后调度生成器不会重新调用模型。
5. 递减 skip score 严格递减。

还执行了恢复训练烟雾测试，checkpoint 能恢复模型、优化器状态与 epoch：恢复后日志从第 3 epoch 继续。

## 3. 当前实验结果（仅烟雾测试，不作性能结论）

### 配置

- 随机生成：24 个训练、8 个验证、12 个测试实例；每个实例 12–24 个任务、3 个资源池、2 个资源维度。
- 固定种子：2026。
- CPU 上训练 2 个 epoch（随后恢复额外 1 epoch）。
- 测试集与资源配置在 Random、Greedy、HEFT 和 WeCAN 间完全相同。

### 基线结果（12 个测试实例）

| 算法 | 可行率 | Makespan 均值 ± 标准差 | 相对 Greedy 胜率 |
|---|---:|---:|---:|
| Random | 1.000 | 140.801 ± 28.168 | 0.000 |
| Greedy | 1.000 | 114.075 ± 29.345 | 0.000（自身） |
| HEFT | 1.000 | 113.741 ± 29.294 | 0.167 |

### WeCAN 烟雾训练后的 greedy 推理

| 算法 | 可行率 | Makespan 均值 ± 标准差 | 相对 Greedy 平均改善 | 相对 Greedy 胜率 |
|---|---:|---:|---:|---:|
| WeCAN greedy（2 epoch） | 1.000 | 123.608 ± 31.227 | -8.585% | 0.000 |

单实例推理输出：

- `forward_calls = 1`；
- 基础约束 `feasible = true`，无 violation；
- 总生成过程没有额外神经网络调用。

**解读：** 这个很小、很短的 CPU 烟雾训练没有超过 Greedy 或 HEFT，且实际落后于 Greedy。这符合预期；它只证明工程能训练、推理、生成可行调度和统计指标，绝不构成“复现成功”或“稳定超过 Greedy”的证据。

完整烟雾结果文件：

- `results/baselines_smoke.json`
- `results/phase1_smoke.json`
- `results/single_instance_smoke.json`
- `results/phase1_smoke.png`
- `checkpoints/wecan_smoke/best.pt`

## 4. 与原论文一致和不一致部分

### 一致部分

- 问题目标：最小化异构资源池的 DAG makespan。
- 可行性约束：依赖、累积容量、兼容性和非负开始时间。
- 一次前向输出静态任务—池 score 和 instance-level skip 参数。
- WeCA 的兼容性系数在 softmax **外部**相乘。
- 使用 LDD 型 DAG attention。
- 有 skip 动作；其 score 按论文的指数递减形式解析计算。
- 调度时动态 mask，训练时按 masked categorical 采样，贪心时 argmax。
- 原始训练家族：REINFORCE + average-sample baseline，而非 PPO。

### 不一致/忠实重实现假设

- 论文未提供 WeCAN 源码、训练数据、checkpoint 或精确网络层数；当前随机数据集不是 TPC-H/RI-W 数据。
- 论文写的是 512/128 维、8 层 LDDGNN；为能在最小示例和 CPU 上快速验证，本项目配置采用 128 维、2 层 LDD，不能视为论文最终超参数复刻。
- 论文没有精确定义 skip 正参数映射，当前采用 `softplus + 1e-4`。
- 论文没有精确定义平均 reward baseline 的集合与归一化，当前采用 batch 内包含自身的 detached mean makespan。
- 论文的 LDD 还定义了固定的八类 head mask；本工程已保留 LDD bias 和可区分的关系类型，但尚未逐 head 完整复刻该八类 mask。该差异在后续基线强化中应补齐。
- HEFT 采用容量感知的 earliest feasible insertion，以适应论文的累积多资源池问题；这不是标准单处理器 HEFT 的逐任务独占 timeline 限制。

## 5. 尚未解决的问题

1. 尚未在论文的真实 TPC-H、计算图或 RI-W 数据上验证，因此不能比较论文表格。
2. 尚未进行足够长的多随机种子训练，也没有泛化到大于训练规模的 DAG 的结果。
3. LDDGNN 的原论文八种 per-head mask、精确层间投影与完整高/低维结构尚未逐项还原。
4. 当前生成器为清晰可测的 Python 逐实例实现；论文训练使用向量化 PyTorch/GPU 生成器，性能基准尚不可直接对比。
5. 还没有通信、成本、能耗、预算、MIP 专家数据、方案 A/B 或逐任务 RL 对照；它们必须留到后续阶段。

## 6. 下一阶段计划

在继续进入第二阶段前，建议先把第一阶段从“最小闭环”提升为“可用于研究对比”的基线：

1. 扩展训练预算与固定多随机种子实验；保留原始 smoke 输出以避免选择性报告。
2. 补齐 LDD 八类 head mask 和论文高/低维架构的可选 profile。
3. 运行训练规模 20–100 任务、测试含超过训练规模 DAG 的泛化实验；报告均值、标准差和 Greedy 胜率。
4. 若上述验证稳定通过，再冻结 Phase-1 接口并进入第二阶段：把 `问题建模.pdf` 中的通信、成本、能耗和预算约束引入约束感知模拟器；再分别实现方案 A、方案 B 和逐任务 RL 对照。

在这些工作完成前，不应声称 WeCAN 复现成功，或称模型稳定优于 Greedy/HEFT。
