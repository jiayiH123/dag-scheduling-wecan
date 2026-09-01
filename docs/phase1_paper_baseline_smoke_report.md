# Phase 1 论文 WeCAN 基线配置与端到端运行报告

**日期：** 2026-08-25  
**范围：** 无通信、无成本/能耗/预算、无 PPO、无 fixed-priority。  
**性质：** 工程运行检查，不是论文数据复现、性能实验或收敛结论。

## 1. 可执行配置

完整配置位于：

```text
configs/phase1_paper.yaml
```

该文件现在同时包含：

- `paper_specification`：论文明确给出并由本配置采用的参数；
- `implementation_assumptions`：论文没有充分规定的本地实现选择；
- `generator`：仅用于本地运行的无通信随机 DAG 数据配置；
- `model`：可直接构造 `WeCANConfig(profile="paper")`；
- `training`：可直接构造 `TrainConfig`，并以 `max_updates=800` 精确表达论文的 800 training batches。

`epochs=800` 只是保证即使数据集每个 epoch 仅有一个 batch，也能到达 800 updates 的安全上限；正式停止条件是 `max_updates=800`，不把论文的 batches 错写成 epochs。

## 2. 严格来自论文的参数

以下参数来自论文第 8 页 Eq. (7)–(8)、第 10–11 页 §IV-C–D，以及第 30 页 Appendix F.1–F.3：

| 参数 | 配置值 |
|---|---:|
| Task input dimension | 3（任务时长 + 二维资源需求） |
| Pool input dimension | 2（二维资源容量） |
| `d_high` | 512 |
| `d_low` | 128 |
| WeCA heads | 8 |
| WeCA compatibility | softmax 外乘 `Kacc` |
| LDDGNN layers | 8 |
| LDD heads | 16 |
| LDD mask types | 8，每类 2 heads |
| `Dmax` | 500 |
| Decoder | `(Wq h)^T(Wk z) + log(Kacc)` |
| Skip MLP | 2 个 hidden layers，hidden width 64 |
| Skip activations | GELU、Sigmoid |
| Skip score | `log(alpha * exp(-gamma*k/(2*n)) + beta)` |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Default batch size | 64 |
| Training length | 800 batches/updates |
| REINFORCE baseline | average reward of all samples；本配置采用论文正文所举的 instance-wise mean 解释 |

代码中对应的 paper profile 为：512/128、8-head WeCA、8 层/16-head LDDGNN、`Dmax=500` 和逐元素 Sigmoid skip 输出。

## 3. Implementation assumptions

以下内容没有被当成论文规定，均在 YAML 中单独记录：

| 项目 | 当前实现选择 | 状态 |
|---|---|---|
| 后续 alternating WeCA 层数 | 2 | 论文未给层数 |
| 训练 rollout 数 `K` | 每实例 8 条 | 论文未给训练 K |
| average-sample baseline 解释 | 同一实例 K 条样本的均值，包含当前样本 | 论文未明确聚合域和是否 LOO |
| LDD distance bias | 每个 head 独立 bias | **待确认；本轮按要求不修改** |
| Transformer block | output projection、post-LayerNorm、2× hidden FFN、无 dropout | 论文未完整给出 |
| Skip head 排列 | `128→64→GELU→64→GELU→3→Sigmoid` | 与论文文字一致的实现解释，但逐层顺序未完整刊出 |
| Decoder projection bias | `False` | 论文未规定 |
| 本地数据 | 固定种子的随机分层 DAG | 不是论文 TPC-H、RI-W 或 computation-graph 数据 |

另外，论文未给出 Adam betas/epsilon、初始化、学习率 schedule、gradient clipping 和训练精度等细节；当前使用 PyTorch 默认 Adam 数值设置，且没有额外 LR schedule/dropout。

## 4. 代码调整

- 补全 `configs/phase1_paper.yaml`，现在可供 `generate_data.py` 和 `train.py` 直接读取。
- `TrainConfig` 增加显式 `optimizer="adam"` 契约；其他 optimizer 会被拒绝。
- `scripts/train.py` 将配置中的 `optimizer` 传入 `TrainConfig`。
- 新增 `tests/test_phase1_paper_config.py`：
  - 检查 YAML 中论文参数与 assumption 分区；
  - 检查配置可构造完整运行对象；
  - 检查 paper 模型输入宽度、forward、Sigmoid skip 和递减 skip score；
  - 检查 sampled rollout、独立 Validator、REINFORCE backward 和日志字段。

未修改模型结构，也未修改 LDD bias。

## 5. 端到端运行检查

使用完整 paper 模型参数和极小本地数据执行：

```bash
python scripts/generate_data.py \
  --config configs/phase1_paper.yaml \
  --output-dir /tmp/wecan-paper-baseline-smoke/data \
  --train-count 1 --validation-count 1 --test-count 1

python scripts/train.py \
  --config configs/phase1_paper.yaml \
  --train-data /tmp/wecan-paper-baseline-smoke/data/train.json \
  --validation-data /tmp/wecan-paper-baseline-smoke/data/validation.json \
  --checkpoint-dir /tmp/wecan-paper-baseline-smoke/checkpoint \
  --device cpu --epochs 1

python scripts/infer.py \
  --checkpoint /tmp/wecan-paper-baseline-smoke/checkpoint/best.pt \
  --data /tmp/wecan-paper-baseline-smoke/data/test.json \
  --index 0 \
  --output /tmp/wecan-paper-baseline-smoke/inference.json \
  --device cpu
```

本次只进行 1 个 optimizer update，不运行正式的 800 updates。

### 运行结果

| 检查 | 结果 |
|---|---|
| Paper 配置加载 | 通过 |
| 数据生成 | train/validation/test 各 1 个无通信实例 |
| 完整 512/128 模型 forward | 通过 |
| 单实例 forward 次数 | 1 |
| Sample rollout | 通过 |
| Skip | 正常产生；训练轨迹 `skip_ratio=0.4889`，其中 active wait ratio `0.3694` |
| REINFORCE backward/Adam step | 通过；`gradient_norm=2058.6338`，有限 |
| Policy loss | `18.4794`，有限并成功记录 |
| Sampled mean makespan | `340.8443`，有限并成功记录 |
| Validation greedy makespan | `176.1477` |
| Test greedy makespan | `188.3640` |
| Test schedule | `feasible=true`，无 violations |
| Checkpoint 写入与重新加载 | 通过 |

这些数值没有性能含义，仅证明完整 paper-size 模型能够在 CPU 上完成数据加载、一次前向、多轨迹生成、skip、REINFORCE 反传、checkpoint、重新加载和合法调度推理。

## 6. 自动测试

聚焦测试：

```text
10 passed in 7.04s
```

覆盖 paper 配置契约、paper profile shape、LDD mask/folding、skip 参数、baseline 数学和一次 REINFORCE update。

## 7. 剩余不确定项

1. LDD bias 应由所有 heads 共享一个标量，还是每个 head 独立；当前保留 head-specific 实现。
2. alternating task/pool WeCA 的准确层数。
3. 训练时每实例 sample/rollout 数。
4. average-sample baseline 是否包含当前样本，以及 “all samples” 的确切聚合范围。
5. Skip MLP 中 Sigmoid 的准确位置、输出缩放和 epsilon。
6. LayerNorm 的 pre/post 顺序、FFN expansion、dropout、projection bias 和初始化。
7. Adam betas/epsilon、学习率调度和数值精度设置。
8. 论文原始数据与完整数据生成实现不可得，因此尚不能复现论文结果表。

## 8. 结论

`configs/phase1_paper.yaml` 现在是一份可以由现有 CLI 端到端执行的论文 WeCAN 基线配置。论文明确给出的模型尺寸和训练参数均已配置；未明确项均单列为 implementation assumptions。完整 paper-size 路径已经通过一次小数据运行检查，但尚未执行 800-batch 正式训练，也没有形成任何论文性能复现结论。
