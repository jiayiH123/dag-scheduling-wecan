# Phase 1.5 阶段报告（进行中）：Oracle 通过，A1-a 已重分类，后续训练未启动

**日期：** 2026-08-01
**当前状态：** Oracle、论文一致模型/训练改造与原始 A1 已完成。原 A1 的旧 gate 将 external Greedy 启发式错误地与初始 policy greedy decode 混用；它已被保留并重新解析为 A1-a“策略收缩和训练正确性验证”。A1-b/A1-c 仅允许进行实例筛选与 Oracle 证明，尚未启动任何训练。

## 1. 完成内容

### 独立正确性模块

已新增三个不复用核心排程逻辑的模块：

- `scheduler/validator.py`：独立 schedule/trace Validator；
- `oracle/milp_oracle.py`：PuLP/CBC 单一时间索引 MILP Oracle；
- `oracle/exhaustive_oracle.py`：4–7 任务独立枚举 Oracle。

Validator 基于完整半开事件区间 `[start,end)` 检查任务完整性、兼容性、执行时长、依赖、所有 pool 的所有非零事件区间容量、makespan 与可选 dispatch/skip trace。它不调用 generator、HEFT 或 Oracle 的调度逻辑。

MILP 仅使用 `y[i,t,p]` 启动变量和 `Cmax_ticks`；Oracle 仅针对整数或可精确有理缩放实例声明全局最优。非精确可缩放实例将标记 `unsupported_time_scale`，不静默近似。每次 CBC 求解保存日志，并同时要求 optimal proof、无中断、incumbent、可得 bound/gap 一致及独立 Validator 才标 `optimal`。

穷举器允许同一时刻连续启动多任务，也允许在仍有可行动作时主动 wait/skip 至最近完成事件；state key 包含当前时间、未调度/完成/运行任务、各运行任务 pool/结束时间及资源使用，避免只按任务集合错误合并。

### Oracle 结果

执行：

```bash
python scripts/run_oracle_validation.py --random-count 50 --seed 2026
pytest -q
```

结果：

```text
Cross-checked 60 instances; failures=0
26 passed
```

60 个实例 = 10 个手工实例 + 50 个随机 4–7 task 实例。每个随机实例均满足：

- MILP CBC 审计结果为 `optimal`；
- Exhaustive Oracle `complete_search=true`；
- 两者 makespan 差值不超过 `EPS=1e-6`；
- 两者均通过独立 Validator；
- Random、Greedy、CA-HEFT 均通过 Validator；
- 不存在 `heuristic_makespan < oracle_makespan - EPS`。

报告：`results/oracle/crosscheck_2026.jsonl`。

### Phase 1.5 模型一致性改造

新增：

- `configs/phase1_paper.yaml`：512 初始 embedding、512 维初始 WeCA + 8 层 LDDGNN、16 LDD heads、投影到 128、8-head WeCA、`Dmax=500`、Sigmoid skip head。
- `configs/phase1_smoke.yaml`：非论文的快速测试 profile。
- 论文 LDD `N1..N8` per-head mask；16 heads 每类 2 heads。
- `Dmax=500` 用户确认的边界折叠：有限 `d<=-500→-499`，`-∞→-500`，有限 `d>=500→499`，`+∞→500`。
- 专用 LDD head 为空邻域时 attention message 显式为零，避免 NaN，不开放原本被 mask 的 self-loop/key。
- `paper_sigmoid` 与 `softplus` skip 参数模式。

### 训练与基线改造

- `CA-HEFT` 已成为原容量感知 HEFT 的正式名称；保留旧 `heft_schedule` 兼容别名。
- 新增 Standard HEFT：每 pool 单任务独占 timeline 的经典保守语义。
- REINFORCE 支持：`batch_global_mean`、`instance_mean`、`instance_leave_one_out`、`current_policy_greedy`、`snapshot_greedy_rollout`。
- `snapshot_greedy_rollout` 持有无梯度独立 snapshot model，配置同步周期，记录同步信息。训练额外 baseline forward 不改变部署单实例一次前向要求。
- 训练日志已记录：policy loss、makespan mean/median/std、advantage mean/std、entropy、gradient norm、skip ratio、per-instance K rollout 时间分布、forward 时间和 snapshot 元数据。

## 2. A1-a：策略收缩和训练正确性验证（旧日志重新解析）

原始训练日志和 checkpoint 已保留在 `results/diagnostics/A1/`，不得覆盖。原始 gate 的字段 `initial_greedy_makespan` / `final_greedy_makespan` 实际取自 external `greedy_schedule`，而不是 policy 的 greedy decode，因而不能用于衡量 policy 的学习改善。

修正协议明确区分：

- `oracle_makespan`：MILP 证明的全局最优；
- `external_greedy_makespan`：独立 critical-path-first 外部启发式；
- `ca_heft_makespan`：外部容量感知 HEFT；
- `initial_policy_greedy_makespan` / `final_policy_greedy_makespan`：模型输出静态 score 后的 deterministic greedy decode；
- `initial_sample_mean_makespan` / `final_sample_mean_makespan`：K=8 sampled rollout 均值。

A1-a 重新解析脚本 `scripts/reanalyse_a1a.py` 从既有 history、gate 和 final checkpoint 读取数据，并将新报告写入独立的 `results/diagnostics/A1-a/report.json`。它不训练、不修改旧日志。当前历史证据为：external Greedy=18、CA-HEFT=18、记录的 policy greedy 从 38 降至 18、sample mean 从 33.375 降至 18、Oracle=18；终止时 LOO advantage std 与 gradient norm 都为 0。final checkpoint replay 重新用独立 Validator 检查 policy greedy 和 K=8 sampled traces。

A1-a gate 要求 100% replay 可行率、sample mean 明显下降、final best-of-K 与 policy greedy 均达到 Oracle、完整 oracle-gap closure、末尾 LOO advantage/gradient 归零、所有指标有限且 replay trace 无非法时间推进。它是“模型可学到最优调度”的验证，不是“优于已经最优的 external Greedy”的验证。

## 3. A1-b/A1-c 实例筛选（仅筛选，未训练）

A1-b 固定使用 seed 3000–3099 的 100 个 8–10 task 整数 tick 候选。所有候选均必须记录 MILP、external Greedy、CA-HEFT 和独立 Validator 结果；选择不得使用任何训练指标。接受条件是 MILP `optimal`、所有调度可行、external Greedy 相对 Oracle gap 至少 5%、绝对差至少 2 ticks；优先选择 5%–30% gap，并按 seed 升序取第一个。

A1-c 使用预声明的六任务手工 fixture：root 先占 y 资源，独立 blocker 占 x，critical child 也占 x 且解锁长 y descendant。主动 wait root 完成、先运行 critical child 后可让 blocker 与 long descendant 重叠；non-delay restricted search 必须先启动 blocker 并产生严格更长的 makespan。`oracle.exhaustive_oracle` 现在提供独立 `solve_exhaustive_with_wait` 与 `solve_exhaustive_without_wait` 入口，并记录最优轨迹的 active-wait decision。

## 4. 未通过项与严格停机状态

根据批准的门槛，以下均**未启动**：

- A2（12–20 task）单实例过拟合；
- 固定 16–32 实例过拟合；
- 三 baseline × 3 seeds 的 200–300 update 动态筛选；
- 任何 800-update 正式工程诊断；
- Phase 2 通信、成本、能耗和预算扩展。

## 5. 下一步

在检查 A1-b/A1-c 的固定实例和报告并得到明确批准前，不运行 A1-b 或 A1-c 的 300-update 训练。批准后，训练命令必须读取已选择的冻结实例，而不是重新按随机 seed 生成实例；每类先只运行一个 seed、最多 300 updates，并记录 policy/external heuristic 分离的指标。
