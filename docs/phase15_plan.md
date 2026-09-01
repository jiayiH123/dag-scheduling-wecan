# Phase 1.5 实施计划：可信 WeCAN 研究基线

> **状态：待用户确认后实施。** 本计划严格停留在 Phase 1.5：不会引入通信时延、成本、能耗、预算、MIP 专家预训练或 Phase 2 的方案 A/B。本文只设计实现、单元测试和**短筛选诊断**；不会在确认前启动 800-step / 9-run 正式训练。

## 0. 目标、验收边界与不可宣称事项

Phase 1.5 的目标是把当前最小闭环升级为可核验的研究基线：论文架构一致性更高、REINFORCE 比较可重复、调度器由独立 Oracle 交叉验证、训练由分层诊断门槛控制。

完成后仍只能称为 **WeCAN 忠实重实现**，不能称官方或完全复现：原论文没有公开 WeCAN 源码、真实训练数据和若干关键训练细节。即使 Phase 1.5 的训练结果超过 Greedy/CA-HEFT，也只对本工程固定随机数据和配置有效，不能等同论文结果。

### Phase 1.5 Oracle 验收门槛

1. 十个手工 fixture 全部通过。
2. 至少 50 个 4–7 任务随机实例中，独立 MILP 与穷举的已证明最优 makespan 在 `EPS=1e-6` 容差内全部一致。
3. 全部 Oracle 调度由独立 Validator 通过。
4. Random、Greedy、CA-HEFT、标准 HEFT（若适用）和 WeCAN 输出均通过独立 Validator。
5. 只有 `heuristic_makespan < oracle_makespan - EPS` 才视为启发式错误地优于已证明 Oracle；出现即抛出失败并停止对应实验。
6. CBC 未证明最优的解必须标记为 `feasible_not_proven_optimal`，不能作为 global optimum label。

### Phase 1.5 训练诊断门槛

1. 先对一个固定 10–20 任务实例使用默认 K=8 leave-one-out 过拟合；以 MILP 目标为参照，需明显降低初始 greedy makespan 或减少与 Oracle 的相对 gap，才可继续。
2. 再对固定 16–32 个 12–30 任务实例以 1 个 seed 过拟合；若无法稳定学习，先检查 score、mask、skip、轨迹 log-probability、reward/advantage、梯度和 entropy，不运行动态训练。
3. 固定小数据集门槛通过后补齐 3 seeds。
4. 动态阶段仅比较三种主基线：`batch_global_mean`、`instance_leave_one_out`、`greedy_rollout`；每种在同一 3 个 seeds 下先运行 200–300 updates 筛选，再对通过项运行至少 800 updates，共最多 9 次正式训练。

## 1. 目标文件结构

```text
baselines/
  algorithms.py                 # 保留；重命名原 heft_schedule 为 ca_heft_schedule
  standard_heft.py              # 新增：标准单资源占用语义 HEFT（仅在适用配置运行）
configs/
  phase1_smoke.yaml             # 更新：小模型/软参数 skip 的快速测试 profile
  phase1_paper.yaml             # 新增：论文 512→128 / LDD-8×16 / sigmoid skip profile
  phase15_diagnostics.yaml      # 新增：分层过拟合、K=8 LOO、训练预算与三 seed 配置
oracle/
  __init__.py
  milp_oracle.py                # 新增：PuLP/CBC 独立全局最优 Oracle
  exhaustive_oracle.py          # 新增：极小实例独立枚举 Oracle
  common.py                     # 新增：仅共享结果数据类和容差；不得共享排程核心逻辑
scheduler/
  generator.py                  # 扩展 trace：entropy、step probability、skip/event 元数据
  types.py                      # 保留 DTO，不再承担独立 validator 责任
  validator.py                  # 新增：完全独立事件线 Validator
models/
  wecan.py                      # 改造：paper/smoke profile、512→128、完整 LDD mask、skip modes
training/
  reinforce.py                  # 改造：五种 baseline、K trajectory、snapshot 同步、诊断指标、更新数控制
  diagnostics.py                # 新增：单实例/固定集/动态集的门槛编排与曲线持久化
evaluation/
  metrics.py                    # 更新：CA-HEFT、HEFT、median、双方胜率、跨 seed 汇总
scripts/
  train.py                      # 扩展 profile/baseline/K/max-updates/实验标签
  run_diagnostics.py            # 新增：严格按分层门槛运行，不默认触发正式训练
  run_oracle_validation.py      # 新增：手工和 50 随机交叉验证，输出机器可读报告
  compare_reinforce.py          # 新增：仅显式指定时运行 3×3 筛选/正式实验
  evaluate.py                   # 更新：使用 Validator，输出新增统计
  plot_results.py               # 更新：训练曲线、seed 曲线、诊断图
tests/
  fixtures.py                   # 新增：十个手工边界实例与随机极小实例生成
  test_milp_oracle.py
  test_exhaustive_oracle.py
  test_oracle_crosscheck.py
  test_handcrafted_edge_cases.py
  test_validator.py             # 新增：恶意无效 schedule/trace
  test_phase15_model.py         # 新增：LDD mask、fold、paper/smoke shape 与 skip mode
  test_reinforce_baselines.py   # 新增：五类 advantage、K、snapshot 同步与日志指标
results/
  oracle/                       # 运行时：逐实例 Oracle JSON/CSV
  diagnostics/                  # 运行时：每层训练曲线、门槛报告
  phase15/                      # 运行时：3×3 独立 seed 结果和汇总
```

原 `configs/phase1.yaml` 将保留为兼容别名或迁移指引，避免现有 smoke checkpoint 不能加载；README 会明确推荐新配置名。

## 2. 论文一致性改造

### 2.1 `paper` 和 `smoke` 两套模型配置

| 项 | `paper` profile | `smoke` profile |
|---|---:|---:|
| 初始 task/pool embedding | 512 | 128（快速测试） |
| 初始 WeCA | 512 维，8 heads | 128 维，8 heads 或显式覆盖 |
| LDDGNN | 8 层，512 维，16 heads | 小规模层数，显式标记非论文配置 |
| 低维投影 | task/pool 各自 `Linear(512,128)`，在初始 WeCA + 8 LDD 层后 | 可选恒等或指定投影 |
| 后续交替 WeCA | 128 维，8 heads | 128 维，8 heads |
| decoder | 128 维 | 128 维 |
| `Dmax` | 500 | 500（保持语义） |
| skip 参数 | 二隐藏层 64、GELU、Sigmoid | 可选择 sigmoid / softplus ablation |

论文没有明确 post-projection alternating WeCA 的层数，也没有给出 512→128 投影细节。计划将 `alternating_weca_layers` 保留配置项（默认固定为当前 2），并在 `phase1_paper.yaml` 和论文差异文档中把它标记为必要实现假设。

### 2.2 完整 LDDGNN：距离、折叠、偏置和 per-head mask

新增可单测的 LDD 特征函数，不把 LDD 与 attention 代码混杂：

1. 对任务对 `(v,w)`：若 `v` 可到达 `w`，取有向最长路径边数为正；若 `w` 可到达 `v`，取负值；同一弱连通分量但互不可达为 `+∞`；不同弱连通分量为 `-∞`。正负方向是论文未明说的必要约定，会记录。
2. `Dmax=500`，距离折叠严格按论文附录 F.1，并区分有限距离与无穷距离：
   - 有限 `d∈[-499,499]`：保持 `d`；
   - 有限 `d>=500 → 499`；
   - 有限 `d<=-500 → -499`；
   - `+∞ → 500`；
   - `-∞ → -500`。
3. 用 `2*Dmax+1 = 1001` 行、每 head 一个标量 bias 的可学习表；零基下标为 `folded_distance + 500`。边界必须单测：有限 `-501/-500/-499` 分别折叠为 `-499/-499/-499`、bias index 均为 `1`；有限 `499/500/501` 分别折叠为 `499/499/499`、bias index 均为 `999`；`+∞` 折叠为 `500`、index `1000`；`-∞` 折叠为 `-500`、index `0`。
4. 16 heads 划为 8 组，每组 2 heads。第 `j` head 使用集合 `N[floor(8*j/16)+1]`：

| heads（零基） | 掩码集合 |
|---|---|
| 0–1 | `N1 = Z ∪ {-∞,+∞}`（全局） |
| 2–3 | `N2 = {+1}` |
| 4–5 | `N3 = {-1}` |
| 6–7 | `N4 = {+2}` |
| 8–9 | `N5 = {-2}` |
| 10–11 | `N6 = {d∈Z : d>=3}` |
| 12–13 | `N7 = {d∈Z : d<=-3}` |
| 14–15 | `N8 = {+∞}` |

未落入对应 head 集合的 logits 加 `-inf`。**允许某节点在专用 head 上没有任何合法 key**：实现必须先识别该 query/head 的全 mask 行，令该行 attention 权重和 message 显式为零，避免 `softmax([-inf,...,-inf])` 产生 NaN。不得为了避免空邻域而开放 self-loop 或任何原本被 mask 的节点；全局 `N1` head 仍按论文定义可见全部关系，但专用 head 必须严格服从其集合。测试会验证 mask 表、16-head 映射、`±∞`、边距离、长距离截断、空邻域 head 的零输出/无 NaN，以及自环值以 `0` 处理这一论文未规定的显式假设。

### 2.3 skip 参数模式

`WeCANConfig.skip_parameterization` 提供：

- `paper_sigmoid`（默认 paper profile）：`MLP(128→64→64→3)`，两隐藏层 GELU，末端 `sigmoid`。其输出在 `(0,1)`，满足论文要求的正性。
- `softplus`：保留现行 `softplus(raw)+epsilon`，只作为消融。

两种模式均记录至 checkpoint / evaluation metadata，确保结果可追溯。

## 3. 三个相互独立的正确性模块

共同只允许共享：`DAGInstance` 输入 DTO、`TaskPlacement/Schedule` 输出 DTO、`EPS=1e-6` 常量和 `OracleResult` 数据类。**不得复用**生成器的 action mask、HEFT 的 earliest-start、或任一 Oracle 的核心排程算法。Validator 也不能通过调用 generator 来判断合法性。

### 3.1 `scheduler/validator.py`：独立 Validator

`validate_schedule(instance, schedule, trace=None, eps=1e-6)` 会建立完整事件时间线，而非只在启动时刻检查：

1. task id 必须刚好覆盖 `0..n-1` 一次；检查遗漏、重复、越界。
2. pool id 合法、兼容性严格正；开始时间非负、有限；`end>=start`。
3. `end-start` 与 `base_duration/compatibility` 在统一容差内一致。
4. 每条 DAG 边的父 end 不晚于子 start。
5. 使用所有 start/end 去重排序得到事件区间；对每个池、每一个相邻的非零区间，以中点判定运行集合，逐资源维度累计并对比容量。
6. `schedule.makespan` 必须等于所有 task end 的最大值。
7. 如果提供 generator trace：检查 action 语义与 placement 对齐、skip 前确有运行任务、skip 后时间严格推进到最早完成事件、无时间倒退、`len(actions)<=2n`、没有重复 dispatch、最终未调度集合为空。对外部 schedule 而没有 trace，返回“trace not supplied”，而不是伪称验证过 skip。

Validator 输出结构化 violations，附带每个事件区间的 usage，供 Oracle 和测试报告写入 JSON。

### 3.2 `oracle/milp_oracle.py`：PuLP/CBC 单一时间索引 MILP

**用途和精确性边界：** 只为 4–10 任务的 correctness Oracle 和小规模最优参考，不进入默认训练策略。本阶段只将其声明为“**对整数或可精确有理缩放小实例的全局最优 Oracle**”，不声明为任意连续实数时间模型的精确 Oracle。

#### 时间尺度与输入资格

1. Oracle fixture 默认使用整数基础时长、兼容性和实际执行时长，因此 `time_scale=1`。
2. 有限小数时长只能通过显式 `time_scale` 精确转为整数 tick。实现优先从输入文本/显式配置的 `Decimal` 或 `Fraction` 构造统一公分母；禁止由二进制 `float` 猜测比例或静默四舍五入。
3. 对每个兼容对，必须验证 `base_duration / compatibility * time_scale` 恰为整数 tick；容量、需求和时间恢复也必须在原始单位与 tick 单位间一致。
4. `time_scale`、转换后的 duration ticks 和时间上界均写入 `DAGInstance` 的 Oracle view、`MilpOracleResult` 与 JSON 报告。
5. 若所需统一比例超过配置 `max_time_scale`，或任一值不能精确转换，返回 `unsupported_time_scale`，不做近似求解。

#### 唯一正式变量

令任务为 `i,j`，资源池为 `p`，资源维为 `r`，离散时间刻度为 `t,τ∈{0,...,H}`。**MILP 只使用如下启动变量，不混入连续开始时间或 pairwise ordering 两套模型：**

- `y[i,t,p] ∈ {0,1}`：任务 `i` 在整数 tick `t` 于兼容资源池 `p` 启动。
- `Cmax_ticks >= 0`：最小化的最终 tick makespan。

这里 `duration_ticks[i,p]` 是输入经过显式 `time_scale` 后的已知整数。优先以经过 Validator 的 CA-HEFT/其他可行启发式 schedule 的 makespan tick 作为有限上界 `H`；若该启发式产生异常、无可行解或无法通过 Validator，必须使用**独立串行可行调度后备方案**：按合法拓扑序，每次只执行一个任务，为该任务选择任一兼容且单任务需求不超过 pool 容量的资源池，并在前一任务完成后才启动下一任务。因为每个 Oracle-eligible task 都有至少一个此类 pool，该后备方案提供有效有限 `H`。实现还可使用 DAG 的 earliest/latest start bounds（由该 `H` 反向传播）裁剪 `y[i,t,p]` 的时间域；该裁剪必须保持可行域完整性，并有单元测试与未裁剪小实例对照。

#### 目标

\[
\min C_{\max}^{\mathrm{ticks}}.
\]

#### 约束

1. **每任务恰好一次兼容启动/分配**：
   \[
   \sum_{p\in P_i}\sum_{t=0}^{H-duration_{i,p}} y_{i,t,p}=1.
   \]
2. **DAG precedence**：令启动 tick 的线性表达为
   \(S_i=\sum_{p,t}t\,y_{i,t,p}\)，完成 tick 为
   \(F_i=\sum_{p,t}(t+duration_{i,p})y_{i,t,p}\)。对每条 \((i,j)\in E\)：
   \[
   S_j\ge F_i.
   \]
3. **任意 pool、tick、资源维度的累积容量**：使用半开区间 `[start,end)`。任务 `i` 在 `τ` 运行的线性指示由所有满足
   \(t\le τ < t+duration_{i,p}\) 的 `y[i,t,p]` 求和；因此：
   \[
   \sum_i demand_{i,r}
   \sum_{t:\,t\le τ<t+duration_{i,p}}y_{i,t,p}
   \le capacity_{p,r},
   \quad \forall p,τ,r.
   \]
4. **makespan**：
   \[
   C_{\max}^{\mathrm{ticks}}\ge F_i,\quad\forall i.
   \]
5. **非负时间**由 `t>=0` 的变量域自然保证。

从 `y` 的唯一 1 直接恢复 assignment、start、end 与 `Schedule`；再转换回原始时间单位，且由独立 Validator 同时检查原始时间 schedule 和 tick schedule 的一致性。

#### CBC 最优状态判定与审计

4–7 任务正式交叉验证默认不设置 time/node/solution limit；若测试故意设置限时，必须记录它并验证不会被误标。每次 CBC 求解都启用并保存原始日志到结果目录，解析或保守判定以下字段：`proven_optimal`、`time_limit_reached`、`node_limit_reached`、`solution_limit_reached`、`abnormal_termination`、incumbent objective、best bound、gap（若 CBC 输出可获得）。

`MilpOracleResult.status="optimal"` **仅当同时满足**：

1. CBC 日志/终止原因明确表明完整搜索完成并证明 optimal；
2. 未触发 time limit、node limit、solution limit 或其他中止；
3. 有可行 incumbent；
4. incumbent objective 与可取得的 best bound/gap 在 `optimality_tolerance` 内一致；若 bound/gap 不可取得，必须仍有 CBC 的明确 optimal proof，不能仅靠 PuLP 状态字符串；
5. 从 `y` 恢复的 schedule 通过独立 Validator，且 tick/原始单位一致。

仅能确认可行 incumbent、PuLP 返回歧义字符串、日志缺失或解析失败时，保守标记 `feasible_not_proven_optimal`；其他状态为 `infeasible`、`unbounded`、`unsupported_time_scale`、`solver_error` 或 `invalid_solution`。非 `optimal` 结果不能参与与 Exhaustive Oracle 的“全局最优一致”验收。

### 3.3 `oracle/exhaustive_oracle.py`：独立完整穷举

**范围：** 仅 4–7 任务；8–10 任务绝不无条件运行完整枚举。

#### 搜索空间、时间语义与完整性

递归状态独立维护：当前绝对时间、未调度任务、已完成任务、运行任务的 `(task,pool,absolute_end,resource_demand)` 签名、每 pool 当前资源占用/可用容量、已选 assignment、dispatch history 和 wait history。半开执行区间严格为 `[start,end)`：任务可在另一任务完成的**同一时刻**启动。

每个状态先依据已经完成的父任务、当前容量和兼容性重新计算 ready/feasible dispatch actions，并分支：

1. 在**同一当前时刻**依次选择所有可启动的 `(task,pool)` 兼容、容量可用动作；递归不会在第一次 dispatch 后自动推进时间，因此能覆盖任意数量任务的同刻并行启动。
2. 只要存在运行任务，始终额外保留一个 `wait_to_next_completion` 分支：即使仍有可行 task，也可主动 wait/skip 到最近 absolute completion event；到达后释放所有在该时刻完成的任务，再重新计算 ready set 与容量。

因此它覆盖：所有兼容 assignment、所有 DAG 合法 dispatch order、不同并行集合，以及非 non-delay schedule 所需的主动等待/skip。不会固定一个拓扑序，也不会调用 WeCAN generator。

#### 剪枝与 canonical state

- DAG：未完成父任务不可 dispatch。
- 兼容与单任务容量：预计算排除。
- Canonical state key 至少包含：当前时间、未调度/运行/已完成集合、每个运行任务的绝对完成时间及 pool、每 pool 当前资源占用、以及未来会影响资源释放的任务—pool assignment；**不得**只以已调度任务集合合并状态。
- 对相同 canonical future state，只保留已知 makespan 下界/历史字典序更优的路径；绝不把仍可能导致不同资源释放的状态合并。
- 下界：`max(current_time, 已放置任务的最大 end, remaining DAG critical-path optimistic bound)` 不小于 incumbent 时剪枝。
- 对称性：pool capacity + compatibility column 完全相同的 pool 按最小 canonical pool id 分支；完全同质、同 parent/child 集合、同 duration/demand/compatibility 的等价任务仅保留最小 id 的先行 dispatch。
- 容量明显不可行立即剪枝。

输出 `ExhaustiveOracleResult`：最优 makespan、最优 assignment、合法 dispatch/skip order、Schedule、search_nodes、pruned_nodes、`complete_search=True/False`、运行时间和 Validator 摘要。若达到明确 `node_limit`/`time_limit`，标为 incomplete，不能参与“与 MILP 相同最优”的验收。

## 4. 必备 fixture 与测试设计

`tests/fixtures.py` 提供下列**离散整数时间**实例，明确预期/Oracle：

1. 单链 DAG。
2. 菱形分支—汇合 DAG。
3. 宽 DAG（多任务可并行）。
4. 容量刚好饱和。
5. 容量不足、必须延迟。
6. 任务—资源不兼容。
7. 多个最优 assignment / 同一 makespan。
8. 主动 wait/skip 才达到最优的反例：资源受限任务先运行会阻塞关键短任务；最优调度要求暂不启动一个当前可行的任务并等待资源释放/父任务完成。穷举结果 history 必须含 wait，验证器必须验证该 trace。
9. 所有任务只能放在一个资源池的退化实例。
10. 两资源池完全对称实例。

### 4.1 新增测试文件

| 文件 | 关键断言 |
|---|---|
| `test_validator.py` | 每项独立 violation：漏/重 task、错误 duration、兼容性、precedence、负时间、capacity 中间区间峰值、错误 makespan、非法 skip/回退/死循环 trace。 |
| `test_milp_oracle.py` | 每个手工实例求出 `optimal` 且经 Validator；time limit 导致 status 不能误标 optimal；8–10 task fixtures 仅 MILP + Validator。 |
| `test_exhaustive_oracle.py` | 4–7 task fixtures `complete_search=True`；返回 assignment/order/计数；等待反例包含 wait；测试每个剪枝不改变与无剪枝小实例的 optimum。 |
| `test_oracle_crosscheck.py` | 固定 seeds 生成 ≥50 个 4–7 task、不同 density/capacity/compatibility 桶；MILP optimum == exhaustive optimum；双方 validator pass；Random/Greedy/CA-HEFT/WeCAN 均 validator pass。仅当 `heuristic_makespan < oracle_makespan - EPS`（统一 `EPS=1e-6`）才判定启发式非法优于已证明 Oracle 并立即失败；容差内的数值波动不误报。 |
| `test_handcrafted_edge_cases.py` | 十个 fixture 的完整交叉验证与对称性/退化性断言。 |
| `test_phase15_model.py` | 512→128 shapes、8 LDD layers、16 heads、N1..N8 masks、Dmax folds、bias index；其中强制覆盖有限 `-501/-500/-499/499/500/501` 与 `±∞` 的折叠和 index；paper sigmoid/softplus mode、一次 forward。 |
| `test_reinforce_baselines.py` | K=4/8/16 shape，四种 baseline advantage 数学、LOO 不引用自身、greedy rollout 无梯度、所有训练指标有限且有值。 |

### 4.2 Oracle 测试报告

`run_oracle_validation.py` 和 pytest failure attachment 写 `results/oracle/crosscheck_<seed>.jsonl`。每行包含：

```json
{
  "instance": "random-00017",
  "tasks": 6,
  "pools": 3,
  "milp_status": "optimal",
  "milp_makespan": 31.0,
  "exhaustive_makespan": 31.0,
  "absolute_difference": 0.0,
  "milp_seconds": 0.12,
  "exhaustive_seconds": 0.04,
  "milp_validator": {"feasible": true, "violations": []},
  "exhaustive_validator": {"feasible": true, "violations": []},
  "crosscheck_passed": true
}
```

任一 mismatch、non-optimal MILP、incomplete exhaustive、validator 失败或启发式优于 Oracle 都会使 pytest/脚本以非零退出。

## 5. 基线命名和语义

| 新名字 | 实现和语义 |
|---|---|
| `Random` | 随机静态 task-pool score，经公共可行生成器解码。 |
| `Greedy` | critical-path-first 静态优先级 + 最短实际执行时间偏好，经公共可行生成器解码。 |
| `CA-HEFT` | 现有 `heft_schedule` 改名 `ca_heft_schedule`：在标准 HEFT 的 upward-rank/earliest finish 思路上，对每个 pool 的完整多维容量事件线寻找最早可行插入；适用于本项目“同一 pool 可并行但容量受限”的语义。 |
| `Standard HEFT` | 新实现，严格采用经典 HEFT 的“每处理器单任务 timeline、最早插入”假设；仅当每个 task demand 等价于占满所选 pool 或以显式 `--allow-standard-heft` 标注为近似对照时纳入汇总。对于一般可并行累积容量实例，它是更保守的可行启发式，不是本问题的容量完整模型。 |

所有评估文件把旧 `heft` label 迁移为 `ca_heft`；README 和阶段报告解释这一重命名，避免把容量感知变体伪称为标准 HEFT。

## 6. REINFORCE 基线、K 条轨迹与诊断指标

### 6.1 轨迹生成原则

对一个实例只运行模型**一次**获得静态 score 和 skip 参数；K 条 sampled trajectory 全部复用这些张量，改变的是 generator 的随机采样和动态 mask。greedy baseline 同样复用已得输出，不进行第二次 forward。因此“单次前向”仍按每个实例/每次推理成立。

训练的 `rollout_time_seconds` 以每 instance 的 K 条轨迹合计时间、以及单 trajectory 时间的均值/中位数/p50/p95 写入日志。

### 6.2 五种可配置 baseline（其中三种为正式动态比较主项）

训练器支持五种模式。为区分论文式独立贪心 baseline 与“复用当前输出”的低成本消融，动态正式比较只比较 `batch_global_mean`、`instance_leave_one_out`、`snapshot_greedy_rollout` 三项：

| 名称 | K | baseline \(b\) | advantage \(A=M-b\) |
|---|---:|---|---|
| `batch_global_mean` | 1（也支持 K） | batch 内全部 rollout makespan 的 detached mean | `M - mean(M_all)` |
| `instance_mean` | 4/8/16 | 同一 instance K 条轨迹 makespan 的 detached mean | `M_ik - mean_k(M_ik)` |
| `instance_leave_one_out` **默认** | 4/8/16 | 其余 K-1 trajectory makespan 的 detached mean | `M_ik - mean_{l!=k}(M_il)` |
| `current_policy_greedy` | 1 或 K | 当前策略对同一实例已计算的静态输出进行 greedy decode（detach、无额外 forward） | `M_ik - M_i^current_greedy`；仅作消融 |
| `snapshot_greedy_rollout` | 1 或 K | 无梯度独立 baseline model 的 greedy rollout | `M_ik - M_i^snapshot_greedy`；论文式 greedy rollout baseline |

`snapshot_greedy_rollout` 持有独立 `baseline_model`：该模型始终 `eval()`、参数 `requires_grad=False`、不参与 optimizer；每 `snapshot_sync_interval_updates` 个 update 从当前 policy 同步 `state_dict`。同步周期、上次同步 update、baseline model 配置/参数哈希和是否发生额外 baseline forward 均写入配置、checkpoint 与每 update 日志。训练阶段这种额外 forward **不改变部署推理的一次网络前向定义**：部署/评测的单实例 policy 推理仍只调用当前 model 一次；额外 forward 仅发生在训练的 baseline 估计路径。
最小化 makespan 的 REINFORCE loss：

\[
L = \operatorname{mean}_{i,k}\left[A_{i,k}\log p_\theta(\omega_{i,k}\mid\mathcal P_i)\right].
\]

不添加 PPO clip、critic、entropy bonus 或 advantage normalization 作为默认行为；如果未来加消融会另设配置并明确不是原论文方法。`instance_mean` 由于包含自身会在 K=1 时 advantage 恒为 0；训练器拒绝此退化组合。LOO 要求 K≥2。

### 6.3 每 update / epoch / run 记录

JSONL + 汇总 JSON/CSV 记录：

- `policy_loss`；
- `mean_makespan`、median、std；
- `mean_advantage`、`advantage_std`；
- 按每个 action categorical 计算的平均 `entropy`；
- 反向传播后、optimizer step 前的 global `gradient_norm`；
- `skip_ratio = #skip / #decisions`；
- 每实例 K trajectories 的 `rollout_seconds_total`、mean、median、p50、p95 和 K；
- `forward_seconds_per_instance`，完整 generation seconds；
- 可行率、训练数据集/配置哈希、seed、checkpoint、profile、baseline mode。

每一个 seed 独立目录，绝不只存 aggregate。

## 7. 分层训练与运行预算（不在确认前启动长训练）

### 阶梯 A1：精确小实例过拟合

- 一个固定、冻结的 **8–10 task** DAG，必须由时间索引 MILP 得到 `optimal` 证明，保存 `C*`、CBC 日志和 `time_scale`。
- 默认 `instance_leave_one_out, K=8`。
- 每 20 updates 评估 deterministic greedy、sample best-of-8 和精确 Oracle gap：`(C-C*)/C*`。
- 最多 300 updates 的筛选预算。若初始 greedy makespan 或精确 Oracle gap 未按配置阈值（初始设为 >=5%）明显下降，标记 `blocked_A1`，停止后续层级。排查顺序：forward count → mask/action support → trace transition → log-prob/entropy → advantage sign/scale → gradient norm → skip ratio。

### 阶梯 A2：中等单实例过拟合

- 一个独立固定 **12–20 task** DAG；不默认要求、也不默认宣称时间索引 MILP 能证明该规模的最优。
- 默认 `instance_leave_one_out, K=8`，最多 300 updates；报告相对初始策略和 Greedy 的 makespan 改善。
- **只有** MILP 明确返回并审计为 `optimal` 时，才附带报告 global optimality gap；否则只报告相对初始策略/Greedy/CA-HEFT 的变化与 MILP status。
- A2 未明显优于初始策略或 Greedy 时标记 `blocked_A2`，不进入固定小数据集。

### 阶梯 B：固定小数据集过拟合

- 固定不重叠 16–32 train instances，12–30 tasks；验证集固定且独立。
- 先一个 seed、200–300 updates，比较开始/结束 train 和 validation 的 Greedy/Oracle gap（对可求的子集）。
- 学习门槛通过再补同一配置为 3 seeds；否则停止，不进入动态训练。

### 阶梯 C：动态随机训练快速筛选

- train/validation/test 预先生成并冻结、互不重叠，不在 epoch 内动态临时生成以保证可复现；“动态”指混合随机实例分布训练而非固定小集合过拟合。
- 训练：20–50 tasks、3 pools、2 dims；batch size 64，OOM 回退 32；每 instance K=8。
- 三种主 baseline：`batch_global_mean`、`instance_leave_one_out`、`snapshot_greedy_rollout`。`current_policy_greedy` 只作显式消融，不计入正式 3×3 比较。
- 3 相同随机 seeds × 3 baseline = 9 runs；每 run 先 200–300 updates。每 run 单独记录是否通过（训练稳定、无 NaN、可行率 1、验证相对 Greedy 不出现系统性恶化的预先设定阈值）。

### 阶梯 D：正式工程诊断

- 仅对通过 C 的同一 9 个 run 继续/重启至至少 800 updates。
- 这三个 seeds 是 Phase 1.5 工程诊断，**不是**最终论文充分统计证据。
- 固定测试三档：20–50、50–100、100–200 tasks；全部与训练/验证实例互不重叠。
- 每档报告每 seed 和跨 seed：makespan mean/median/std、相对 Greedy/CA-HEFT improvement、严格胜过 Greedy/CA-HEFT 的实例比例、可行率、网络 forward、完整生成时间、entropy、gradient norm、skip ratio。

不会预先保证任一 run 超过 Greedy 或 CA-HEFT；若未通过，报告原因和数据，停在 Phase 1.5，不进入 Phase 2。

## 8. 文档、依赖和最终交付

1. `requirements.txt` 增加 `PuLP>=3.0`；README 说明 `pip install -r requirements.txt` 后用 `python -c "import pulp; print(pulp.PULP_CBC_CMD().available())"` 检查 CBC。
2. `docs/wecan_analysis.md` 更新附录 F.1/F.2 的 paper profile、一致性差异和不确定项。
3. `docs/phase15_report.md` 记录：Oracle 结果、手工实例、50 个交叉验证摘要、测试数、每层训练是否达门槛、9 seed run 的实际结果、失败原因、以及仍未达到论文的原因。
4. README 增加 Oracle 和诊断命令：

```bash
pytest -q tests/test_milp_oracle.py tests/test_exhaustive_oracle.py \
  tests/test_oracle_crosscheck.py tests/test_handcrafted_edge_cases.py
python scripts/run_oracle_validation.py --random-count 50 --seed 2026
python scripts/run_diagnostics.py --stage single-instance --config configs/phase15_diagnostics.yaml
python scripts/run_diagnostics.py --stage fixed-set --config configs/phase15_diagnostics.yaml
# 仅在前两关通过、且用户/显式命令允许时：
python scripts/compare_reinforce.py --stage screen --seeds 2000 2001 2002
python scripts/compare_reinforce.py --stage formal --seeds 2000 2001 2002 --updates 800
```

## 9. 实施顺序

1. 抽出独立 Validator，迁移当前测试/评测到它；先用现有 smoke schedule 回归验证。
2. 建立十个手工 fixture（默认整数时间）、时间索引 PuLP/CBC MILP 与独立穷举器；实现各自单文件 Oracle tests。
3. 添加 ≥50 个 4–7 task 极小随机实例交叉验证和报告脚本；只有全部 MILP `optimal`、穷举 `complete_search` 且相等时，才允许改模型。
4. 改造 WeCAN 的 paper/smoke profile、LDD folding/bias/8 类 16-head mask、512→128 和 skip modes；补模型单测。
5. 重命名 CA-HEFT、实现 Standard HEFT 并更新 metrics/README。
6. 改造 generator trace 与 REINFORCE K trajectory / 四 baseline / diagnostics；补数学和日志单测。
7. 仅在 Oracle、模型和 REINFORCE 单测全部通过后运行阶梯 A；再依据门槛决定 B。A/B 均通过后才显式运行 C 的 200–300 update 筛选。
8. 仅对筛选通过的 runs 运行 D 的 800 update、三 seed 工程诊断；写报告而不是预设结论。

## 10. 已知风险和处理

- **累积容量的 MILP 精确性：** 将 Oracle 限制在整数/明确量化的 4–10 task fixture，不将 tick MILP 伪称为任意连续实数全局 Oracle。
- **穷举爆炸：** 仅 4–7 task；超预算必标 incomplete，绝不把 incumbent 当作 optimum。
- **CBC 状态歧义：** `Optimal` 证明与 Validator 是两道独立门；其他状态不生成 optimal 标签。
- **paper 模型显存/耗时：** `paper` profile 仅用于必要模型一致性测试、单实例/小集诊断；正式 9-run 可通过配置采用相同语义的受控 resource profile，但报告必须注明模型 profile。若必须缩小，不能把结果称为 paper architecture results。
- **“动态训练”可重复性：** 预生成冻结数据集而非实时无记录抽样；每个 split、seed 和配置都有哈希。
- **训练无改善：** 这是有效诊断结果；保存 logs/checkpoints/plots，排查后再决定，绝不跳过门槛进入 Phase 2。
