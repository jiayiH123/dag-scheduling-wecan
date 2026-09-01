# 项目概览：从 WeCAN 忠实重实现到固定任务优先级下的节点分配

> **文档生成日期：** 2026-08-25  
> **审查基线：** Git `45fda42`（`evaluate phase21c graph aware allocator`）  
> **项目目录：** `/lpai/volumes/ss-sai-bd-ga/huangjiayi/codes_my_own/dag_0729`  
> **说明：** 本文基于目标论文、《问题建模.pdf》、当前源码、实验产物、阶段文档和 Git 历史交叉整理。文中的“节点选择”特指为任务选择计算节点/资源池，而不是从 ready set 中选择下一个 DAG 任务。

---

## 1. 一页结论

这个项目已经不再只是“复现 WeCAN 论文”，而是形成了两条有继承关系、但研究问题不同的技术路线：

1. **Phase 1：WeCAN 忠实重实现与正确性基础设施。**  
   已实现异构资源池、DAG 依赖、累积多维容量、兼容性、一次网络前向、递减 skip、REINFORCE，以及独立 Validator、MILP/穷举 Oracle。由于论文没有公开代码、原始训练数据和全部细节，这一部分只能称为 faithful reimplementation，不能称为官方或完全复现。

2. **Phase 2：面向自定义场景的通信感知调度。**  
   当前已实现非竞争通信时延及其在数据结构、generator、Validator、MILP、穷举搜索中的一致语义；尚未实现《问题建模.pdf》中的成本、能耗/电费和预算约束。

3. **当前研究主线（Phase 2.1c）：固定任务优先级 + 离线节点分配。**  
   系统先按一个静态、拓扑合法的任务顺序，为全部任务选择资源池；完整 assignment 冻结后，再由同一静态优先级驱动无主动等待的事件式 priority-list realizer。当前重点已经从“直接复刻 WeCAN 的联合在线 task-pool+skip 策略”转为：
   - 固定优先级会损失多少全局调度能力？
   - 固定优先级后，节点分配还有多少可学习空间？
   - 一个一次前向的 allocator 能否学会这部分空间？

4. **当前最重要的实验证据是“结构上有空间，学习上尚未成功”。**
   - 32 个小型开发实例上，三个固定优先级策略的 `PriorityGap` 都为 0；这只说明在该小样本、N=4/6、P=2 范围内，最优节点分配可以弥补固定 priority-list 的限制，不能外推为固定优先级普遍无损。
   - R-EFT 在 12/32 个实例上弱于固定优先级下的精确节点分配，平均 gap 7.0%、最大 40.0%，说明节点分配确实存在可优化空间。
   - 主动等待在 3/32 个实例上改善全局最优，说明它不是永远无关，但当前频率不足以证明必须放入第一版 Route B。
   - 当前学习 allocator 的 smoke、固定集过拟合以及图感知 v2 均显著弱于 R-EFT 和精确 F。图结构编码本身没有解决问题，下一步不应直接扩大训练规模，而应先诊断训练信号、目标与数据可识别性。

5. **工程基础较完整，但研究结论仍处于开发诊断阶段。**  
   当前完整测试为 **106 passed**；固定优先级 realizer、通信语义、Oracle 和训练路径均有测试。不过目前没有正式 Route A/B 选择、没有规模化泛化结果、没有成本/能耗/预算实现，也没有正式 Phase 2 checkpoint。

---

## 2. 目标论文：项目最初想复现什么

目标论文是 *A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling*，核心方法称为 **WeCAN（Weighted Cross-Attention Network）**。

### 2.1 论文问题

论文研究离线异构 DAG 调度。实例包含：

- DAG `G=(V,E)`；
- 任务基础时长 `t(v)` 和资源需求向量 `ρ(v)`；
- 异构资源池集合 `C` 及容量向量 `λ(c)`；
- 任务—资源池兼容/加速系数 `K_acc(v,c)`；
- 实际执行时长 `t_act(v,c)=t(v)/K_acc(v,c)`。

任务可以在同一资源池并发执行，只要任意时刻累计资源需求不超过容量。目标是在 DAG 依赖、容量和兼容性约束下最小化 makespan。

### 2.2 论文真正关键的思想：生成器也会造成最优性缺口

论文指出，传统 list scheduling 一旦存在可执行任务就立即派发，因此无法表达“现在有任务能运行，但为了未来更好的资源组合而主动等待”的调度。它定义了 generation-induced optimality gap：即使评分模型完美，如果生成映射根本到不了某类最优调度，学习也无法消除这部分损失。

论文的解决方式是给动作空间加入 `skip`：

- dispatch：在当前时间启动一个合法 task-pool 对；
- skip：即使存在合法 dispatch，也主动推进到下一个完成事件；
- skip 分数随决策步递减，防止无限或过度空闲。

因此，**Gap-Aware Generation 的核心不是损失函数中的 gap penalty，而是通过主动等待扩充调度生成器的可达空间。**

### 2.3 WeCAN 的两阶段、单次前向结构

1. 网络对整个实例只前向一次，输出：
   - 所有任务—资源池对的静态 logits；
   - 三个实例级 skip 参数。
2. 事件驱动 generator 反复使用静态 logits、动态可行性 mask 和递减 skip score 生成完整调度，不再调用网络。

模型主体包括：

- **WeCA**：任务与资源池之间的兼容性感知交叉注意力；兼容系数在 softmax 外部相乘；
- **LDDGNN**：用最长有向距离编码 DAG 依赖；
- **REINFORCE**：以 makespan 为代价训练，不是 PPO。

### 2.4 论文复现边界

项目中的 `docs/wecan_analysis.md` 已明确记录：

- 没有论文官方源码、原始数据或 checkpoint；
- 论文附录其实明确给出了主要架构与训练超参数，包括 3/2 维输入、`d_high=512`、`d_low=128`、8-head WeCA、8 层/16-head LDDGNN、八类 mask、`Dmax=500`、两层 64 宽 skip MLP、Adam `1e-4`、batch 64（TPC-H-100 为 32）和 800 batches；
- 仍未完全规定的主要是后续交替 WeCA 层数、训练时每实例 rollout 数、average-sample baseline 的精确聚合域/是否 leave-one-out、skip 输出的精确激活排列，以及若干优化器和数值实现细节；
- 当前随机 DAG 数据不能复现论文 TPC-H/RI-W 表格；
- 因此本工程验证的是方法语义和本地实验，不应宣称重现了论文数值；但也不应笼统表述为论文缺少主要网络参数。

此外，原论文的基础问题**不包含通信、成本、能耗和预算**；这些是本项目自定义问题带来的扩展。

---

## 3. 《问题建模.pdf》描述的自定义场景

《问题建模.pdf》共 2 页，给出一个带异构计算、通信和电力支出预算的连续时间调度模型。

### 3.1 实体与变量

- 任务 `u,v∈V`；计算节点 `i,j∈N`；
- 节点资源容量 `Cap_i`、计算速度 `f_i`、功率 `P_i`、单位电价 `Cost_i`；
- 节点间带宽 `bw_ij`；
- 任务工作量 `workload_v`、资源需求 `S_v`；
- DAG 边传输数据量 `dsize`；
- 二元 assignment `x_{v,i}`；
- 连续开始/完成时间 `st_v,ct_v`；
- 用于描述并发重叠和容量的 `δ`、`Z`、`h` 辅助变量。

### 3.2 主要约束

1. 每个任务恰好选择一个节点：
   
   \[
   \sum_i x_{v,i}=1.
   \]

2. 节点异构执行时长：

   \[
   ct_v=st_v+\sum_i x_{v,i}\frac{workload_v}{f_i}.
   \]

3. 同一节点允许并发，但重叠任务的资源需求之和不能超过 `Cap_i`。

4. 对依赖 `u→v`，若父子落在节点 `i,j`，后继需等待：

   \[
   st_v\ge ct_u+\frac{dsize}{bw_{ij}}.
   \]

5. 总支出预算：

   \[
   \sum_v\sum_i x_{v,i}\frac{workload_v}{f_i}Cost_iP_i\le Budget.
   \]

### 3.3 文档中的重要缺口

- **没有显式目标函数。** 结合项目后续设计，最可能的目标是“预算约束下最小 makespan”，但这不是 PDF 两页直接写出的事实。
- `δ_{u,v}` 的文字描述像无方向的任务重叠变量，而公式更接近“任务 `u` 在任务 `v` 启动时仍在运行”的有向事件变量。
- `Z` 的逻辑“当且仅当”线性化只写了下界，缺少通常需要的上界。
- `dsize` 没有明确写成边属性 `dsize_{u,v}`。
- 没有明确同节点通信是否为零、通信是否占用链路、是否存在链路竞争和固定 latency。
- 文中列出的“电力节点”没有进入后续约束。
- `S_v`/`Cap_i` 看起来是单维资源，而当前代码已推广到多维资源。

因此，《问题建模.pdf》目前更像问题草稿，而不是足够严谨、可直接编码的最终规格。

---

## 4. 三套问题语义的关系

| 维度 | 原 WeCAN 论文 | 《问题建模.pdf》 | 当前 Phase 2.1c |
|---|---|---|---|
| 目标 | 最小 makespan | 未显式写出；推测为预算下最小 makespan | 最小 makespan |
| 异构执行 | `base_duration / compatibility` | `workload / node_speed` | `base_duration / compatibility` |
| 资源 | 多维累积容量 | 公式表现为单维累积容量 | 多维累积容量 |
| 通信 | 无 | `dsize / bandwidth` | 同池 0；跨池 `latency + data_size/bandwidth` |
| 通信竞争 | 无 | 未明确；公式未建模 | 非竞争，仅形成 child release delay |
| 成本/能耗/预算 | 无 | 有合并后的电力支出预算 | **尚未实现** |
| 决策方式 | 在线 task-pool + skip | 联合 assignment 与连续 start time | 先离线完整 assignment，再固定优先级 realization |
| 主动等待 | 核心机制 | 连续时间模型未禁止 | 当前 primary Route B 禁止，仅保留全局 Oracle 对照 |
| 单次前向 | 是 | 不涉及学习架构 | 是 |

当前系统不是对《问题建模.pdf》的完整求解器，也不是对原 WeCAN 的继续照搬。它是在两者之间提炼出一个更窄、可精确测量的问题：

> **在通信感知、异构、多维累积容量条件下，固定任务优先级后，如何为每个任务选择资源池，使确定性 priority-list realizer 的 makespan 最小？**

---

## 5. 项目演进与关键转折

### 5.1 Phase 1：先完成 WeCAN 最小闭环

早期阶段实现了：

- `DAGInstance`、随机分层 DAG 生成器；
- Random、Greedy、capacity-aware HEFT；
- WeCA、LDD attention、静态 task-pool score、递减 skip；
- REINFORCE、训练/恢复、统一评估、单实例推理；
- 单次前向计数和基础可行性测试。

最初烟雾实验只有 24/8/12 个 train/validation/test 实例和 2 个 epoch：WeCAN greedy makespan 均值 123.608，弱于 Greedy 114.075 和 HEFT 113.741。这个阶段仅证明工程路径可运行，未证明性能复现。

### 5.2 Phase 1.5：从“能运行”转向“结论可被证明”

Git 历史从 `b4ca577`、`aa6b92d` 开始集中补强实验协议与诊断。关键变化是：

- 新增独立 `scheduler/validator.py`；
- 新增 PuLP/CBC 时间索引 MILP Oracle；
- 新增独立事件式 exhaustive Oracle；
- 对 10 个手工实例和 50 个随机小实例交叉验证，60/60 无失败；
- 修正 A1 指标混淆：原报告把 external Greedy 误当成 policy greedy；重新分析后确认 A1 的意义是策略收缩/训练正确性，而不是超过外部 Greedy。

这次转折很重要：项目不再把“训练曲线下降”直接当成算法有效，而是先建立可行性、全局最优和数据隔离证据。

### 5.3 Phase 1.6：冻结小型可证明诊断集

在 `08dcbc3` 完成 Phase 1.6 fixed-set diagnostic，随后 `839ebbd` 冻结验收元数据：

- 24 train、12 validation、30 test，共 66 个结构唯一实例；
- 训练 300 updates，K=8，单次前向复用；
- 最优 validation checkpoint 在 update 179；
- test 平均 normalized Oracle gap 从 0.5976 降到 0；
- 最优 checkpoint 在 30 个 test 上全部达到 Oracle；
- F2 无主动等待，F3 每个 test 最佳轨迹都使用主动等待；
- 全部 train/validation/test 轨迹可行，测试集只在 checkpoint 锁定后评估。

这是一个强工程诊断成功，但仍不是论文真实数据上的一般化结论：它证明当前 WeCAN-like 路径能在精心构造、可精确验证的小型分布上学到关键行为。

### 5.4 Phase 2.1a：引入非竞争通信

`ee40b0d` 把通信语义落实到实例、generator、Validator 和两类 Oracle：

- 每条 DAG 边有 `data_size`；
- pool 对有有向 bandwidth 与 latency；
- 同 pool 通信为 0；
- 跨 pool 延迟为 `latency + data_size/bandwidth`；
- 通信不占计算或网络资源，只改变 child release time；
- 通信实例使用严格整数 tick，避免浮点近似污染 exact proof。

P2-a 诊断中，duration-only assignment 的 makespan 为 12，而 Oracle 与手工静态 score generator 都为 5，说明节点选择必须考虑通信共址，而不能只看单任务执行速度。

### 5.5 Phase 2.1b：曾计划继续做通信版 WeCAN，但尚未训练

`7f29145` 设计了 `CommunicationWeCAN`：使用 DAG edge data、pool 网络方向、bandwidth/latency 和 task-pool pair relation；仍保持静态 logits + skip 参数的一次前向契约。

`50cdbfa` 完成 Gate 0：8 个固定 fixture 的 MILP、with/without-wait exhaustive、DO-G、CA-G 和 Validator 语义一致；其中 `CW` 证明 active wait 可将 makespan 从 11 降到 10。

但这个阶段没有继续实现完整 `CommunicationWeCAN` 训练。随后研究问题发生转折：先判断“固定优先级 + 学习节点分配”是否是更可控、更值得投入的路线。

### 5.6 Phase 2.1c：转向固定任务优先级 + 节点分配

`36b89d6` 至 `add939b` 设计并澄清 G/F/R 框架：

- **G**：全局 exact optimum，分别计算允许/禁止 active wait；
- **F**：固定 priority-list realization 下，对全部合法完整节点 assignment 求精确最优；
- **R**：局部节点分配启发式，包括 duration 和更公平的 prefix-EFT。

定义：

\[
PriorityGap_p=\frac{F_p-G_{no\_wait}}{G_{no\_wait}},
\qquad
GreedyGap_{EFT,p}=\frac{R_{EFT,p}-F_p}{F_p},
\]

\[
WaitGap=\frac{G_{no\_wait}-G_{wait}}{G_{wait}}.
\]

这三个量把原先混在一起的误差拆开：

1. 固定优先级本身造成的结构损失；
2. 节点分配策略相对该受限问题最优值的损失；
3. 禁止主动等待造成的损失。

`ea134bb` 实现精确固定优先级 assignment 枚举，`834e81e` 通过 8 个语义一致性 case，证明新 realizer 与冻结 generator 的 no-active-wait 行为一致。

### 5.7 当前最新阶段：学习 allocator 未通过开发诊断

- `90c7bbd`：精确求解规模校准；N=4、P=2 稳定，N=6 有一个 MILP 超时未证最优，N=8 被当前 canonicalizer 的 `MAX_CANONICAL_TASKS=7` 阻断。
- `7966a0f`：32 实例 G/F/R 开发筛选；固定优先级在该集合上没有观察到 gap，但 R-EFT 有明确节点分配 headroom。
- `51bc5c9`：固定优先级 allocator smoke 能训练并保持 100% 可行，但 model greedy 比 R-EFT 差 38.62%。
- `58d805b`：在同一 32 个实例上做三 seed 过拟合，模型 greedy 仍比 F 差约 28.5%–28.9%，比 R-EFT 差约 19.8%–20.1%。
- `45fda42`：加入显式 DAG 双向消息和有向 pool 网络消息的 graph-aware v2；三 seed 结果仍与 v1 基本相同，平均 model greedy 约 21.97–22.00，而 F=17.375、R-EFT=18.4688。

最新结论不是“固定优先级路线失败”，而是：

> **当前静态 allocator + 当前 REINFORCE 信号，即使加入更完整的图关系表示，也没有学会已由 exact F 证明存在的优质节点分配。**

---

## 6. 当前有效算法语义

### 6.1 固定 priority 的两个角色

当前 Route B 使用同一个静态 priority，但它承担两个不同角色：

1. `allocation_decision_order`：离线依次为任务选择 pool；此时没有 current time、running set 或 communication-ready 状态。
2. `schedule_realization_priority`：完整 assignment 固定后，在每个事件时刻从当前 eligible tasks 中选最高优先级任务。

两者不能混淆。当前没有预算前缀约束时，完整枚举 F 的 assignment 可行集不受遍历顺序影响；F 随 priority 变化，是因为 realization priority 改变了实际时间表。

### 6.2 当前支持的静态优先级

实现位于 `scheduler/priority_policies.py`：

- `topological-min-id-v1`：Kahn 拓扑顺序，按 task ID 确定性打破平局；
- `critical-path-min-duration-v1`：按最短兼容执行时长计算静态关键路径值；
- `heft-upward-rank-directed-static-v1`：按平均兼容执行时长和有向平均通信延迟计算 HEFT-style upward rank。

所有 priority 必须覆盖全部任务且拓扑合法，代码会拒绝非法顺序，不会静默修复。

### 6.3 离线节点分配

`scheduler/fixed_priority_allocation_decoder.py`：

- 网络一次输出 `[num_tasks,num_pools]` logits；
- 按 priority order 逐任务 argmax 或 softmax sample；
- hard mask 只检查兼容性和该任务能否单独放入 pool；
- 不在 allocation 阶段模拟跨任务并发容量；
- 全部任务完成 assignment 后再调用 realizer；
- sample 模式累加 log-prob，供 REINFORCE 使用。

### 6.4 固定 assignment 的事件式实现

`scheduler/complete_assignment_realizer.py`：

- assignment 不可更换；
- 当前任务必须满足：父任务完成、通信 release 已到、assigned pool 容量足够；
- 在 eligible set 中选静态优先级最高者；
- 高优先级任务若当前不 eligible，不阻塞低优先级合法任务；
- 同一时刻会重复 dispatch，允许多任务并发；
- 只有没有 eligible task 时才被动推进到最近 completion 或 communication release；
- **没有 voluntary active wait**；
- 最终 schedule 必须由独立 Validator 检查。

所以当前 Route B 不是“固定全序强阻塞”，也不是在线联合 task-pool 选择，而是：

> **固定完整 assignment + ready/communication/capacity-aware priority-list realization。**

---

## 7. 当前代码架构

| 目录/模块 | 当前职责 |
|---|---|
| `data/instance.py` | DAG、任务、pool、多维资源、兼容性、边通信、序列化和 schema 验证 |
| `data/phase2_communication.py` | 通信实例的置换、canonical identity/fingerprint |
| `data/phase21c_master_generator.py` | Phase 2.1c 可复现的开发实例生成 |
| `scheduler/generator.py` | WeCAN 风格在线 task-pool + skip 事件解码 |
| `scheduler/validator.py` | 权威独立调度/trace 验证器 |
| `scheduler/priority_policies.py` | 三种静态、拓扑合法 priority |
| `scheduler/fixed_priority_allocation_decoder.py` | 一次前向 logits 到完整离线节点 assignment |
| `scheduler/complete_assignment_realizer.py` | 固定 assignment 的 no-active-wait priority-list realization |
| `models/wecan.py` | Phase 1 WeCAN-like 模型 |
| `models/fixed_priority_allocator.py` | Phase 2.1c v1 静态 allocator |
| `models/fixed_priority_allocator_graph.py` | Phase 2.1c v2 图感知 allocator |
| `oracle/milp_oracle.py` | 时间索引 MILP/CBC 全局 Oracle |
| `oracle/exhaustive_oracle.py` | 小实例 with/without-active-wait 精确搜索 |
| `oracle/fixed_priority_assignment_oracle.py` | 固定 priority-list 下完整 assignment 枚举 F |
| `baselines/algorithms.py` | Random、Greedy、CA-HEFT、standard HEFT 等 |
| `baselines/priority_allocation_greedy.py` | Route B 的 duration 与 prefix-EFT allocation baseline |
| `training/reinforce.py` | Phase 1 通用 REINFORCE |
| `training/phase21c_*` | G/F/R、smoke、过拟合、图模型等开发诊断 |
| `scripts/` | 25 个数据、训练、评估、Oracle 和阶段诊断 CLI |
| `tests/` | 25 个测试文件，覆盖 Phase 1 至 Phase 2.1c |
| `results/` | 已跟踪的 JSON/报告/诊断结果；模型权重和 CBC 日志由 `.gitignore` 排除 |

技术栈：Python、PyTorch、NumPy、PyYAML、PuLP/CBC、pytest、matplotlib。

### 一个已知维护风险

`scheduler/types.py` 仍保留较旧的轻量 `validate_schedule()`，它不完整检查通信；当前主路径使用的是 `scheduler/validator.py`。后续应统一或显式弃用旧接口，避免调用者误用。

---

## 8. 当前实验事实

### 8.1 已被较强证据支持的事实

1. **基础调度与 exact 工具链可信。**  
   Phase 1.5 的 60 个小实例上，MILP 与 exhaustive 全部一致且通过独立 Validator。

2. **WeCAN-like 学习路径可以在构造小分布上学到最优与主动等待。**  
   Phase 1.6 的冻结 66 实例实验中，最佳 checkpoint 在 30 个 test 上达到 0 normalized Oracle gap；F3 的 active-wait 行为也被 exact counterfactual 支持。

3. **通信会改变最优节点分配。**  
   P2-a 中，仅看最快执行节点得到 makespan 12，而通信感知最优 assignment 为 5。

4. **当前小规模 Route B 中，节点分配比 priority 更值得先优化。**  
   32 实例筛选中三个 priority 的 `PriorityGap=0`，而 R-EFT 在 12 个实例上有正 gap，最大 40%。

5. **当前 allocator 学习方案不够。**  
   smoke、v1 fixed-set overfit、v2 graph-aware overfit 均未接近 R-EFT/F；图消息传递没有带来实质改善。

### 8.2 不能据此声称的内容

- 不能声称已经复现论文 TPC-H/RI-W 数值；
- 不能声称固定优先级对一般实例无损；
- 不能声称 Route B 已优于 Route A；
- 不能声称主动等待可以删除；
- 不能声称图感知 allocator 架构本身无效，只能说当前实现与训练信号的组合未通过固定集诊断；
- 不能声称已经解决成本、能耗和预算问题；
- 不能把 development screen、overfit diagnostic 或 diagnostic weights 当作正式训练/测试结论。

---

## 9. 为什么会从论文复现转向“固定优先级 + 选节点”

从仓库演进看，这不是一次随意改题，而是由诊断逐步推动：

1. 初始 WeCAN 复现证明完整在线 task-pool+skip 机制可以实现，但短训练并不优于强启发式。
2. Oracle 与 Validator 建成后，项目开始能够把“实现错误、生成器限制、启发式差距、学习失败”分开测量。
3. Phase 1.6 证明复杂的 skip 行为在设计分布上可学，但该成功依赖小型、精确构造的任务族，并不能直接说明加入通信后仍易学。
4. Phase 2.1a 证明通信使节点 assignment 本身成为显著决策：局部最快节点可能因跨池传输而非常差。
5. 因此 Phase 2.1c 先固定任务 priority 和 deterministic realization，把研究难点压缩到“完整节点 assignment”，再用 G/F/R 判断这个简化是否值得。
6. 小型 screen 暂时支持该分解：没有观察到 priority loss，却观察到 allocation headroom。但随后的学习诊断表明，**把动作空间简化为节点选择不等于问题已经容易学习**。

该转折的科学意义是把一个联合在线调度问题拆成两个可分别证伪的问题：

- 固定 priority 是否把最优解排除在可达空间之外？
- 若没有，学习器能否找到优质 assignment？

---

## 10. 当前阻塞点与技术债

### 10.1 研究阻塞点

1. **训练信号问题尚未定位。**  
   v2 表示更强但结果没有改善，可能涉及 REINFORCE 方差、逐任务分解的 credit assignment、同实例 rollout baseline、分布可辨识性或 greedy/sample mismatch。

2. **固定优先级证据规模太小。**  
   当前 exact G/F/R 主要限于 N=4/6、P=2；`PriorityGap=0` 不能外推。

3. **exact scaling 被 structural canonicalizer 阻塞。**  
   当前通信 canonicalizer 上限为 7 个任务，N=8 校准未进入求解阶段。

4. **成本、能耗、预算尚未进入实例和算法。**  
   这使当前实现仍未覆盖《问题建模.pdf》的完整场景。

5. **Route A/B 尚未正式选择。**  
   现有材料只支持继续诊断 Route B，不构成最终路线结论。

### 10.2 工程与文档债

- 根 `README.md` 仍称“当前仅完成 Phase 1”，已落后于 Phase 2.1c 代码和结果；
- `docs/phase2_design.md`、`docs/phase21b_model_training_design.md` 的开头仍保留当时的“设计待审批”历史状态，阅读时应结合后续 commit/report；
- 默认 `data/generated/` 和正式 checkpoint 未提交，README 的 Phase 1 train/evaluate 命令需先生成数据；
- 图感知 v2 直接读取完整 bandwidth/latency 矩阵，只适用于通信字段完备的实例，不能无修改用于零通信 Phase 1 数据；
- 两套 `validate_schedule` 接口存在语义漂移风险；
- 部分 result 中的 diagnostic weights 不属于正式 checkpoint，不应被误用于后续结论。

---

## 11. 建议的下一步顺序

### 优先级 1：先诊断 allocator 的学习目标，不扩大规模

在当前 32 个 certified-F 实例上进行最小、可归因的诊断：

1. 用 exact F assignment 做监督分类/排序上界，判断模型表示是否能拟合目标；
2. 区分“表示不可分”与“REINFORCE 学不到”：
   - 若监督也拟合不了，修改 pair features/architecture；
   - 若监督能拟合而 REINFORCE 不行，重点改 credit assignment、baseline 或目标；
3. 记录每步 chosen-pool probability、assignment regret、同一任务不同 pool 的 logit margin，而不只看最终 makespan；
4. 保持固定 seeds、固定数据和同一 realizer，避免同时改变模型、数据和优化器。

这一步完成前，不建议直接扩大数据、更新数或切换 PPO。

### 优先级 2：扩大 G/F/R 的可证明覆盖

- 先解决 N≥8 的结构 identity/canonicalization；
- 再校准 N=8/10/12 和 P=3 的 exact 可计算边界；
- 构造更容易暴露 priority-list 损失的 DAG/资源冲突模式；
- 分层报告通信强度、资源竞争、异构程度和拓扑族；
- 保留 active-wait global reference，避免因当前 Route B 不等待而忽略论文的核心风险。

### 优先级 3：再决定 Route B 是否升级为正式模型路线

只有当：

- `PriorityGap` 在目标分布中足够小；
- `GreedyGap_EFT` 稳定为正且有实际量级；
- allocator 至少能在 fixed-set 上达到或稳定超过 R-EFT；

才值得进入正式训练/泛化阶段。否则应回到 Route A，或使用“固定 task-pool logits + 在线 mask + skip”的中间方案。

### 优先级 4：补齐自定义场景的预算语义

最终若目标是《问题建模.pdf》，还需：

- 明确并补写正式目标函数；
- 决定成本与能耗是合并预算还是两个独立硬预算；
- 将预算字段加入 instance schema、fingerprint、Validator 和 Oracle；
- 在离线 allocation 中使用“当前已承诺预算 + 剩余任务可完成性下界”的 mask；
- 重新审计固定 allocation order，因为前缀预算 mask 会使决策顺序真正影响可行轨迹。

---

## 12. 可运行入口

### Phase 1 基础流程

```bash
python scripts/generate_data.py \
  --config configs/phase1.yaml \
  --output-dir data/generated \
  --train-count 128 --validation-count 32 --test-count 64

python scripts/train.py \
  --config configs/phase1.yaml \
  --train-data data/generated/train.json \
  --validation-data data/generated/validation.json \
  --checkpoint-dir checkpoints/wecan_phase1 \
  --device cpu

python scripts/evaluate.py \
  --config configs/phase1.yaml \
  --test-data data/generated/test.json \
  --checkpoint checkpoints/wecan_phase1/best.pt \
  --output results/phase1_evaluation.json
```

### 当前 Phase 2.1c 诊断入口

```bash
python scripts/run_phase21c_realizer_conformance.py --help
python scripts/run_phase21c_solver_calibration.py --help
python scripts/run_phase21c_gfr_screening.py --help
python scripts/run_phase21c_fixed_priority_smoke.py --help
python scripts/run_phase21c_fixed_priority_overfit.py --help
python scripts/run_phase21c_fixed_priority_overfit_v2_graph.py --help
```

注意：部分脚本会拒绝覆盖已有结果目录；运行前应查看参数并使用新的输出位置，而不是删除冻结工件。

### 测试

2026-08-25 在当前 HEAD 实际执行：

```bash
python -m pytest -q
```

结果：

```text
106 passed in 105.97s
```

CBC 也已检测可用。当前工程的主要阻塞不是导入或基础测试失败，而是研究路线和学习效果尚未通过下一阶段门槛。

---

## 13. 推荐阅读顺序

若要快速接手项目，建议按以下顺序阅读：

1. 本文档；
2. `docs/wecan_analysis.md`：原论文方法与复现边界；
3. `docs/phase1_acceptance_report.md`：Phase 1 冻结结论；
4. `docs/phase2_design.md`：自定义通信/预算语义的整体设计；
5. `docs/phase21b_preflight_report.md`：通信 exact 语义与固定 fixtures；
6. `docs/phase21c_gfr_experiment_design.md`：当前 G/F/R 与 Route B 的精确定义；
7. `docs/phase21c_gfr_screening_report.md`：当前路线筛选证据；
8. `docs/phase21c_fixed_priority_smoke_report.md`；
9. `docs/phase21c_fixed_priority_overfit_report.md`；
10. `docs/phase21c_fixed_priority_overfit_v2_graph_report.md`：最新学习失败诊断；
11. `scheduler/complete_assignment_realizer.py`、`scheduler/fixed_priority_allocation_decoder.py`、`models/fixed_priority_allocator_graph.py`：当前主线实现。

---

## 14. 最终状态判断

截至 `45fda42`，项目已经具备：

- 可运行的 WeCAN-like 单次前向基线；
- 独立 Validator 和双 exact Oracle；
- 严格、可测试的非竞争通信语义；
- 固定优先级 + 完整节点 assignment 的确定性实现；
- G/F/R 结构性 gap 分解；
- v1/v2 allocator 的训练与失败诊断；
- 较完整的数据隔离、证据分级和实验边界意识。

但当前项目仍是**方法选择与机制诊断阶段**，不是完成的论文复现，也不是完整的自定义场景求解器。最准确的状态表述是：

> 已证明“固定优先级下的节点分配”在小型开发实例上存在可优化空间，并已搭建完整的精确评估和学习流水线；但当前 REINFORCE allocator 尚不能利用这部分空间，图感知表示也未解决该问题。成本、能耗与预算仍待实现，固定优先级路线是否能推广到更大、更复杂实例仍待证明。
