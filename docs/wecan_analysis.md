# WeCAN 复现分析：单次前向异构 DAG 调度基线

> **范围声明。** 本文档基于项目内论文 *A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling*（arXiv:2603.23249v2）整理。当前项目未包含 WeCAN 官方源码、数据、训练脚本或检查点，论文也未提供 WeCAN 代码链接。因此后续实现只能称为 **WeCAN 忠实重实现（faithful reimplementation）**，不能称为“完全复现”或“官方复现”。

## 1. 问题定义

### 1.1 输入

论文 §II-A（p.3，式(1)）定义一个实例为

\[
\mathcal P=(V,E,\mathcal C,t,\rho,\lambda,K_{\mathrm{acc}}).
\]

- \(G=(V,E)\)：任务 DAG；边 \((v,w)\) 表示任务 \(v\) 必须完成后，任务 \(w\) 才能开始。
- 每个任务 \(v\)：基础处理时间 \(t(v)>0\)，资源需求向量 \(\rho(v)\)。
- 每个资源池（pool）\(c\in\mathcal C\)：容量向量 \(\lambda(c)\)。
- \(K_{\mathrm{acc}}(v,c)\ge0\)：任务—资源池兼容性系数。\(K_{\mathrm{acc}}(v,c)=0\) 表示不兼容；否则实际处理时间是
  \[
  t_{\mathrm{act}}(v,c)=t(v)/K_{\mathrm{acc}}(v,c).
  \]

论文第一阶段的资源池允许多个任务并行，只要时刻资源需求总量不超过容量。

### 1.2 输出

调度 \(x\) 为每个任务输出开始时间与资源池：

\[
x: v\mapsto(s(v),c(v)),\qquad s(v)\ge0.
\]

神经网络**不直接输出**这些开始时间或完整调度；它输出静态分数，随后由调度生成器构造 \(s(v),c(v)\)。

### 1.3 目标与基础约束

目标为最小化完工时间（makespan）：

\[
\min_x f(x)=\min_x\max_{v\in V}\left(s(v)+t_{\mathrm{act}}(v,c(v))\right).
\]

约束为：

1. **依赖约束**：对每条 \((v,w)\in E\)，
   \[
   s(v)+t_{\mathrm{act}}(v,c(v))\le s(w).
   \]
2. **累积资源容量约束**：任意时刻 \(\tau\)、资源池 \(c\)，运行中任务的需求和不超过 \(\lambda(c)\)。
3. **兼容性约束**：\(K_{\mathrm{acc}}(v,c(v))>0\)。
4. **非负开始时间约束**：\(s(v)\ge0\)。

论文式(1)中运行集合的上界漏写了 \(s(v)+\)；严格实现采用正确的区间
\([s(v),s(v)+t_{\mathrm{act}}(v,c(v)))\)，将此视为论文记号笔误。

## 2. 一次网络前向具体输出什么

论文 §IV-C（pp.10–11）中的 Stage I 对**整个实例仅前向一次**，输出：

1. 每个兼容任务—资源池对的静态 action score
   \[
   u_{v,c}=(W_s^Qh_v^{(L)})^\top(W_s^Kz_c^{(L)})+\log K_{\mathrm{acc}}(v,c).
   \]
2. 一个实例级 skip 参数向量
   \[
   \psi=(\alpha,\beta,\gamma)
   =\operatorname{MLP}\!\left(\frac1{|V|}\sum_{v\in V}h_v^{(L)}\right).
   \]
3. 中间的最终任务嵌入 \(h_v^{(L)}\) 和资源池嵌入 \(z_c^{(L)}\)（用于上述解码器）。

它不直接输出任务开始时间、最终资源分配、逐步状态、逐步动作分数、value/critic 或完整调度。Stage II 在不再调用网络的前提下，只使用上述静态 score、skip 参数和动态可行性 mask 来完成调度。

对 \(K_{\mathrm{acc}}=0\) 的对，\(\log K_{\mathrm{acc}}\) 无定义。论文的动作集合和文字均表明这些对会被 mask；忠实重实现会在数值上为其保留有限占位值并在 mask 中强制排除。

## 3. 加权交叉注意力（WeCA）

设任务嵌入矩阵 \(H\in\mathbb R^{|V|\times d}\)，资源池嵌入矩阵 \(Z\in\mathbb R^{|\mathcal C|\times d}\)。论文 §IV-C（p.10）对资源池 \(\rightarrow\) 任务方向定义：

\[
Q=HW^Q,\quad K=ZW^K,\quad V=ZW^V,
\]
\[
\operatorname{WeCA}(H,Z,K_{\mathrm{acc}})=
\left[\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)\odot K_{\mathrm{acc}}\right]V.
\]

逐任务写为：

\[
A_{v,c}=\operatorname{softmax}_{c'}(q_v^\top k_{c'}/\sqrt d)_c,
\quad
m_v=\sum_c A_{v,c}K_{\mathrm{acc}}(v,c)V_c.
\]

关键是兼容性矩阵位于 softmax **外部**，因此 gated attention 行和一般不为 1。论文有意不用
\(\operatorname{softmax}(QK^\top/\sqrt d+\log K_{\mathrm{acc}})V\)：外部相乘保留“一个任务可兼容多少/多好资源池”的整体质量信息。反向任务 \(\rightarrow\) 资源池更新对 \(K_{\mathrm{acc}}^\top\) 对称使用。实际架构采用多头注意力、残差连接；先做 WeCA 与 LDDGNN，再交替更新任务/资源池嵌入。

论文还使用 LDDGNN（longest-directed-distance graph neural network）编码 DAG 先后关系：对每对节点求有向最长路径距离并将其作为 attention bias/mask。该模块是架构的一部分；实现中需与 WeCA 一同保留。

## 4. 跳过动作与递减跳过分数

论文 §IV-B（pp.8–10，式(7)、算法2）将动作空间扩为

\[
\bar{\mathcal A}=\mathcal A\cup\{a_{\mathrm{skip}}\}.
\]

- **dispatch 动作** \((v,c)\)：在当前时刻把任务 \(v\) 启动到资源池 \(c\)。
- **skip 动作**：即使当前存在可启动任务，也将当前时间推进到下一个任务完成事件，并释放对应资源。
- 若无运行任务，skip 被 mask，因为不存在下一个完成事件。

在第 \(k\) 个决策（\(n=|V|\)）时，skip 的概率质量和 log-score 为

\[
q_{\mathrm{skip}}(k)=\alpha\exp\left(-\frac{\gamma k}{2n}\right)+\beta,
\qquad
u_{\mathrm{skip}}(k)=\log q_{\mathrm{skip}}(k),
\]

其中 \(\alpha,\beta,\gamma>0\)。它严格递减，避免一个固定/非递减 skip 分数不断压制 dispatch；同时不需逐步重新运行神经网络。论文证明调度最多有 \(n\) 次 dispatch 和 \(n\) 次 skip，即最多 \(2n\) 次决策。

## 5. 从评分生成可行调度

算法2的生成过程如下：

1. 初始化 \(t_{\mathrm{cur}}=0\)、未调度集合、各资源池可用容量和运行任务集合。
2. 在当前时间根据**已完成的前驱**、当前可用容量、兼容性建立动态 action mask。
3. 用一次前向得到的静态 \(u_{v,c}\)，及由 \(\psi\) 在当前步解析计算的 \(u_{\mathrm{skip}}(k)\)，形成 masked categorical：
   \[
   p_\theta(a_t=a)=\frac{\exp(u_a+M_a)}
   {\sum_{\tilde a\in\bar{\mathcal A}}\exp(u_{\tilde a}+M_{\tilde a})}.
   \]
4. 训练/采样模式从该分布采样；贪心模式选最大 score。
5. 选择 \((v,c)\) 时在 \(t_{\mathrm{cur}}\) 启动任务、扣除容量并移出未调度集合；选择 skip 时推进到下一完成事件并释放所有完成任务。
6. 重复至所有任务被启动。调度可行性由每一步 mask 保证，而不是由网络直接保证。

论文 Theorem 1 声称：无论 score 为何，该算法会终止并输出可行调度；在采样模式下，串行 SGS 能生成的每个调度有严格正的采样概率。因此该结论是**可达性/存在性**，并不保证一个已训练模型必然找到全局最优解。

## 6. 原论文强化学习方法（不是 PPO）

论文原始方法是 **REINFORCE / score-function policy gradient**，不是 PPO。PPO-BiHyb 仅是论文实验的对照基线。

论文 §IV-D（p.11）定义：令 sampled trajectory 为 \(\omega\)，生成器产生 \(S_d(\omega)\)，且
\(\hat f(\omega)=f(S_d(\omega))\)。优化目标是

\[
\min_\theta L(\theta)=
\mathbb E_{\mathcal P}\mathbb E_{\omega\sim p_\theta(\cdot\mid\mathcal P)}[\hat f(\omega)],
\]

梯度估计式为

\[
\nabla L(\theta)=
\mathbb E[(\hat f(\omega)-b(\mathcal P))\nabla_\theta\log p_\theta(\omega\mid\mathcal P)],
\]
\[
\log p_\theta(\omega\mid\mathcal P)=\sum_t\log p_\theta(a_t\mid s_t,\mathcal P).
\]

采样是在每一个生成步骤、动态 mask 后的 categorical 分布中完成。网络只在 rollout 开始前计算一次 score 和 \(\psi\)。

附录 F.3（p.30）给出：

- Adam，学习率 \(10^{-4}\)；
- 默认 batch size 64（TPC-H-100 用 32）；
- 训练 800 batches；
- 比较两种 REINFORCE baseline：一是“所有样本的平均 reward”，二是周期同步的独立模型贪心 rollout reward；
- 作者观察平均 reward baseline 更快、更平滑，因此将其作为默认；
- 训练和批量采样推理使用 PyTorch/GPU 版生成器，单实例 greedy 推理使用 CPU 版。

## 7. 无法完全复现、必须显式假设的细节

以下信息没有公开源码且论文未充分指定。后续实现会写入配置和 README，不会伪装为论文事实：

1. **REINFORCE 数值 loss 的符号与 reduction。** 论文一处称 reward 为负 makespan，另一处最小化 makespan；实现需采用与给出的梯度等价的 loss，并记录该选择。
2. **“average reward of all samples”定义。** 未说明是 batch 内平均、每实例多样本平均、是否 leave-one-out、是否包含当前样本或标准化。
3. **每个训练实例的 rollout 数量。** 未明确。
4. **熵正则、critic、advantage normalization、梯度裁剪、权重衰减、Adam betas、学习率调度、checkpoint 选择。** 均未说明；不能擅自称论文使用过它们。
5. **skip 参数正性变换。** 论文要求 \(\alpha,\beta,\gamma>0\)，但只说 MLP 使用 GELU 与 sigmoid，未规定 sigmoid/softplus/epsilon/缩放方式。
6. **交替 WeCA 层的确切数量。** 论文给出了高层结构和总超参数，但未完全消除层数歧义。
7. **特征归一化、通用资源维度处理。** 论文实验以两维资源为例（任务输入 3 维、资源池输入 2 维），未给出泛化预处理协议。
8. **argmax 平局处理、mask 的有限数值实现、\(\log0\) 的实现。** 均未规定。
9. **论文数据与官方训练集不可得。** 本项目将用固定种子随机 DAG 生成器，不会把其结果和论文 TPC-H 数字直接比较。

## 8. 本工程第一阶段的“忠实重实现”边界

第一阶段保留论文核心：兼容性定义、累积容量、WeCA、LDDGNN、静态任务—池分数、一次前向、递减 skip、mask 生成器和 REINFORCE。由于缺少上述细节，本工程会：

- 在配置中显式固化所有假设；
- 通过前向调用计数、依赖/容量/兼容性校验来测试语义；
- 在相同随机测试集、相同资源配置上比较 Random、Greedy 和 HEFT；
- 报告均值、标准差、可行率、前向时间、完整生成时间和胜率；
- **不**在未训练、未测试前声称复现成功、超过 Greedy/HEFT 或达到论文结果。

第二阶段才引入用户混合整数规划模型中的跨节点通信、成本、能耗与预算约束；它们不属于本阶段 WeCAN 基线的原始问题。
