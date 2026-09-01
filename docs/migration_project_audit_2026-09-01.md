# DAG 调度项目迁移审计与当前进度

> 审计日期：2026-09-01  
> 当前目录：`/Users/huangjiayi/Documents/dag_0729`  
> 审计范围：当前根目录源码、配置、测试、结果、隐藏迁移残留，以及历史对话“CH-DAGSP调度方案建议”中的最近进度。  
> 重要说明：本文件描述的是“迁移后实际落盘内容”，不把旧文档中提到但当前根目录缺失的文件误当成现有实现。

## 1. 一页结论

当前项目不是空壳，WeCAN 的核心模型、事件式 Generator、REINFORCE、Validator、MILP/穷举 Oracle、论文风格 500-task synthetic generator、第一轮 Generator 性能修复和性能诊断脚本均已保留。

当前最准确的主线状态是：

```text
WeCAN faithful reimplementation
        ↓
论文参数配置和小规模端到端路径跑通
        ↓
论文 computation-graph synthetic generator 已实现（默认 500 tasks）
        ↓
500-task Generator 首次实测约 317 s/rollout
        ↓
加入 topology cache，约 317 s → 2.9~3.5 s
        ↓
old/new decode 与梯度等价验证完成
        ↓
真实训练吞吐预检：batch=1、K=8 约 35.9 s/update
        ↓
定位第二热点 `_dispatch_mask`
        ↓
【当前代码停在这里】第二轮 short-circuit 优化尚未实施
        ↓
论文规模 batch=64、800 updates 正式训练尚未开始
```

迁移状态有四个必须先记住的事实：

1. **原始 `.git` 历史没有恢复；当前已新建一个空白本地 Git 仓库用于重新建立管理基线。** 因此旧对话中的 `87b2bb6`、`6aafbc5`、`bd9da0d` 等历史提交仍无法直接检出、比较或回退。
2. **论文规模正式 checkpoint 仍不存在。** 当前新增了 paper-profile GPU short-run checkpoint，但它只有 1 update、实例平均约 17.9 tasks；没有 500-task/800-update 正式训练权重。
3. **Phase 1.6 冻结数据、训练结果和 checkpoint 已在本次审计后补回。** manifest 列出的 13 项数据、权重、history、evaluation 和配置现已全部位于原始路径，且 SHA-256 13/13 完全匹配。
4. **迁移内容发生了阶段错位。** `docs/project_overview.md` 描述的是更晚的 Phase 2.1c 固定优先级节点分配路线，但对应源码、配置和结果大量缺失于当前根目录。好消息是，这些源码多数仍存在于 `.claude/worktrees/` 的只读式快照中；坏消息是这些 worktree 的 `.git` 指针仍指向迁移前的 Linux 路径，已经失效，不能当作正常 Git worktree 使用。

因此，当前根目录应被看作一个“**可读、语法完整，但依赖环境和版本历史尚未恢复的源码快照**”，而不是已经可以直接继续正式训练的完整仓库。

## 2. 当前目录概况

补回 Phase 1.6 和 checkpoint 后，排除 `.claude/worktrees` 和 `__pycache__`，当前项目约有 232 个文件。主要目录体积如下：

| 路径 | 约占用 | 作用 |
|---|---:|---|
| `.claude/` | 872 MB | 迁移前 Claude agent worktree 快照和设置；不是当前运行源码，但有恢复价值 |
| `checkpoints/` | 475 MB | paper-profile 1-update GPU short-run 与早期 smoke 的 best/last checkpoint |
| `results/` | 25 MB | Phase 1.6 冻结数据/权重、历史 smoke、diagnostic、Oracle、图表和小模型 checkpoint |
| `data/` | 392 KB | 实例定义、生成器、固定 fixture、现有小规模 JSON 数据 |
| `tests/` | 312 KB | 当前根目录的 41 个测试函数及缓存 |
| `scripts/` | 260 KB | 数据、训练、评估、诊断、profile、差分验证入口 |
| `training/` | 196 KB | REINFORCE 与 A1 系列诊断逻辑 |
| `oracle/` | 112 KB | MILP 与穷举精确求解器 |
| `docs/` | 160 KB | 阶段报告、论文配置说明和历史总览 |
| `scheduler/` | 92 KB | Generator、调度类型、动作上界、独立 Validator |
| `models/` | 44 KB | WeCAN 模型 |

当前根目录仍没有：

- 原仓库的 Git objects、refs、remote 配置和 `pyproject.toml`；
- 根目录中的论文 PDF 和《问题建模.pdf》；
- 500-task、800-update 正式训练 checkpoint；
- Phase 2.1c 固定优先级路线的大部分根目录源码和结果；
- 可直接运行项目的本地 Python 虚拟环境。

其中 `README.md`、`pyproject.toml`、两份 PDF 及 Phase 2.1c 的大量源码仍可在多个 `.claude/worktrees/agent-*` 快照内找到，尚未彻底丢失。用户已另外提供一套 `.gitignore` 和 `.gitattributes` 文本；worktree 快照中还保存着另一套、内容不同的版本。因此恢复新仓库时不能直接把任一套称作唯一“原始规则”，需要按目标历史节点选择并统一。

## 3. 当前可运行代码的主数据流

```text
DAGInstance / JSON dataset
        │
        ├─ task duration、resource demand、pool capacity、compatibility
        ├─ DAG edges
        └─ 可选 communication metadata
                ↓
              WeCAN
        ├─ 一次神经网络 forward
        ├─ 输出 N×P task-pool 静态 scores
        └─ 输出 α、β、γ 三个 skip 参数
                ↓
      SkipExtendedGenerator.decode
        ├─ 动态 dispatch feasibility mask
        ├─ dispatch / active wait / passive advance
        ├─ 事件推进、容量释放
        └─ schedule + log_probability + trace metrics
                ↓
        Validator / evaluation
                ↓
        REINFORCE（K 条 rollout 复用一次 forward）
```

论文基础基线当前仍是：无通信、无成本/能耗/预算、无 PPO、无 fixed-priority。`DAGInstance` 和 Validator/Oracle 已经支持项目自定义的非竞争通信时延语义，但论文 500-task synthetic generator 默认不写通信元数据。

## 4. 各目录与文件作用

### 4.1 根目录与隐藏目录

| 文件/目录 | 作用 | 当前判断 |
|---|---|---|
| `requirements.txt` | 声明 PyTorch、NumPy、PyYAML、PuLP、pytest、matplotlib 依赖 | 文件完整；本机系统 Python 尚未安装这些依赖 |
| `.gitignore` | 规定不进入 Git 的缓存、运行产物、数据和权重 | 已重建统一版本；排除 `.claude`、cache、生成数据和 checkpoint 扩展名 |
| `.gitattributes` | 规定文本换行、二进制文件和 Markdown whitespace | 已重建；当前不使用 Git LFS，JSONL 以普通文本管理 |
| `README.md` | 项目入口、当前停点、目录、环境和仓库存储策略 | 已新建并链接本审计与核心阶段文档 |
| `.DS_Store` | macOS Finder 元数据 | 与研究代码无关 |
| `.claude/settings*.json` | 旧开发环境设置 | 不参与模型运行；不要在未审查前复制到新环境 |
| `.claude/worktrees/agent-*` | 21 组迁移前 agent worktree 文件快照 | Git 指针失效，但包含 Phase 2.1c 源码、README、PDF 等可恢复内容；暂勿删除 |
| `__pycache__/`、`*.pyc` | Python 字节码缓存 | 来自旧 Python 3.12 环境，不应当作源码或测试证据，可在环境恢复后重建 |

### 4.2 `configs/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `phase1.yaml` | 12–24 task smoke 模型/数据/训练默认配置 | 可作为小规模开发配置 |
| `phase1_smoke.yaml` | 更小的快速 smoke 配置 | 用于最短路径检查 |
| `phase1_paper.yaml` | 论文参数 profile：512/128、8-head WeCA、8 层/16-head LDD、batch 64、800 updates；同时显式记录本地假设 | 当前正式 baseline 的核心配置；尚未完成 800-update 训练 |
| `phase15_diagnostics.yaml` | A1/A1-b/A1-c 诊断 gate 与 300-update 小规模训练配置 | 历史诊断使用 |
| `phase16_fixed_set.yaml` | Phase 1.6 冻结数据集与 family-balanced 训练协议 | 已从一致 worktree 快照恢复，SHA-256 与验收 manifest 匹配 |
| `phase16_t2_supplement_v2.yaml` | Phase 1.6 T2 补充数据构造协议 | 已恢复，SHA-256 与验收 manifest 匹配 |
| `a1b_exact.yaml` | A1-b exact 小实例筛选协议 | 已产生筛选与训练结果 |
| `a1b_scale_proposed.yaml` | 更大 A1-b 分布建议，只允许生成/难度测试 | 提案，未批准正式训练 |
| `a1b_structured_hard_proposed.yaml` | 被 exact/scale 拆分方案取代的旧提案 | 仅留作追踪 |

### 4.3 `data/`

| 文件/目录 | 作用 | 状态 |
|---|---|---|
| `instance.py` | `DAGInstance`、通信边数据、合法性检查、序列化、随机 DAG 生成器、数据集读写 | 核心领域模型；支持可选非竞争通信时延 |
| `paper_computation_graph.py` | 按论文 Appendix C.2 实现 layered、Erdős–Rényi、stochastic block 三类 synthetic graph；默认 10×50=500 tasks | 已实现；论文未给出的细节已在注释中标为 assumption |
| `generated/train.json` | 24 个小规模训练实例，12–24 tasks，均值约 17.33 | smoke 数据，不是论文 500-task 正式数据 |
| `generated/validation.json` | 8 个小规模验证实例，15–24 tasks，均值约 19.25 | smoke 数据 |
| `generated/test.json` | 12 个小规模测试实例，13–23 tasks，均值约 16.92 | smoke 数据 |
| `fixtures/phase2/p2a_communication_tradeoff.json` | 通信权衡手工样例 | 用于通信语义验证 |
| `fixtures/phase21b/*.json` | chain、fan-in/out、通信 overlap、zero-data、active-wait 等 Phase 2.1b fixture | 8 个固定 fixture 与 manifest 均在 |
| `__init__.py` | Python package 标记 | 空文件，正常 |

### 4.4 `models/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `wecan.py` | WeCAN faithful reimplementation：初始 embedding、Weighted Cross-Attention、LDDGNN、task-pool decoder、skip head、paper/smoke/custom profile | 当前唯一根目录模型；paper profile 代码已在 |
| `__init__.py` | Python package 标记 | 空文件 |

当前根目录不存在旧总览提到的 `fixed_priority_allocator.py` 和 `fixed_priority_allocator_graph.py`；它们仅在 `.claude/worktrees` 快照中。

### 4.5 `scheduler/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `generator.py` | WeCAN Algorithm-2 风格 skip-extended event decoder；生成完整 schedule、log-probability 和 skip/advance 统计 | 已包含第一轮 `_TopologyCache` 优化；第二轮 dispatch-mask short-circuit 尚未实现 |
| `validator.py` | 独立检查 placement 完整性、duration、precedence、通信 release、容量与 trace 行为 | 正确性基础设施核心 |
| `types.py` | `TaskPlacement`、`Schedule` 等基础类型，以及一个较轻量的兼容校验入口 | 与独立 Validator 配合使用 |
| `action_bounds.py` | 计算 event-driven decode 的安全动作上界，防止异常死循环 | 已由 Generator 调用 |
| `__init__.py` | Python package 标记 | 空文件 |

当前 `_dispatch_mask` 的实际顺序仍是：先计算 `parents_done`，再为全部 pool 计算 capacity 和 `_ready_time`，最后才判断 task 是否仍 `unscheduled` 以及 compatibility。因此旧对话中计划的第二轮优化**尚未落到代码**。

### 4.6 `training/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `reinforce.py` | REINFORCE trainer、五类 baseline、K trajectory、checkpoint、训练统计 | 当前逐实例、逐 rollout 的 Python 串行实现，是大 batch 吞吐瓶颈之一 |
| `diagnostics.py` | A1、A1-c fixed-instance gate、指标、轨迹汇总与 counterfactual 检查 | 已产生历史诊断结果 |
| `diagnostic_instances.py` | A1-b 候选与 A1-c fixture 构造/筛选 | 筛选工具 |
| `a1b_exact.py` | A1-b exact integer-tick 实例生成、Oracle 审计和筛选 | A1-b-exact gate 已通过 |
| `a1b_audit.py` | 对已保存的 A1-b 候选分布做统计审计 | 不重新生成或训练 |
| `__init__.py` | Python package 标记 | 空文件 |

### 4.7 `baselines/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `algorithms.py` | Random、duration-only greedy、communication-aware greedy、Greedy、CA-HEFT、standard HEFT 等基线 | 可用于评估和 Oracle sanity check |
| `__init__.py` | Python package 标记 | 空文件 |

### 4.8 `oracle/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `common.py` | 将浮点实例严格缩放成 integer ticks，定义 Oracle 结果和公共工具 | 避免用近似取整冒充 exact proof |
| `milp_oracle.py` | 基于 PuLP/CBC 的时间索引 MILP Oracle | 适合可控小实例；保存 solver audit 信息 |
| `exhaustive_oracle.py` | 事件式完整搜索，区分允许/禁止 active wait | 与 MILP 交叉验证 |
| `fixtures.py` | 手工 tiny instances | 支撑边界与 Oracle 测试 |
| `__init__.py` | Python package 标记 | 空文件 |

本地保存的 `results/oracle/crosscheck_2026.jsonl` 有 60 条记录，MILP 与 exhaustive 的完成状态和 makespan 差异检查均无失败。

### 4.9 `evaluation/` 与 `environment/`

| 文件 | 作用 |
|---|---|
| `evaluation/metrics.py` | 统一运行 baseline/WeCAN，汇总 makespan、feasibility、skip 等指标 |
| `environment/config.py` | 读取 YAML 配置 |
| 两个目录的 `__init__.py` | Python package 标记 |

### 4.10 `scripts/`

#### 常规数据、训练与评估

| 文件 | 作用 |
|---|---|
| `generate_data.py` | 根据配置生成确定性 train/validation/test JSON |
| `train.py` | 从 YAML 和数据集启动 WeCAN REINFORCE；支持 `--resume` |
| `evaluate.py` | 在固定测试集上统一评估基线和 WeCAN |
| `infer.py` | 载入 checkpoint，对单个实例做一次 forward 和调度 |
| `plot_results.py` | 从评估 JSON 生成 makespan 柱状图 |

#### 正确性与诊断

| 文件 | 作用 |
|---|---|
| `run_oracle_validation.py` | 运行手工 + 随机实例的 MILP/exhaustive 交叉验证 |
| `run_diagnostics.py` | 显式启动一个 Phase 1.5 diagnostic gate |
| `reanalyse_a1a.py` | 不重训，重新解释旧 A1 结果 |
| `augment_frozen_diagnostic_report.py` | 给冻结诊断补充确定性最终轨迹信息 |
| `screen_a1b_candidates.py` | A1-b 预注册 seed 筛选，不训练 |
| `screen_a1b_exact.py` | A1-b-exact 4000–4099 Oracle/heuristic 筛选 |
| `screen_a1c_candidates.py` | A1-c active-wait fixture 筛选 |
| `audit_a1b_distribution.py` | 审计已保存 A1-b 候选分布 |

#### 论文规模性能与差分验证

| 文件 | 作用 | 当前价值 |
|---|---|---|
| `run_gpu_short.py` | A100 小规模 batch=64、17-task 左右的一次/短更新能力检查 | 已留下 1-update 报告，不等于 500-task 正式训练 |
| `run_preflight_training.py` | 500-task、K=8 的 forward/rollout/backward/batch 1/2/4 分项计时 | 旧对话的训练吞吐结论来源；输出未单独保存到 `results/` |
| `profile_generator.py` | N=50/100/200 scaling、CPU/GPU scores A/B、cProfile | 第二轮热点定位脚本 |
| `oracle_capture.py` | 在改 Generator 前抓取 reference 输出 | 用于优化前 correctness freeze |
| `diff_test_save_inputs.py` | 保存固定 instance、score 和 skip tensors | 差分测试输入准备 |
| `diff_test_decode.py` | 在旧/新代码上比较 greedy 与 sampled decode | 注释中写明 reference `87b2bb6` 与 cache `6aafbc5` |
| `diff_test_grad.py` | 对 sampled log-probability backward，比较 score/skip 梯度 | 确保优化不破坏 REINFORCE 梯度 |

注意：这些差分脚本依赖旧 Git worktree 或两个独立代码目录。当前 `.git` 缺失后，脚本仍在，但原有复现流程不能原样执行，需先重建 reference/current 两份源码快照。

### 4.11 `tests/`

当前根目录共有 14 个测试文件、41 个显式 `test_*` 函数：

| 文件 | 覆盖范围 |
|---|---|
| `test_phase1.py` | 随机实例、baseline feasibility、Generator、单次 forward、skip 单调性 |
| `test_phase15_model.py` | paper profile shape、LDD mask/folding、skip parameterization |
| `test_phase1_paper_config.py` | paper YAML 契约、forward、rollout、Validator、REINFORCE backward |
| `test_reinforce_baselines.py` | baseline 数学、LOO 条件、K-trajectory 日志 |
| `test_validator.py` | placement/容量/trace 边界错误 |
| `test_time_advance_metrics.py` | active wait 与 passive advance 分离统计 |
| `test_milp_oracle.py` | MILP optimality 与 tick 转换 |
| `test_exhaustive_oracle.py` | 完整搜索、active wait、规模保护 |
| `test_oracle_crosscheck.py` | 手工 + 50 随机 tiny instances 的双 Oracle 一致性 |
| `test_handcrafted_edge_cases.py` | 固定边界场景 |
| `test_a1_diagnostics.py` | A1 指标语义与候选可重复性 |
| `test_a1b_exact.py` | A1-b-exact 生成和代表实例 |
| `test_frozen_diagnostics.py` | 冻结 fixture、历史目录保护、gate 复现 |
| `fixtures.py` | 测试实例辅助函数，不是测试用例本身 |

审计时无法运行 pytest，因为当前 macOS 系统 Python 3.9.6 中没有安装 `torch/numpy/yaml/pulp/pytest/matplotlib`。使用独立临时 bytecode 缓存做了纯语法编译，当前所有 Python 文件均通过 `compileall`。因此：

> 现在只能证明“语法完整”，不能把旧文档中的 `106 passed` 或当前 41 个测试声明为已在迁移后环境重新通过。

### 4.12 `docs/`

| 文件 | 作用 | 审计注意 |
|---|---|---|
| `wecan_analysis.md` | WeCAN 论文架构、可复现边界与本地实现假设 | 仍是理解 baseline 的重要资料 |
| `phase1_report.md` | 最初 Phase 1 工程闭环报告 | 早期状态，不代表当前最新进度 |
| `phase15_plan.md` | Phase 1.5 可信基线计划 | 文首仍写“待确认”，但其中部分工作后来已执行 |
| `phase15_report.md` | Oracle 完成、A1 重分类、后续诊断状态 | 中间阶段报告 |
| `phase1_acceptance_report.md` | Phase 1.6 fixed-set 验收报告 | 引用的 manifest 核心产物现已恢复并通过哈希校验 |
| `phase16_data_report.md` | Phase 1.6 冻结数据集构造与 Oracle 审计报告 | 已从 21 个内容一致的 worktree 快照之一恢复 |
| `phase16_train_report.md` | Phase 1.6 300-update 训练、验证锁定与测试报告 | 已从一致快照恢复 |
| `phase1_paper_baseline_smoke_report.md` | 论文参数配置与 1-update 端到端检查 | 能证明路径跑通过，不能证明性能或收敛 |
| `a1b_structured_hard_distribution.md` | 旧 structured-hard 分布建议 | 已被拆分方案取代 |
| `a1b_scale_distribution.md` | 更大规模 A1-b proposal | 尚未执行正式 gate |
| `project_overview.md` | 截至旧 Git `45fda42` 的全项目总览，包含 Phase 2.1c | **与当前根目录不一致，只能作为恢复线索和历史记录** |

### 4.13 `results/`

| 路径 | 内容 | 判断 |
|---|---|---|
| `phase1_smoke.json/.png` | 早期 smoke 评估与图 | 工程检查，不是论文结果 |
| `single_instance_smoke.json` | 单实例推理结果 | 工程检查 |
| `baselines_smoke.json` | baseline smoke 结果 | 工程检查 |
| `gpu_short_test/report_1upd.json` | A100、batch=64、平均约 17.9 tasks、K=8、1 update；约 32.69 s | 证明小图 paper-size 模型可训练，不代表 500-task 吞吐 |
| `oracle/crosscheck_2026.jsonl` | 60 个双 Oracle 交叉验证记录 | 当前保留的重要正确性证据 |
| `diagnostics/A1-b-screen/` | 100 个候选，未选出符合旧条件的实例 | screening only |
| `diagnostics/A1-b-exact-screen/` | 100 个候选，50 个 accepted，选中 seed 4002 | 已完成筛选 |
| `diagnostics/A1-b-exact-train/` | 300 updates、best/last checkpoint、gate | gate 通过；final greedy 达 Oracle 8 |
| `diagnostics/A1-c-screen/` | 选中 6-task active-wait fixture | 已完成筛选 |
| `diagnostics/A1-c-train/` | 300 updates、best/last checkpoint、gate | gate 通过；final greedy 达 Oracle 13，并需要 active wait |
| `diagnostics/A1/` | 旧 A1 训练与 checkpoint | gate 未通过；历史上已被重解释 |
| `diagnostics_gap/A1/` | 另一组旧 A1 诊断 | gate 未通过 |
| `phase1_acceptance_manifest.json` | Phase 1.6 文件哈希与路径清单 | 清单仍在；所列 13 项核心产物现已全部恢复并匹配 |
| `phase16_final/` | 24 train、12 validation、30 test，共 66 个冻结实例、selected manifest 与 Oracle audit | 已补回 `results/phase16_final/`；manifest 所列哈希全部匹配 |
| `phase16_train_seed2026/` | initial/best/last checkpoint、history、validation/test evaluation 和 train report | 已补回 `results/phase16_train_seed2026/`；manifest 所列哈希全部匹配 |
| `../checkpoints/gpu_short_test/` | paper-profile、batch=64、K=8、平均约 17.9 tasks 的 1-update best/last checkpoint | 226 MB/个；是工程能力检查，不是 500-task 正式权重 |
| `../checkpoints/wecan_smoke/` | 早期 smoke 模型 3 epoch best/last checkpoint | 12 MB/个；仅用于早期小图 smoke |

## 5. 已完成工作的可信梳理

### 5.1 WeCAN 基础闭环

当前源码已经具备：

- 异构 pool、二维累积资源容量、task-pool compatibility；
- DAG precedence；
- WeCA + LDDGNN + 静态 task-pool score；
- 递减 skip score；
- active wait 与被动时间推进；
- K-rollout REINFORCE；
- checkpoint 写入/恢复接口；
- Random、Greedy、HEFT 类基线；
- 独立 Validator；
- MILP 与 exhaustive Oracle。

正确说法是“faithful reimplementation”，不能称作官方代码复现，也不能声称已复现论文数值结果。

### 5.2 小规模可信诊断

本地结果支持以下结论：

- Oracle 交叉验证：60 条记录、0 个状态/数值差异失败；
- A1-b-exact：final policy greedy 从 12 降至 Oracle 8，gate 通过；
- A1-c：final policy greedy 从 24 降至 Oracle 13，且最佳轨迹包含 active skip，gate 通过；
- 这些结论只证明小规模、精心构造的工程诊断能够学到预期行为，不证明对论文规模或一般分布的泛化。

### 5.3 论文参数配置和端到端 smoke

`configs/phase1_paper.yaml` 已配置：

- `d_high=512`、`d_low=128`；
- 8-head WeCA；
- 8-layer、16-head LDDGNN；
- `Dmax=500`；
- `128→64→64→3` skip head，GELU + elementwise Sigmoid；
- Adam、`lr=1e-4`；
- batch 64、800 updates；
- 本地假设 `K=8`。

旧 smoke 证明完整 paper profile 能进行 forward、sample rollout、backward、optimizer step、保存并重载 checkpoint。但当时用的是极小本地数据，不是论文 500-task 正式训练。

### 5.4 论文风格 500-task synthetic 数据

`data/paper_computation_graph.py` 已实现：

- layered；
- Erdős–Rényi DAG；
- stochastic block DAG；
- 默认每个 problem 由 10 个互不连边的 50-task 子图组成，共 500 tasks；
- 论文指定的 task duration GMM、processor/memory demand、task type、pool capacity 和 compatibility；
- 论文未明确的拓扑细节均在代码注释中标记为本地 assumption。

当前 `data/generated/*.json` 仍是 12–24 task smoke 数据，并没有保存一套正式 500-task train/validation/test 数据集。性能脚本会运行时按 seed 临时生成 500-task problem。

### 5.5 第一轮 Generator 性能修复

历史对话记录的实测为：

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| 500-task greedy Generator | 约 317 s | 约 2.9–3.5 s |
| 加速 |  | 约 109× |

根因是旧实现反复访问 `instance.parents/children`，每次 property access 都重新扫描 edges。当前 `scheduler/generator.py` 已在 decode 开始构建 `_TopologyCache`，证明第一轮修复确实存在于迁移后的源码中。

历史对话还记录了 old commit `87b2bb6` 与 cache commit `6aafbc5` 的严格差分验证：覆盖 greedy/sample decisions、placements、makespan、Validator、log probability、score gradient 与 skip gradient，差异为 0。对应的差分测试脚本已保留，但差分输出文件没有保存在当前 `results/`，且旧 Git reference 已随 `.git` 丢失。因此这项结论当前属于“有脚本和对话记录支持，但不能在本机立即重放”的证据等级。

### 5.6 论文规模训练吞吐预检

历史对话记录的 500-task、A100、paper profile、K=8 实测为：

| 阶段 | 时间 |
|---|---:|
| 单实例 neural forward | 约 4.87 s |
| 单条 sampled rollout | 约 3.56 s |
| K=8 rollout | 约 28.5 s |
| backward | 约 2.18 s |
| batch=1 完整 update | 约 35.9 s |

batch scaling：

| batch | 实测总时间 |
|---:|---:|
| 1 | 约 34.1 s |
| 2 | 约 67.1 s |
| 4 | 约 144.1 s |

因为 `training/reinforce.py` 明确以 Python 循环逐 instance forward，再逐条执行 K 次 decode，所以粗略外推 batch=64 约 38 分钟/update、800 updates 约 21 天。结论不是“数学上不能训练”，而是：

> 按当前实现，论文规模正式训练在工程上不具备可接受的迭代可行性。

这些数值的脚本仍在，但原始终端输出/JSON 没有落入 `results/`；迁移后无法在当前无依赖、无 GPU 环境中复测。

## 6. 当前精确停点：第二轮 Generator 优化前

最近一次 profile 已把热点定位到 `_dispatch_mask()`。历史对话记录的单条 500-task sampled rollout 约有：

- 994 个 decision；
- `_dispatch_mask` 约 994 次；
- `_ready_time` 约 86.9 万次；
- `communication_delay_ticks` 约 217.8 万次；
- capacity check 约 149.1 万次。

当前代码仍对每个 decision 扫完整 `N×P` action space，并为已经 scheduled、incompatible 或 parents 未完成的 action 做本可避免的 capacity/ready-time 工作。

已经确认的下一步不是改成：

```python
for task in unscheduled:
```

因为这会缩短 mask、改变 action index，且 set 迭代顺序不代表原始 task ordering。

正确方向必须继续保持完整 `N×P` mask 和固定映射：

```text
(task0,pool0), (task0,pool1), ...,
(task1,pool0), ...
```

只把判断顺序改成尽早返回 `False`：

```text
task 已 scheduled？      → 原位置 False
task-pool incompatible？ → 原位置 False
parents 未完成？         → 原位置 False
capacity 不足？          → 原位置 False
以上均通过               → 才计算 ready_time
```

实施后必须重新做：

1. old/new greedy decisions 完全一致；
2. 多 seed sampled decisions 完全一致；
3. placements、makespan、Validator 完全一致；
4. log probability 一致；
5. `task_pool_scores.grad` 与 `skip_parameters.grad` 一致；
6. 重新 profile 单 rollout、K=8 和真实 update。

在这一步完成前，不建议启动 batch=64、800 updates 正式训练。

## 7. checkpoint 审计

### 7.1 仍不存在的正式 checkpoint

以下关键权重不存在：

- 任何 paper 500-task、batch=64、800-update WeCAN checkpoint；
- Phase 2.1c fixed-priority allocator 的正式 checkpoint。

由于旧对话明确说明 800-update 正式 baseline 尚未开始，第一项不是单纯“迁移丢失已完成成果”，而是本来就没有完成正式训练。Phase 2.1c 的开发 checkpoint 仍可能存在迁移缺失。

### 7.2 当前已恢复的 checkpoint

Phase 1.6 验收权重已按原路径恢复，且三个哈希全部与 manifest 匹配：

```text
results/phase16_train_seed2026/checkpoint/{initial,best,last}.pt
```

此外新增：

```text
checkpoints/gpu_short_test/{best,last}.pt
checkpoints/wecan_smoke/{best,last}.pt
```

前者是完整 paper-size 模型的 1-update GPU short-run 权重；后者是早期 smoke 权重。二者都不是 500-task、800-update 正式 baseline。

原先已有的 8 个 diagnostic checkpoint 仍在：

```text
results/diagnostics/A1/checkpoint/{best,last}.pt
results/diagnostics_gap/A1/checkpoint/{best,last}.pt
results/diagnostics/A1-b-exact-train/checkpoint/{best,last}.pt
results/diagnostics/A1-c-train/checkpoint/{best,last}.pt
```

每个 diagnostic 文件约 1.8 MB，和 64-dim smoke/diagnostic 模型规模相符。它们的用途是复现小规模 gate，不应作为论文 baseline 继续训练的起点。

## 8. 迁移一致性问题与风险

### 8.1 原 Git 历史仍丢失，新的管理基线已推送到远程

当前迁移快照已经建立新的 Git 历史，并推送到私有远程仓库 `git@github.com:jiayiH123/dag-scheduling-wecan.git`。后续变更已有可回退、可同步的管理基线，但原历史仍无法通过 `log/diff/checkout` 访问。`.claude/worktrees/*/.git` 都只是文本指针，指向迁移前的：

```text
/mnt/volumes/.../dag_0729/.git/worktrees/...
```

该路径在当前机器不存在。没有原主 `.git` object database 时，这些 worktree 不能恢复原提交关系、分支、tag 或 ignored-file 历史。新建本地仓库只会从当前迁移快照开始新的历史，不会伪造旧 commit 关系。

风险：

- 不能证明根目录精确对应哪个 commit；
- 不能安全回放 `87b2bb6 → 6aafbc5 → bd9da0d`；
- 不能识别迁移前的 untracked/modified 状态；
- 后续修改前没有可靠回退点。

### 8.2 根目录与历史总览错位

`docs/project_overview.md` 以旧 commit `45fda42` 为审查基线，描述了 Phase 2.1c fixed-priority 路线。当前根目录至少缺少：

```text
models/fixed_priority_allocator.py
models/fixed_priority_allocator_graph.py
scheduler/complete_assignment_realizer.py
scheduler/fixed_priority_allocation_decoder.py
scheduler/priority_policies.py
oracle/fixed_priority_assignment_oracle.py
baselines/priority_allocation_greedy.py
data/phase21c_master_generator.py
多份 run_phase21c_*.py、配置、报告和结果
```

这些源码大多仍存在于 19 个左右的 `.claude/worktrees` 快照中。这说明“后来路线全部不可恢复”并不成立；但不同 worktree 可能含并行 agent 修改，不能直接把任一整个目录覆盖到根目录。

### 8.3 Phase 1.6 验收产物已恢复

用户补回 Phase 1.6 数据和训练目录后，已将它们移动到 manifest 记录的原始路径，并从 21 个内容一致的 worktree 快照中恢复两份配置和两份报告。manifest 列出的 13 项文件现已全部存在，SHA-256 13/13 完全匹配。Phase 1.6 的迁移完整性问题已基本解决。

### 8.4 运行环境未恢复

当前系统 Python 为 3.9.6，项目依赖全部未安装。历史验收环境记录的是 Python 3.12.3、Torch 2.7.0a0 CUDA build、NumPy 1.26.4、PyYAML 6.0.2、PuLP 3.0.0、pytest 8.1.1，且论文规模性能数据来自 A100。

迁移后的当前检查结果：

- 全部 Python 源码纯语法编译通过；
- pytest 无法启动，因为 `pytest` 未安装；
- `torch/numpy/yaml/pulp/matplotlib` 均未安装；
- 当前机器环境不能验证 CUDA/A100 性能数据。

### 8.5 结果证据保存不完整

第一轮 topology-cache 差分结果、500-task preflight 详细输出、第二轮 profile 计数目前主要保存在旧对话中，脚本虽在，但没有对应的稳定 `results/*.json`。后续复测应把环境、配置、commit/源码 hash、原始分项时间与 correctness comparison 一并落盘。

### 8.6 两套 `.gitignore` / `.gitattributes` 规则的影响

当前找到两套不同规则。

**A. 用户补充版 `.gitignore`** 主要忽略：

- Python cache、build/egg 产物；
- `.env`、`*.env`、临时配置；
- log、编辑器、macOS、backup、Jupyter cache；
- 任意层级的 `.claude`；
- 任意层级的 `_runs`；
- `data/raw`、`data/sft`；
- `*.out`。

用户补充版 `.gitattributes` 将 `*.db` 和 `*.jsonl` 交给 Git LFS。

**B. `.claude/worktrees` 快照版 `.gitignore`** 明确忽略：

- `checkpoints/`；
- `results/**/checkpoint/`；
- `results/**/*.pt`；
- `results/**/*.cbc.log`；
- `data/generated/`；
- `.claude/worktrees/`；
- 常规 Python cache、`.venv`、临时包和归档文件。

快照版 `.gitattributes` 没有 Git LFS 规则，只为 `docs/phase1_acceptance_report.md` 保留 Markdown hard-break 尾随空格。

由此可得：

1. `.claude/worktrees` 不属于正常 Git 历史。它们现在存在，是因为迁移了整个文件夹，而不是因为 Git 保存了它们。因此这些快照可用于人工抢救文件，但不能替代丢失的主 `.git`。
2. checkpoint 和 `.pt` **在 worktree 对应的仓库阶段确实被忽略**，所以无法指望从普通 Git clone/pull 恢复这些权重。它们只能从旧服务器文件系统、备份、对象存储或现存迁移副本找回。
3. 用户补充版没有忽略 `results/`、`checkpoints/` 或 `*.pt`，说明它可能来自另一历史阶段、上层仓库或后来调整。没有主 `.git` 时，暂时无法证明两套规则的提交先后关系。
4. 用户补充版第一条写作 `**pycache**/`，很可能可以匹配包含 `pycache` 的目录，但不如标准的 `**/__pycache__/` 清晰。新仓库建议改成标准写法并用 `git check-ignore -v` 验证。
5. 用户补充版的 `**/.claude` 会忽略整个 `.claude`，快照版只忽略 `.claude/worktrees/`；这也说明两者并非同一文件的小差异。
6. 当前机器没有安装 Git LFS。若最终选择用户补充版的 LFS 策略，需要先安装 Git LFS，避免把 pointer 或大型 JSONL 以错误方式提交。

当前新仓库已经选择一种统一策略：

- 源码、配置、测试、文档和小型可审计结果进入普通 Git；
- `.claude`、cache、生成数据、`checkpoints/`、`*.pt`、`*.pth`、`*.ckpt` 不进入 Git；
- 大型 checkpoint 进入独立对象存储，同时在 Git 中保存 manifest、SHA-256、生成配置和远端位置；
- 当前 JSONL 文件均较小，按普通文本管理，不启用 Git LFS。

该策略已写入新建的 `.gitignore`、`.gitattributes` 和 `README.md`。

## 9. 建议的恢复和继续顺序

### P0：先冻结迁移快照

1. 暂时不要删除 `.claude/worktrees`，其中包含根目录缺失的 Phase 2.1c 源码和文档。
2. 已对照两套规则重建统一 `.gitignore`、`.gitattributes`，并初始化新的 `main` 仓库；后续需保留这套 artifact 管理约定。
3. 如果旧服务器、云盘或原机器还可访问，现在只需优先找回主 `.git/`、Phase 2.1c 开发权重及其他未纳入 manifest 的 ignored checkpoint；Phase 1.6 已完成校验恢复。
4. 若找不回主 `.git`，把“当前 root”和“某个完整 Phase 2.1c worktree 快照”分别复制到独立恢复区做文件级比较，不要直接覆盖合并。

### P1：恢复可验证环境

1. 使用项目级虚拟环境，建议 Python 3.12 以贴近历史环境；
2. 安装 `requirements.txt`；
3. 先在 CPU 跑 41 个当前测试，再在目标 CUDA 环境跑 paper-profile 与 Generator 测试；
4. 把实际 Python/Torch/CUDA/GPU 和测试结果写入新的迁移验收记录。

### P2：完成当前明确的性能任务

1. 在 `scheduler/generator.py::_dispatch_mask` 实施保持 `N×P` ordering 的 short-circuit；
2. 重建 reference/current 两份源码并运行 decode + gradient differential tests；
3. 把差分结果落到 `results/performance/` 或同类稳定目录；
4. 在同一 A100、同一 instance、同一 seeds 下复测单 rollout、K=8、batch=1/2/4；
5. 若仍值得，再做“无通信实例 fast path”，但不能改变 communication-enabled 语义；
6. 再决定是否需要 K-rollout 并行化和 batch forward 重构。

### P3：正式 baseline 训练

只有当吞吐达到可接受水平并通过差分验证后，才：

- 固化 paper synthetic train/validation/test seeds 或生成协议；
- 保存初始化 checkpoint；
- 启动 batch=64、800-update baseline；
- 定期保存 `last`、基于 validation 的 `best`、history、环境信息和可恢复的 optimizer state；
- 训练稳定后再进入 communication、cost/budget 或 Phase 2.1c 路线。

## 10. 当前不建议做的事

- 不要把当前 8 个 diagnostic checkpoint 当作 paper baseline checkpoint；
- 不要直接启动 800 updates；
- 不要在 short-circuit 时改用 `for task in unscheduled`；
- 不要在恢复前删除 `.claude/worktrees`；
- 不要把 `docs/project_overview.md` 描述的 Phase 2.1c 当作当前根目录已经可运行；
- 不要用旧 `__pycache__` 是否存在判断源码是否执行成功；
- 不要在没有重跑测试的情况下沿用“106 passed”作为迁移后测试状态；
- 不要在 baseline 性能关卡未过时同时引入通信、预算、PPO 或其他算法改动。

## 11. 建议采用的当前项目状态表述

可以把项目对外/对自己的当前状态统一写成：

> 已完成 WeCAN faithful reimplementation、论文参数配置、论文风格 500-task synthetic generator、独立 Validator 与双 Oracle 正确性基础设施，并完成 Generator topology cache 的语义等价加速。Phase 1.6 冻结数据、训练权重和核心验收产物已按 manifest 完整恢复；新的 Git 管理基线已建立并推送到私有远程仓库。当前论文规模训练仍受逐 rollout Generator 和逐实例 trainer 吞吐限制，第二轮 `_dispatch_mask` correctness-preserving short-circuit 尚未实施，800-update 正式 baseline 尚未开始。迁移后仍缺失原 Git 历史和 500-task 正式 checkpoint，且根目录与历史 Phase 2.1c 总览存在文件错位；部分缺失源码仍可从 `.claude/worktrees` 快照恢复。

## 12. 核心文件 SHA-256（迁移审计基线）

以下哈希可用于确认后续文件是否发生变化：

| 文件 | SHA-256 |
|---|---|
| `requirements.txt` | `300b4a9e1ba114ab55bea3cf590bc58ba1f527c6ac20bc3f0da3b6c80489b151` |
| `configs/phase1_paper.yaml` | `53f94b1df570dd3a3e75a4256038981a7806b934590ab762eee2d911bafd7a0f` |
| `data/instance.py` | `5e266551bd5336db217dcdb6592c8df0328a3bd8fffe3c758b1a9a9b9b8afafd` |
| `data/paper_computation_graph.py` | `f889d2968532f592a23ed386075e0d60a76bb2c87d6cea94eb847f1ee1bd911a` |
| `models/wecan.py` | `94e47f8a203c01aeefc823620b0575bf53d97dc4cbf80329dc710d3011177c96` |
| `scheduler/generator.py` | `99a67b8bc8e08d8b4ea32cf503e0df7e2342ea0099a1b518c2eebfa03b8af148` |
| `scheduler/validator.py` | `5e27a53b52868423985d987143604d17e3d268a8e3a49612345f31a634489bab` |
| `training/reinforce.py` | `9e1f6113f1be19ad2e6cd3a62a2e737db1e8887ec165acc455e7f6acdef9d3c0` |
| `scripts/profile_generator.py` | `dbb2e75ce1015cb60dfd08b1a5aaab1382b173027e60882cfb3c3a7e3b52eaa2` |
| `scripts/run_preflight_training.py` | `35d276b32c82c05c94acb4b4760963a041289879e0b307b5ccc8eb8e35663b94` |

## 13. 审计结论

新的版本控制基线已经初始化并推送到私有 GitHub 仓库；下一步是恢复可运行依赖环境。之后技术主线非常明确：继续第二轮 Generator 等价优化，复测论文规模吞吐，再决定是否进入正式 800-update baseline。

这次迁移没有保留主 Git 历史和 500-task 正式 checkpoint，但核心 WeCAN/Generator/Oracle/诊断代码并未丢失，Phase 1.6 已完成精确恢复，Phase 2.1c 的大量缺失源码也仍藏在 worktree 快照中。因此项目可恢复，且当前研究停点可以被较精确地续接。
