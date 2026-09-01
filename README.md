# DAG 异构调度与 WeCAN 忠实重实现

本项目研究离线异构 DAG 调度，当前代码主线是 WeCAN（Weighted Cross-Attention Network）的 faithful reimplementation，以及用于验证调度语义的 Generator、Validator、MILP/穷举 Oracle 和 REINFORCE 训练基础设施。

当前研究停点：论文风格 500-task synthetic problem 已可运行，Generator 第一轮 topology-cache 优化已完成；第二轮 `_dispatch_mask` correctness-preserving short-circuit 尚未实施，论文规模 batch=64、800-update 正式训练尚未开始。

完整迁移审计、文件说明和进度见：

- [`docs/migration_project_audit_2026-09-01.md`](docs/migration_project_audit_2026-09-01.md)
- [`docs/wecan_analysis.md`](docs/wecan_analysis.md)
- [`docs/phase1_acceptance_report.md`](docs/phase1_acceptance_report.md)
- [`docs/phase1_paper_baseline_smoke_report.md`](docs/phase1_paper_baseline_smoke_report.md)

## 目录

```text
baselines/       Random、Greedy、HEFT 等基线
configs/         smoke、paper profile 与诊断配置
data/            DAGInstance、fixture、随机与论文 synthetic generator
models/          WeCAN 模型
scheduler/       Generator、Validator、schedule 类型与动作上界
training/        REINFORCE 与诊断 gate
oracle/          MILP 与 exhaustive Oracle
evaluation/      统一评估与汇总
scripts/         数据、训练、推理、评估、profile 和差分测试入口
tests/           单元与语义一致性测试
docs/            论文分析、阶段报告和迁移审计
results/         小型、可审计的实验元数据与历史结果
```

## 环境

建议使用 Python 3.12 的项目级虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

论文规模性能测量需要 CUDA GPU；历史性能数据来自 NVIDIA A100。

## 仓库存储策略

远程 Git 仓库管理源码、配置、测试、文档和小型审计结果。以下本地运行产物默认不推送：

- `.claude/` 和 Python cache；
- `checkpoints/`；
- `results/**/checkpoint/` 与 `results/**/*.pt`；
- `data/generated/`；
- 临时运行目录和日志。

重要 checkpoint 应保存到独立对象存储，并在 Git 中保留路径、配置、环境和 SHA-256 manifest。

## 复现边界

该实现不是论文官方代码。论文没有公开完整源码、原始训练数据和 checkpoint；未明确的架构及训练细节在配置和文档中显式标记为本地假设。因此项目可以用于语义与本地实验复现，但不能直接宣称复现了论文数值结果。
