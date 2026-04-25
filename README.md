# Label-Only-MIA-Go

`Label-Only-MIA-Go` 是我们实现的一套基于 Go 和 Python 的 label-only membership inference audit 系统。  
本项目提供了基于 Docker 的运行方式，用来统一 Python 推理环境、减少复现时的环境差异，并简化模型服务与 Go 审计流程的启动过程。

## 一分钟上手

### 1. 环境要求

- 已安装 Docker 和 Docker Compose

### 2. 快速运行命令

```bash
./scripts/docker/run_smoke.sh
```

该命令会：

- 启动 target 模型服务
- 启动 shadow 模型服务
- 运行一次缩小规模的审计流程
- 在 `output/audit_report.json` 生成审计结果

这是最适合第一次验证环境和快速复现流程的运行方式。

### 3. 完整运行命令

```bash
./scripts/docker/run_halfrelease.sh
```

该命令会运行默认参数下的完整审计流程。  
由于本项目使用黑盒边界攻击，完整运行时间会明显长于 smoke test。

## 运行后会看到什么

程序正常结束后，终端会输出：

- 阈值诊断信息
- 审计样本数量
- Accuracy / Precision / Recall

并且会生成：

- `output/audit_report.json`

报告中每个样本包含：

- 样本编号
- 标签
- 是否为真实成员
- shadow loss
- 边界距离均值
- 波动系数
- 最终风险结论

## 本项目做什么

本项目用于对分类模型执行 **label-only membership inference audit**。

这里的核心问题是：

- 给定一个目标模型
- 只允许查询模型输出的类别标签，不直接访问训练数据和内部参数
- 判断某个样本是否更像“训练时见过的成员样本”

我们把这个问题实现成了一套可运行的审计流程，而不是只停留在理论分析。

## 基本原理

整个流程由两部分组成：

1. Go 侧负责审计编排  
   包括样本读取、并发调度、边界攻击、结果融合和报告输出。

2. Python 侧负责模型推理服务  
   负责加载 target / shadow 模型，并通过 HTTP 接口向 Go 侧提供预测结果。

在审计时，我们主要结合两类信号：

- **shadow signal**
  - 用 shadow 模型对样本行为做迁移式判断
  - 通过 logits 计算 loss，衡量样本是否更接近成员分布

- **geometry signal**
  - 用黑盒边界攻击估计样本到决策边界的距离
  - 再结合多个微扰变体的波动情况，判断样本是否更像训练成员

最后，我们会把这些信号融合成一份更直观的风险结论。

## 为什么采用 label-only

很多真实场景里，外部只能访问模型最终输出的标签，而拿不到：

- 训练数据
- 模型参数
- 置信度分数

本项目关注的正是这种限制更强、也更贴近真实黑盒环境的设定。  
因此，我们的重点不是重新训练模型，而是研究在 **仅有标签反馈** 的条件下，能否仍然恢复出成员信息风险。

## 本项目的意义

本项目的价值主要在三个方面：

- 我们把 label-only MIA 从概念和公式落成了可运行系统
- 我们把 Go 的调度能力和 Python 的模型服务结合起来，便于复现和扩展
- 本项目提供了从模型加载、审计执行到结果输出的完整链路，方便后续继续改模型、改阈值和改攻击策略

## 推荐使用方式

第一次使用时，建议这样做：

1. 先运行 `./scripts/docker/run_smoke.sh`
2. 确认容器启动、模型加载、审计完成和报告生成都正常
3. 再运行 `./scripts/docker/run_halfrelease.sh`

原因很简单：  
这里慢的不是训练，而是审计阶段反复进行黑盒边界攻击查询。模型虽然已经训练好，但完整审计仍然需要较长时间。

## 常用命令

### 快速自检

```bash
./scripts/docker/run_smoke.sh
```

用途：

- 验证 Docker 环境是否正常
- 验证模型文件是否能加载
- 验证 target/shadow 服务是否互通
- 验证 Go 审计流程是否能跑通

### 完整审计

```bash
./scripts/docker/run_halfrelease.sh
```

用途：

- 按默认参数运行完整审计流程
- 适合正式生成结果

### 重算影子模型阈值

```bash
./scripts/docker/rebuild_shadow_config.sh
```

用途：

- 重新生成 `shadow_config.json`
- 在更换 shadow 模型后必须执行一次

### 生成交付包

```bash
./scripts/docker/package_release.sh
```

用途：

- 在 `dist/labelscan-halfrelease/` 下生成一份独立交付目录
- 适合整理一份独立的复现目录

## 默认使用的文件

### 模型文件

- Target：
  - `python_server/CIFAR10/target/3000/best_checkpoint_ep.pth`
- Shadow：
  - `python_server/CIFAR10/shadow/3000/best_checkpoint_ep.pth`

### 数据文件

- 成员样本：
  - `data/cifar-10-batches-bin/data_batch_1.bin`
- 非成员样本：
  - `data/cifar-10-batches-bin/test_batch.bin`

### 配置文件

- 审计阈值：
  - `shadow_config.json`

## 项目结构

- `main.go`
  - Go 审计主入口
- `pkg/`
  - Go 侧核心逻辑，包括攻击、审计、数据读取、并发调度、数学工具
- `python_server/server.py`
  - Python 模型服务入口
- `python_server/calc_thresholds.py`
  - shadow 阈值重算脚本
- `docker-compose.yml`
  - Docker 编排入口
- `scripts/docker/`
  - 运行、检查、打包脚本
- `docs/`
  - 维护和使用说明

## 可配置项

可选环境变量见 `.env.example`，常用的有：

- `TARGET_MODEL_PATH`
- `SHADOW_MODEL_PATH`
- `SHADOW_CONFIG_PATH`
- `OUTPUT_REPORT_PATH`
- `CALIBRATION_CANDIDATE_COUNT`
- `CALIBRATION_TARGET_COUNT`
- `MEMBER_SAMPLE_COUNT`
- `NON_MEMBER_SAMPLE_COUNT`
- `AUDIT_WORKERS`

如果只是正常复现和运行，一般不需要改这些值。

## 维护说明

- 更换 shadow 模型后，先执行 `./scripts/docker/rebuild_shadow_config.sh`
- 如果只想快速确认环境，不要直接跑完整版，先跑 `./scripts/docker/run_smoke.sh`
- 如果修改了 Docker、模型路径或配置，建议重新做一次 smoke test

更详细的维护清单见：

- `docs/maintenance-checklist.md`
