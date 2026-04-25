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

## 为什么采用 label-only

很多真实场景里，外部只能访问模型最终输出的标签，而拿不到：

- 训练数据
- 模型参数
- 置信度分数

本项目关注的正是这种限制更强、也更贴近真实黑盒环境的设定。  
因此，我们的重点不是重新训练模型，而是研究在 **仅有标签反馈** 的条件下，能否仍然恢复出成员信息风险。

## 审计原理

本项目的判定不是依赖单一指标，而是把两类互补的信号放到一起：

- 一类来自 **迁移攻击**
- 一类来自 **边界攻击**

这两类信号分别回答两个不同的问题：

- 这个样本在 shadow 模型上的行为，是否更像训练成员
- 这个样本在 target 模型决策边界附近的几何特征，是否更像训练成员

最后再把两边的判断融合成最终风险结论。

### 1. 迁移攻击信号

迁移攻击这部分由 shadow 模型提供。

基本思路是：

- 先用 shadow 模型模拟一份与 target 模型相近的成员 / 非成员行为分布
- 对待审计样本，在 shadow 模型上取 logits
- 再结合 target 模型对该样本的预测标签计算交叉熵损失

如果一个样本在 shadow 模型上的损失更低，通常意味着它的行为更接近成员样本分布。  
本项目会先离线统计 shadow member 和 non-member 的 loss 分布，再从中生成 `tau_95` 和 `tau_opt` 两个阈值，写入 `shadow_config.json`。审计时，样本的 shadow loss 会与这两个阈值比较，形成第一路风险信号。

### 2. 边界攻击信号

边界攻击这部分使用的是 HSJA（HopSkipJumpAttack）式黑盒边界搜索。

在 label-only 设定下，我们拿不到置信度或梯度，因此不能直接用白盒或分数型攻击。  
HSJA 的作用是只通过模型标签反馈，不断查询 target 模型，估计样本到决策边界的距离。

本项目里，边界攻击不是只测一次原图，而是会：

- 先对原始样本做一次边界距离测量
- 再生成一组轻微扰动变体
- 分别测量这些变体到决策边界的距离

这样得到的不是单点结果，而是一组几何特征。我们会从中提取：

- 平均边界距离 `MeanDistance`
- 波动系数 `VolatilityCV`

直观上看，成员样本通常离决策边界更远，并且在小扰动下更稳定；非成员样本更容易靠近边界，波动也更明显。  
本项目会先用一批有效路人样本做 calibration，得到 `tau_d` 和 `tau_cv`，作为第二路风险信号的判据。

### 3. 两类信号如何融合

本项目不会只看 shadow loss，也不会只看边界距离，而是把两路信号做离散化后再融合。

- 迁移攻击侧会给出 `RED / YELLOW / GREEN`
- 边界攻击侧也会给出 `RED / YELLOW / GREEN`

融合规则是偏保守的：

- 两路都强烈指向成员时，直接判为高风险成员
- 只要有一路明显偏红，或者两路都给出中间风险，就提升最终风险等级
- 只有两路都偏安全时，才判为非成员

这样做的目的很直接：  
单独依赖某一种信号容易误判，而把行为分布信号和几何稳定性信号结合起来，可以让结果更稳，也更容易解释。

### 4. 为什么使用 Go

本项目没有把所有逻辑都写在 Python 里，而是采用了 `Go + Python` 的拆分方式。

原因主要有三点：

- Go 更适合做审计编排  
  样本读取、任务分发、并发探测、结果汇总这些流程天然更适合用 Go 组织。

- Go 更适合承接大量黑盒查询  
  边界攻击会对多个样本、多个变体反复发起查询，本项目里这部分并发量比较高，Go 在 goroutine 和工程组织上更顺手。

- Python 更适合承接模型推理  
  target / shadow 模型本身依赖 PyTorch，继续放在 Python 侧加载和提供 HTTP 服务，改动成本最低，也便于后续替换模型。

所以本项目的分工是：

- Go 负责审计主流程和并发调度
- Python 负责模型加载与推理接口

这种拆分不是为了“混合用语言”，而是为了把审计编排和模型推理解耦，既方便维护，也方便后续替换攻击策略或模型文件。

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
