# Label-Only-MIA-Go

`Label-Only-MIA-Go` 是一个基于 Go 和 Python 的 label-only membership inference audit 项目。  
当前仓库已经整理成适合演示、比赛提交和后续维护的半发布结构：模型服务通过 Docker 启动，审计主流程由 Go 程序完成。

## 一分钟上手

### 1. 环境要求

- 已安装 Docker 和 Docker Compose

### 2. 推荐演示命令

```bash
./scripts/docker/run_smoke.sh
```

这个命令会：

- 启动 target 模型服务
- 启动 shadow 模型服务
- 运行一次缩小规模的审计流程
- 在 `output/audit_report.json` 生成审计结果

这是最适合比赛现场或第一次检查环境的运行方式。

### 3. 完整运行命令

```bash
./scripts/docker/run_halfrelease.sh
```

这个命令会运行默认参数下的完整审计流程。  
由于项目使用黑盒边界攻击，完整运行时间会明显长于 smoke test。

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

## 这个项目适合怎么展示

如果是比赛或答辩现场，推荐这样用：

1. 先运行 `./scripts/docker/run_smoke.sh`
2. 展示容器启动、模型加载、审计完成和报告生成
3. 再展示提前离线跑好的完整结果

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
- 适合整理比赛提交包或演示包

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
  - 维护和比赛交付说明

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

如果只是正常演示或提交，一般不需要改这些值。

## 维护说明

- 更换 shadow 模型后，先执行 `./scripts/docker/rebuild_shadow_config.sh`
- 如果只想快速确认环境，不要直接跑完整版，先跑 `./scripts/docker/run_smoke.sh`
- 如果修改了 Docker、模型路径或配置，建议重新做一次 smoke test

更详细的维护清单见：

- `docs/maintenance-checklist.md`

比赛交付说明见：

- `docs/competition-delivery.md`
