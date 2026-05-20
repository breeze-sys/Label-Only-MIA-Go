# LabelScan-Go

LabelScan-Go 是一个面向分类模型的 **Label-Only Membership Inference Audit** 检测工具。  
本项目只依赖模型返回的预测标签作为主要攻击反馈，通过影子模型迁移信号和黑盒边界攻击信号，判断输入样本是否存在成员泄露风险。

当前版本以 CIFAR-10 模型为默认示例，提供完整的检测流程、命令行参数、结构化 JSON 报告和可读 HTML 报告。

## 快速运行

在第一个终端启动目标模型服务：

```bash
MODEL_PATH=python_server/CIFAR10/target/3000/best_checkpoint_ep.pth PORT=8000 conda run -n mia python python_server/server.py
```

在第二个终端启动影子模型服务：

```bash
MODEL_PATH=python_server/CIFAR10/shadow_json_aligned/best_checkpoint_ep.pth MODEL_DROPOUT=1 PORT=8001 conda run -n mia python python_server/server.py
```

然后运行一次快速检测：

```bash
go run . --preset smoke
```

运行完成后会生成：

- `output/audit_report.json`
- `output/audit_report.html`

HTML 报告可以直接在浏览器中打开，用来查看检测摘要、风险比例和逐样本结果。

## 常用命令

快速自检：

```bash
go run . --preset smoke
```

默认检测：

```bash
go run . --preset standard
```

更大规模检测：

```bash
go run . --preset full
```

自定义样本数量：

```bash
go run . \
  --preset custom \
  --member-samples 20 \
  --non-member-samples 20 \
  --calibration-targets 10 \
  --workers 20
```

指定输出路径：

```bash
go run . \
  --json-report output/custom_report.json \
  --html-report output/custom_report.html
```

## 工具做了什么

LabelScan-Go 的检测流程分为四步：

1. 读取 `shadow_config.json` 中的影子模型阈值。
2. 从非成员样本池中选取参考样本，现场定标边界距离阈值。
3. 对待审计样本执行 label-only 黑盒检测。
4. 输出终端摘要、JSON 报告和 HTML 报告。

每个样本会同时计算两类信号：

- `shadow loss`：由影子模型 logits 与目标模型预测标签计算得到，用来模拟成员样本行为。
- `boundary distance`：由 HSJA 式边界攻击估计样本到目标模型决策边界的距离。

最终判定会融合迁移攻击信号和边界攻击信号，给出 `confirmed_member`、`high_risk`、`medium_risk` 或 `non_member`。

## 参数说明

模型服务参数：

- `--target-api`：目标模型服务地址，默认 `http://localhost:8000`
- `--shadow-api`：影子模型服务地址，默认 `http://localhost:8001`
- `--shadow-config`：影子模型阈值配置，默认 `shadow_config.json`

数据参数：

- `--calibration-data`：现场定标数据文件
- `--member-data`：成员样本数据文件
- `--non-member-data`：非成员样本数据文件
- `--member-samples`：检测的成员样本数量
- `--non-member-samples`：检测的非成员样本数量

攻击与并发参数：

- `--workers`：Go 侧并发审计工人数
- `--max-queries`：HSJA 单样本最大查询次数
- `--max-iterations`：HSJA 最大迭代轮数
- `--num-evals`：HSJA 每轮梯度估计采样数

报告参数：

- `--json-report`：JSON 报告路径
- `--html-report`：HTML 报告路径，留空则不生成

## 报告内容

JSON 报告包含：

- 本次运行配置
- 迁移信号和几何信号阈值
- Accuracy / Precision / Recall
- 风险样本比例
- 每个样本的 shadow loss、边界距离、波动系数和最终风险等级

HTML 报告提供同样内容的可视化摘要，便于展示和人工检查。

## 影子模型调优

本项目提供 JSON 对齐影子模型训练入口：

```bash
conda run -n mia python python_server/main.py --action 7
```

训练脚本支持：

- 标签平滑
- Dropout
- weight decay
- early stopping
- 训练索引记录

重算阈值：

```bash
SHADOW_MODEL_PATH=python_server/CIFAR10/shadow_json_aligned/best_checkpoint_ep.pth \
SHADOW_TRAIN_INDICES_PATH=python_server/CIFAR10/shadow_json_aligned/train_indices.json \
conda run -n mia python python_server/calc_thresholds.py
```

## 项目结构

- `main.go`：检测工具主入口
- `pkg/attack/`：HSJA 黑盒边界攻击
- `pkg/audit/`：成员风险判定和信号融合
- `pkg/client/`：Go 到 Python 模型服务的 HTTP 客户端
- `pkg/dataset/`：CIFAR-10 二进制数据读取
- `pkg/worker/`：并发审计调度
- `python_server/server.py`：PyTorch 模型推理服务
- `python_server/main.py`：JSON 对齐影子模型训练入口
- `python_server/calc_thresholds.py`：影子模型阈值生成工具
- `output/`：检测报告输出目录

## 测试

运行 Go 单元测试：

```bash
GOCACHE=/tmp/labelscan-go-build-cache go test ./...
```

运行 Python 语法检查：

```bash
conda run -n mia python -m py_compile python_server/main.py python_server/server.py python_server/calc_thresholds.py python_server/classifier.py
```
