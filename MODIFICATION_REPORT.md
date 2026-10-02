# TaskNav / Independent LaneRobotV2 修改报告

> 更新日期：2026-10-02  
> 当前主线代码基线：main（2026-10-02，已统一核心 Row Anchor fallback）  
> 说明：本文区分“当前代码已实现”“已有历史验证”“仍需本机重新验证”三类状态。

## 1. 当前基线范围

项目当前以四任务 Independent LaneRobotV2 为基线：

~~~text
shared backbone / neck / P4+P5 fusion
  ├─ independent branch 0: Conv1x1 -> Pool -> FC1 -> cls/offset
  ├─ independent branch 1: Conv1x1 -> Pool -> FC1 -> cls/offset
  ├─ independent branch 2: Conv1x1 -> Pool -> FC1 -> cls/offset
  └─ independent branch 3: Conv1x1 -> Pool -> FC1 -> cls/offset
~~~

当前默认 tensor protocol：

~~~text
x_grids:       160
no-lane index: 160
row_anchors:   56
num_tasks:     4

cls:           [B, 161, 56, 4]
offset:        [B,   1, 56, 4]
~~~

当前根目录 ONNX 导出采用两个独立输出：

~~~text
cls_logits
offset
~~~

Head 内部的 concat export 兼容形式仍可存在，但不是当前 export_onnx.py 的默认部署契约。

## 2. 已完成的主要代码改造

### 2.1 Independent Head

已经加入：

~~~text
SingleLaneRobotV2Branch
LaneRobotV2Independent
~~~

每个 task 拥有独立 prediction branch；task 之间不共享 cls/offset Head 参数。

### 2.2 四任务 Loss

LaneRobotLoss 支持四任务输出，并按 task 维分别计算后聚合。

当前 V2 定位协议已经统一为：

~~~text
continuous target_x = x_norm * (x_grids - 1)
nearest class       = round(target_x)
offset_gt           = target_x - nearest class
decode              = argmax(cls) + offset
~~~

soft-label 以 nearest discrete class 为中心，避免与 offset 重复表示同一 fractional residual。

### 2.3 Row Anchor / manual 标签协议

当前正式 manual 协议：

~~~text
row 0  = y 1.0
row 55 = y 0.3333333333
order  = bottom-to-top
rows   = 56
~~~

data YAML：

~~~text
ultralytics/cfg/datasets/lane-robot-4tasks.yaml
~~~

已启用：

~~~yaml
strict_labels: true
~~~

新增：

~~~text
ultralytics/models/yolo/lane/protocol.py
tests/test_tasknav_lane_protocol.py
~~~

严格检查包括：

- 56 对 x/y；
- 固定 y 序列；
- bottom-to-top 顺序；
- x=-1 或 [0,1]；
- task_id 范围；
- 重复 task_id；
- NaN / Inf；
- strict 模式缺失 txt。

### 2.4 Validator

Validator 已重构为“定位质量”和“存在性”分离。

定位：

~~~text
matched MAE
matched MAE px
Acc@1
Acc@3
Acc@5
~~~

存在性：

~~~text
Miss Rate
Precision
Recall
F1
Accuracy
~~~

no-lane sentinel -1 不再参与位置 MAE。

MAE_px 比例修正为：

~~~text
(width - 1) / (x_grids - 1)
~~~

当前 fitness：

~~~text
Acc@3
+ 0.5 * Acc@5
- 0.003 * matched_MAE
+ 0.05 * Exist_F1
~~~

新增：

~~~text
tests/test_lane_validator_metrics.py
~~~

### 2.5 ONNX 导出

export_onnx.py 已重构：

- 使用 argparse；
- 默认权重指向项目 runs/lane/train/weights/best.pt；
- 支持 --weights / --output / --imgsz / --opset / --device；
- 支持单文件或 --external-data；
- 自动读取 Head 的 x_grids / row_anchors / num_lanes；
- 自动校验输出 shape；
- onnx.checker 门禁；
- 默认 ONNX Runtime parity；
- 检查输出名称、shape、NaN / Inf；
- atol / rtol 数值一致性检查。

当前仍是：

~~~text
cls_logits
offset
~~~

两输出。

## 3. 历史验证记录

初始 Independent Head 改造阶段曾记录以下历史验证：

~~~text
Gate 1: copied independent branch 与单任务 LaneRobotV2 数值一致
Gate 2: cls [1,161,56,4], offset [1,1,56,4]
Gate 3: 四个 branch Parameter storage 独立
Gate 4: 单 task loss 只给对应 branch 非零梯度，共享 backbone 有梯度
Gate 5: joint per-task loss finite，四个 branch 均有梯度
Gate 6: 四个 lane_id 可加载为 [56,4]
Gate 7: 四个单任务 checkpoint 可映射到 branch 0..3
~~~

这些结果属于早期 Independent baseline 的历史验证，不应替代最新 main 上的重新 smoke test。

## 4. 当前新增测试的状态

仓库当前包含：

~~~text
tests/test_tasknav_lane_protocol.py
tests/test_lane_validator_metrics.py
~~~

GitHub 当前没有 Actions workflow / commit status 自动运行这些测试。

因此当前准确表述是：

> 测试用例已经加入仓库，但最新 commit 没有 GitHub CI 结果证明它们已自动通过。

正式训练前应在目标训练环境执行：

~~~bash
pytest tests/test_tasknav_lane_protocol.py
pytest tests/test_lane_validator_metrics.py
~~~

## 5. 当前仍存在的代码问题

### 5.1 Row Anchor fallback 已统一

核心 Lane 模块的最后 fallback 现已统一为：

~~~text
1.0 -> 0.3333333333
~~~

与 default.yaml、lane-robot-4tasks.yaml 和 strict manual protocol 一致。标准配置缺失部分几何参数时，不再回退到历史 0.67 -> 1.0 协议。

### 5.2 默认 lane-robot.yaml 仍为 legacy

~~~text
ultralytics/cfg/datasets/lane-robot.yaml
~~~

当前仍是旧单任务、旧机器路径、旧 y 几何。

四任务 baseline 不使用它，但 Ultralytics 通用 lane 默认映射仍可能引用该文件，因此后续应重定向或更新。

### 5.3 动态 N 尚未实现

当前 num_lanes=4 同时存在于：

~~~text
default.yaml
lane-robot-4tasks.yaml
yolo26*-lane-independent.yaml
~~~

当前四任务一致，但 TaskNav 的“default.yaml 为唯一运行时 N 真值源”还没有实现。

### 5.4 Helper / inference 脚本仍有历史参数

部分根目录 helper 和独立推理脚本仍含历史机器路径或固定常量。

这些脚本当前不作为四任务训练正确性的唯一依据。

### 5.5 文档此前长期滞后

本次文档更新已经统一以下事实：

~~~text
row0  = 1.0
row55 = 1/3
strict manual protocol
matched-only MAE
exist P/R/F1
ONNX 两输出 + ORT parity
动态 N 尚未实现
~~~

## 6. 当前不能继续沿用的旧实验结论

在以下规则变化之后：

- offset GT 修正；
- V2 decode 修正；
- soft-label 中心修正；
- Validator / fitness 修正；
- strict manual protocol 固化；

旧规则下训练得到的 checkpoint 不能直接用来代表当前代码的最终精度。

历史 checkpoint 可以用于：

- 结构兼容检查；
- 权重迁移实验；
- debug；

但最新 baseline 精度必须重新训练得到。

## 7. 当前推荐验证流程

~~~text
1. 处理 legacy lane-robot.yaml，并确认核心 Row Anchor fallback 已统一
2. 本机跑 protocol + validator 单元测试
3. 1~3 epoch smoke run
4. 检查 loss / shape / metrics / plots / checkpoint
5. 导出 ONNX
6. 通过 ONNX Runtime parity
7. 从头正式训练 baseline
8. 记录 best epoch 与完整四任务指标
9. 冻结 baseline
10. 再开始 TaskNav Evidence / Missing Geometry 实验
~~~

## 8. Baseline 冻结时应记录

至少保存：

~~~text
commit SHA
default.yaml
model YAML
data YAML
dataset version
train / valid split
manual protocol version
random seed
best epoch
matched MAE / MAE px
Acc@1 / Acc@3 / Acc@5
Miss Rate
Exist Precision / Recall / F1 / Accuracy
PyTorch -> ONNX parity
推理输入尺寸
ONNX opset
~~~

只有这些信息固定后，后续 TaskNav 改动的收益才可可靠比较。
