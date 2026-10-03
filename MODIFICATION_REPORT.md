# TaskNav / Independent LaneRobotV2 修改报告

> 更新日期：2026-10-03  
> 当前主线：main  
> 当前阶段：baseline 模型正确性修复，尚未开始新一轮正式精度结论。

## 1. 当前模型

~~~text
shared YOLO26 backbone
+ P4/P5 fusion
+ LaneRobotV2Independent
  ├─ branch 0
  ├─ branch 1
  ├─ branch 2
  └─ branch 3
~~~

输出：

~~~text
cls    [B,161,56,4]
offset [B,1,56,4]
~~~

当前几何协议：

~~~text
target_x     = x_norm * (x_grids - 1)
target_class = round(target_x)
offset_gt    = target_x - target_class
decode       = argmax(cls) + offset
~~~

该 nearest-grid + signed-offset 协议保留。

## 2. 已完成的 baseline 工程修复

已经完成：

- Independent four-branch head；
- Row Anchor fallback 统一到 1.0 -> 1/3；
- strict manual parser；
- train / val 全量 label preflight；
- matched-only MAE；
- Exist P/R/F1 / Miss Rate；
- grid-to-pixel 比例修正；
- ONNX shape / checker / ORT parity；
- protocol tests；
- validator tests。

这些改动解决的是协议和评测正确性，不代表 Loss 设计已经完全正确。

## 3. 与原始单任务仓库对比后确认的模型问题

对比 TarochLee/ULTRALYTICS_LANE_ROBOT 后确认：

### 3.1 lane_loc hard-argmax 梯度问题是当前仓库后续改法带来的

原始仓库 lane_loc 通过 soft-argmax 计算位置，因此能向 cls logits 回传梯度。

当前仓库为修正 offset 定义，已经改成正确的 nearest-grid + signed offset，并在 V2 decode 中使用 hard argmax。这个推理协议本身保留，但 lane_loc 也走 hard argmax 后导致几何梯度无法进入 cls logits。

因此正确修法不是回滚到旧 floor-offset，而是：

~~~text
inference: hard argmax + signed offset
training lane_loc: differentiable soft position + offset
~~~

### 3.2 smooth / curvature 直线偏置来自原始仓库

原始仓库使用：

~~~text
d1_pred -> 0
d2_pred -> 0
~~~

当前仓库继承了该设计。

本轮改为：

~~~text
d1_pred -> d1_gt
d2_pred -> d2_gt
~~~

### 3.3 8x10 Pool 来自旧 256x320 设计

当前 640x640 输入仍使用固定 8x10 AdaptiveAvgPool。

它不是已证明的 bug，因此本轮增加：

~~~text
8x10 control
10x10
16x16
~~~

三组消融，不直接拍脑袋替换。

### 3.4 lane_label_smoothing 是继承的死配置

default.yaml 中存在该项，但 LaneRobotLoss 未读取。

本轮实现其真实作用并加测试。

## 4. 本轮六项修改

1. 修 lane_loc 可导训练路径；
2. smooth / curv 改 GT-relative geometry；
3. 新增 lane loss gradient / geometry 单测；
4. 新增 CE + Exist + Offset core baseline 配置；
5. 新增 8x10 / 10x10 / 16x16 Pool 消融配置；
6. 实现 lane_label_smoothing。

详细修改要求见：

~~~text
CODEX_MODEL_FIX_TARGET.md
~~~

## 5. 本轮不改

不要在本轮加入：

- Evidence Head；
- Visibility prediction head；
- task supervision mask；
- 比赛 / 大创混训；
- 动态 N；
- Depth；
- Transformer / GNN / BEV / temporal；
- Completion Module；
- Row Anchor 数量变化；
- nearest-grid + signed-offset inference protocol 变化。

## 6. 大创研究主线修正

旧文档中的 Evidence Head 计划作废。

后续大创主线：

~~~text
Complete Geometry
+ Visibility Mask
+ Navigation Cue Dropout
+ Missing-region weighted loss
+ Visible / Missing robustness evaluation
+ RDK X5 deployment
~~~

Visibility 是 supervision / weighting / evaluation metadata，不要求模型预测。

比赛模型和大创模型分开训练。

## 7. 重新训练要求

由于已经修改过：

- offset target；
- decode；
- soft-label center；
- validator；
- strict protocol；

且本轮还要修改：

- loc；
- smooth；
- curvature；
- label smoothing；

因此旧 checkpoint 不能用于代表最终 baseline 精度。

顺序：

~~~text
pytest
-> 1~3 epoch smoke
-> core baseline
-> full corrected loss baseline
-> pool ablation
-> ONNX parity
-> full retrain
-> freeze baseline
~~~

## 8. Baseline 冻结记录

至少记录：

~~~text
commit SHA
model yaml
experiment config
data yaml
dataset version
seed
pool size
loss weights
best epoch
matched MAE / MAE px
Acc@1 / Acc@3 / Acc@5
Exist P / R / F1 / Miss
params / latency
ONNX size
PyTorch -> ORT parity
~~~
