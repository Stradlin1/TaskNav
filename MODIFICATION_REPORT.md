# TaskNav / Independent LaneRobotV2 修改报告

> 更新日期：2026-10-03  
> 当前主线：main  
> 当前阶段：baseline 六项模型正确性代码修复已完成，尚未开始新一轮正式精度训练。

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

这些协议与评测修复之上，本轮又完成了下述六项 Loss、配置和测试修复。

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

### 3.4 lane_label_smoothing 曾是继承的死配置

`default.yaml` 中原本存在该项，但 LaneRobotLoss 未读取。本轮已实现其真实作用并加测试。

## 4. 本轮已完成的六项修改

1. 修 lane_loc 可导训练路径；
2. smooth / curv 改 GT-relative geometry；
3. 新增 lane loss gradient / geometry 单测；
4. 新增 CE + Exist + Offset core baseline 配置；
5. 新增 8x10 / 10x10 / 16x16 Pool 消融配置；
6. 实现 lane_label_smoothing。

具体实现：

- `_decode_x_train()` 仅对 `0..159` 的 logits 做 softmax 完整期望，并加 clamp 后的 signed offset；`_decode_x()` 与部署解码仍为 hard argmax + offset。
- smooth 比较 `d1_pred` 与 `d1_gt`，curv 比较 `d2_pred` 与 `d2_gt`，分别使用 valid pair / triplet mask。
- hard CE 使用 PyTorch label smoothing；visible soft target 只在可见 x-grid 上做 uniform smoothing，no-lane target 为 0；invalid Row 走 no-lane CE。
- `train.py` 新增可选 `--cfg`，不传时仍使用 `ultralytics/cfg/default.yaml`，同一 cfg 同时用于模型构建和 `model.train()`。
- core baseline：`ultralytics/cfg/experiments/lane_core_baseline.yaml`。
- full-loss 配置：`lane_full_loss_8x10.yaml`、`lane_full_loss_10x10.yaml`、`lane_full_loss_16x16.yaml`。
- 新模型配置：`yolo26s-lane-independent-10x10.yaml`、`yolo26s-lane-independent-16x16.yaml`；原 8x10 控制组未修改。

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

本轮已经修改：

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

## 8. 本轮实际验证

~~~text
pytest tests: 21/21 PASS, 6 subtests PASS
core baseline random legal batch loss/backward: PASS
full-loss 8x10 random legal batch loss/backward: PASS
core 1-image train + 1-image val, 64x64, 1-epoch CPU smoke: PASS
full-loss 8x10 1-image train + 1-image val, 64x64, 1-epoch CPU smoke: PASS
8x10 complete model build/forward/backward: PASS
10x10 complete model build/forward/backward: PASS
16x16 complete model build/forward/backward: PASS
8x10 historical best.pt ONNX checker / two-output ORT parity: PASS
10x10 random full model ONNX checker / two-output ORT parity: PASS
16x16 random full model ONNX checker / two-output ORT parity: PASS
~~~

ONNX 输出仍为 `cls_logits [1,161,56,4]` 与 `offset [1,1,56,4]`。历史 8x10 `best.pt` 回归中，最大绝对误差分别为 `4.5776367e-05` 与 `3.4198165e-06`；随机初始化 10x10 / 16x16 的最大绝对误差均不超过 `2.2351742e-08`。

未运行：全量数据 1–3 epoch smoke、core/full 120 epoch 从头训练、三组 Pool 正式精度消融。微型训练只验证 Trainer/DataLoader/optimizer/loss/Validator 链路；历史权重的 parity 只能证明接口未回归，两者都不能代表修复后 Loss 的新精度。

## 9. Baseline 冻结记录

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
