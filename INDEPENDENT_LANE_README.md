# TaskNav 四任务 Independent LaneRobotV2 基线

> 更新日期：2026-10-03  
> 当前主线：main  
> 当前阶段：先修模型本体正确性，再重新训练可信 baseline；大创研究增强在 baseline 冻结后进行。

## 1. 当前模型

~~~text
RGB 640x640
  ↓
YOLO26 Backbone
  ↓
P4 / P5 Fusion
  ↓
LaneRobotV2Independent
  ├── branch 0
  ├── branch 1
  ├── branch 2
  └── branch 3
~~~

四个 branch 共享 Backbone / Fusion，但每个 prediction branch 独立包含：

~~~text
Conv1x1
AdaptiveAvgPool2d
Flatten
FC1
ReLU
cls_fc2
offset_fc
~~~

task 维度之间没有 Softmax 竞争。模型不学习 task 名称文本。

比赛与大创分开训练，不使用 task supervision mask 混训。

## 2. 当前输出与几何协议

默认：

~~~text
imgsz       = 640
x_grids     = 160
row_anchors = 56
num_lanes   = 4
~~~

输出：

~~~text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
~~~

分类：

~~~text
0..159 = x grid
160    = no-lane
~~~

连续坐标：

~~~text
target_x = x_norm * (x_grids - 1)
~~~

离散目标：

~~~text
target_class = round(target_x)
~~~

offset：

~~~text
offset_gt = target_x - target_class
~~~

推理解码：

~~~text
pred_x = argmax(cls_logits) + offset
~~~

这套 nearest-grid + signed-offset 协议继续保留。

## 3. Row Anchor 与 strict manual protocol

~~~text
row 0  -> y = 1.0
row 55 -> y = 0.3333333333
order  -> bottom-to-top
~~~

当前标签：

~~~text
lane_id x0 y0 x1 y1 ... x55 y55
~~~

当前 x：

- [0,1]：有效点；
- -1：不存在；
- strict baseline 当前不接受 -2。

训练器已在 train / val DataLoader 创建前执行全量 label preflight。

## 4. 当前真正需要修的模型问题

### 4.1 lane_loc 训练路径不可导

当前 V2 推理使用 hard argmax 是正确的，但训练时 lane_loc 如果也用：

~~~text
argmax(cls) + offset
~~~

则 lane_loc 无法通过 argmax 给 cls logits 几何梯度。

目标：

- inference 继续 hard argmax；
- training lane_loc 使用可导 soft position；
- lane_loc.backward() 后 cls logits 梯度必须非零。

### 4.2 smooth / curvature 旧定义偏向直线

旧逻辑：

~~~text
d1_pred -> 0
d2_pred -> 0
~~~

会主动惩罚真实弯曲结构。

目标：

~~~text
d1_pred -> d1_gt
d2_pred -> d2_gt
~~~

只在连续 valid GT Row 上计算。

### 4.3 新增 Loss 单测

新增：

~~~text
tests/test_lane_loss_geometry.py
~~~

至少检查：

- lane_loc 对 cls logits 有非零梯度；
- lane_loc 对 offset 有梯度；
- 完美 GT 曲线的 smooth / curv 接近 0；
- 直线预测弯曲 GT 时几何 loss 增大；
- invalid Row 不参与；
- Loss finite。

### 4.4 新增 core baseline

新增独立实验配置，只启用：

~~~text
CE + Exist + Offset
~~~

即：

~~~text
lane_loc    = 0
lane_smooth = 0
lane_curv   = 0
~~~

用来验证 Backbone / Fusion / Head / nearest-grid / signed-offset 本身。

### 4.5 Pool size 消融

当前每个 branch 固定：

~~~text
AdaptiveAvgPool2d(8,10)
~~~

该设置来自早期 256x320 LaneRobot 设计，而当前输入为 640x640。

增加控制实验：

~~~text
8x10
10x10
16x16
~~~

除 feat_h / feat_w 外其他训练条件保持一致。

在结果出来前，不直接删除 8x10。

### 4.6 lane_label_smoothing 必须生效

当前配置项：

~~~yaml
lane_label_smoothing: 0.02
~~~

目前 Loss 没有真正读取。

本轮要求实现：

- LaneRobotLoss 读取该参数；
- 校验 [0,1)；
- hard-label CE 生效；
- soft-label CE 也有明确 smoothing 语义；
- 0.0 与非 0 值在测试中产生不同 loss。

## 5. 六项修复后的 Loss

~~~text
L =
  lambda_ce     * L_ce
+ lambda_loc    * L_loc_differentiable
+ lambda_exist  * L_exist
+ lambda_smooth * L_first_order_gt
+ lambda_curv   * L_second_order_gt
+ lambda_offset * L_signed_offset
~~~

注意：

- training loc position 可以使用 soft expectation；
- inference 仍使用 hard argmax + offset；
- smooth / curv 必须相对 GT 几何计算。

## 6. 本轮不要顺手改的东西

本轮不要引入：

- Evidence Head；
- Visibility prediction；
- task supervision mask；
- 比赛 / 大创混训；
- 动态 N；
- Transformer / GNN / BEV / 时序；
- Depth Branch；
- Completion Module；
- Row Anchor 数量变化；
- nearest-grid + signed-offset 推理解码变化。

## 7. 推荐新增实验配置

建议在项目中建立明确 experiment configs，例如：

~~~text
ultralytics/cfg/experiments/
  lane_core_baseline.yaml
  lane_full_loss_8x10.yaml
  lane_full_loss_10x10.yaml
  lane_full_loss_16x16.yaml
~~~

要求每个配置都能独立复现，避免训练前手工改 default.yaml。

## 8. 验证顺序

~~~text
1. 修 lane_loc
2. 修 smooth / curv
3. 实现 label smoothing
4. 加 Loss tests
5. 建 core baseline config
6. 建 8x10 / 10x10 / 16x16 configs
7. 跑 protocol / validator / loss tests
8. 1~3 epoch smoke
9. 跑 core baseline
10. 跑完整修正版 Loss
11. 跑 Pool 消融
12. ONNX parity
13. 从头正式训练
14. 冻结 baseline
~~~

## 9. Baseline 冻结后才进入大创研究增强

大创主线：

~~~text
Complete Geometry
+ Visibility Mask
+ Navigation Cue Dropout
+ Missing-region weighted loss
+ specialized robustness benchmark
+ RDK X5 deployment
~~~

Visibility 只作为标签 / loss weighting / evaluation metadata，不新增 Evidence Head。

## 10. 当前测试

已有：

~~~text
tests/test_tasknav_lane_protocol.py
tests/test_lane_validator_metrics.py
~~~

本轮新增：

~~~text
tests/test_lane_loss_geometry.py
~~~

仓库当前没有 GitHub Actions workflow，因此仍需在训练环境手动执行测试。

## 11. 模型修复完成判据

至少满足：

- lane_loc 对 cls logits 有非零梯度；
- GT-relative smooth / curv 逻辑正确；
- label smoothing 真正生效；
- core baseline 可复现；
- 8x10 / 10x10 / 16x16 均能构建 / forward / backward；
- smoke 无 NaN / Inf；
- cls / offset 输出 shape 不变；
- ONNX 两输出协议不变；
- ORT parity 通过。

完整 Codex 修改任务见：

~~~text
CODEX_MODEL_FIX_TARGET.md
~~~
