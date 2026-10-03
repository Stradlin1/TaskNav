# TaskNav 四任务 Independent LaneRobotV2 基线

> 更新日期：2026-10-03  
> 当前主线：main  
> 当前阶段：六项模型正确性代码修复和 1 epoch 微型训练 smoke 已完成，等待全量数据 smoke、从头重训与 Pool 消融；大创研究增强在 baseline 冻结后进行。

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

## 4. 已完成的六项模型正确性修复

### 4.1 lane_loc 可导训练路径

推理继续使用：

~~~text
argmax(cls) + offset
~~~

训练几何损失改用：

~~~text
soft_x = sum(softmax(cls_logits[0:160]) * grid_index)
train_x = soft_x + clamp(offset, -0.5, +0.5)
~~~

`lane_loc` 对 `train_x` 与 `target_x` 的归一化坐标计算 SmoothL1。梯度测试确认 cls x-grid logits 与 offset 均得到非零梯度；推理路径未改变。

### 4.2 smooth / curvature 对齐 GT 几何

旧逻辑 `d1_pred -> 0`、`d2_pred -> 0` 已删除，当前为：

~~~text
d1_pred -> d1_gt
d2_pred -> d2_gt
~~~

一阶项只使用相邻两行均 valid 的 pair，二阶项只使用连续三行均 valid 的 triplet；两边均除以 `x_grids - 1` 后计算 SmoothL1。

### 4.3 Loss 单测

新增：

~~~text
tests/test_lane_loss_geometry.py
~~~

当前覆盖：

- lane_loc 对 cls logits 有非零梯度；
- lane_loc 对 offset 有梯度；
- 完美 GT 曲线的 smooth / curv 接近 0；
- 直线预测弯曲 GT 时几何 loss 增大；
- invalid Row 不参与；
- 四任务 total 与六个分量 finite；
- 8x10 / 10x10 / 16x16 branch shape 与 backward；
- hard / soft label smoothing 语义与非法边界。

### 4.4 core baseline

`ultralytics/cfg/experiments/lane_core_baseline.yaml` 只启用：

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

### 4.5 Pool size 消融配置

当前每个 branch 固定：

~~~text
AdaptiveAvgPool2d(8,10)
~~~

该设置来自早期 256x320 LaneRobot 设计，而当前输入为 640x640。

已提供：

~~~text
8x10
10x10
16x16
~~~

除 feat_h / feat_w 外其他训练条件保持一致。

在结果出来前，不直接删除 8x10。

### 4.6 lane_label_smoothing 已生效

当前配置项：

~~~yaml
lane_label_smoothing: 0.02
~~~

`LaneRobotLoss` 读取并校验 `[0,1)`。hard-label 分支直接传给 `F.cross_entropy`；visible soft-label 分支在 `0..159` 上混合 uniform，no-lane 概率保持 0；invalid/no-lane Row 继续走带 label smoothing 的 no-lane CE。

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

## 7. 实验配置

当前配置：

~~~text
ultralytics/cfg/experiments/
  lane_core_baseline.yaml
  lane_full_loss_8x10.yaml
  lane_full_loss_10x10.yaml
  lane_full_loss_16x16.yaml
~~~

四份配置的数据、seed、epochs、batch、optimizer 保持一致；三份 full-loss 配置只切换 model YAML，避免训练前手工改 `default.yaml`。

## 8. 验证状态与后续顺序

~~~text
1. 六项代码修改：完成
2. protocol / validator / loss tests：21/21 通过
3. core / full 8x10 随机合法 batch loss backward：通过
4. 8x10 / 10x10 / 16x16 完整模型 build + forward + backward：通过
5. 历史 8x10 best.pt 与随机初始化 10x10 / 16x16 ONNX checker / ORT parity：通过
6. core / full-loss 8x10 的 1 图 train + 1 图 val、64×64、1 epoch CPU 微型训练：通过
7. 全量数据 1~3 epoch smoke：未运行
8. core baseline / full corrected loss 正式训练：未运行
9. Pool 消融与从头正式训练：未运行
10. 冻结 baseline：待正式实验结果
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

仓库当前没有 GitHub Actions workflow。本轮在 `lane_robot` 环境安装 pytest 9.1.1 后执行三份指定测试和整个 `tests/`，实际 21 项及 6 个 subtests 全部通过；为避开系统 ROS pytest 插件，运行时设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。

## 11. 模型修复状态

代码级判据已经满足：

- lane_loc 对 cls logits 有非零梯度；
- GT-relative smooth / curv 逻辑正确；
- label smoothing 真正生效；
- core baseline 可复现；
- 8x10 / 10x10 / 16x16 均能构建 / forward / backward；
- 随机合法 batch loss smoke 无 NaN / Inf；
- cls / offset 输出 shape 不变；
- ONNX 两输出协议不变；
- ORT parity 通过。

尚未完成的是全量数据 1–3 epoch smoke、正式 core/full baseline 重训和三组 Pool 精度消融，因此还不能冻结新的精度 baseline。已完成的微型训练仅验证 Trainer、DataLoader、label preflight、optimizer、loss backward 和 Validator 能连通，不作为精度结果。

完整 Codex 修改任务见：

~~~text
CODEX_MODEL_FIX_TARGET.md
~~~
