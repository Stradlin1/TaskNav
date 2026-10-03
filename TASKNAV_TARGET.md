# TaskNav 目标设计文档

> 更新日期：2026-10-03  
> 当前状态：四任务 Independent LaneRobotV2 baseline 仍处于模型正确性修复阶段。  
> 当前代码基线：main。  
> 原则：比赛模型与大创研究模型分开训练；当前先修模型本体，再进入 TaskNav 研究增强。

## 1. 项目定位

TaskNav 面向移动机器人在真实赛道中的断线、遮挡、稀疏锥桶、颜色边界和异质视觉载体问题。

核心研究目标不是识别“白线、黄线、锥桶是什么”，而是恢复对控制有意义的导航结构。

大创研究主线使用三个核心导航功能：

~~~text
current_guide
left_boundary
right_boundary
~~~

比赛工程可以使用四个独立输出。比赛与大创分开训练，不使用 task supervision mask 将两套监督强行混在同一个训练任务中。

task_id 只是固定输出槽位。模型本身不理解名称，也不是 VLM，因此不把命名问题当成模型改造重点。

## 2. 当前模型结构

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

四个 branch 共享 Backbone / Fusion，但 prediction branch 独立：

~~~text
Conv1x1
AdaptiveAvgPool2d
Flatten
FC1
ReLU
cls_fc2
offset_fc
~~~

当前输出：

~~~text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
~~~

当前 V2 几何协议：

~~~text
target_x     = x_norm * (x_grids - 1)
target_class = round(target_x)
offset_gt    = target_x - target_class
decode       = argmax(cls) + offset
~~~

signed offset 范围约为：

~~~text
[-0.5, +0.5]
~~~

这套 nearest-grid + signed-offset 定义继续保留，不回退到旧的 floor-offset 定义。

## 3. 当前 manual Row Anchor 协议

~~~text
row_anchors = 56
row 0       = y 1.0
row 55      = y 0.3333333333
order       = bottom-to-top
~~~

x 当前定义：

~~~text
x in [0,1] : 当前 Row 有 Geometry 点
x = -1     : 当前 Row 不存在该结构
~~~

strict parser 当前不接受 -2。后续 Complete Geometry / ignore 扩展时再修改协议。

## 4. 当前已经完成的 baseline 加固

已经完成：

- Row Anchor fallback 统一；
- 四任务 Independent Head；
- nearest-grid + signed offset 协议统一；
- strict manual label parser；
- train / val 全量 label preflight；
- Validator 将定位与存在性分开；
- matched MAE / Acc@1/3/5；
- Exist Precision / Recall / F1 / Miss Rate；
- ONNX 两输出与 ORT parity 检查；
- protocol / validator 单元测试。

这些工作不代表模型 Loss 已完全正确。当前仍有一轮模型本体修复必须先完成。

## 5. 当前必须修复的六个模型问题

### 5.1 lane_loc 对 cls 几何梯度被 hard argmax 截断

当前 V2 解码：

~~~text
pred_x = argmax(cls_logits) + offset
~~~

推理这样做是正确的，但训练时如果 lane_loc 也直接依赖这个 hard argmax，则 lane_loc 无法通过 argmax 向 cls logits 传播几何梯度。

目标：

- 推理仍使用 hard argmax + signed offset；
- CE 仍监督 nearest grid；
- offset 仍监督 signed residual；
- lane_loc 训练路径必须使用可导的 soft position；
- 不允许为了可导而改变最终部署解码协议。

推荐实现：

~~~text
soft_x = differentiable local expectation from cls logits
train_pred_x = soft_x + offset
lane_loc(train_pred_x, target_x)
~~~

也允许使用 straight-through estimator，但必须通过单测证明 lane_loc backward 后 cls logits 存在非零梯度。

### 5.2 smooth / curvature 不能再把预测往 0 导数压

当前旧逻辑本质上是：

~~~text
d1_pred -> 0
d2_pred -> 0
~~~

这会鼓励预测更直，可能压制真实弯道、绕障和入口转向。

目标改为：

~~~text
d1_pred -> d1_gt
d2_pred -> d2_gt
~~~

即：

~~~text
d1_pred = pred_x[:,1:] - pred_x[:,:-1]
d1_gt   = target_x[:,1:] - target_x[:,:-1]

d2_pred = pred_x[:,2:] - 2*pred_x[:,1:-1] + pred_x[:,:-2]
d2_gt   = target_x[:,2:] - 2*target_x[:,1:-1] + target_x[:,:-2]
~~~

只在对应连续 GT Row 都 valid 时计算。

### 5.3 增加 Loss 单元测试

新增测试至少覆盖：

1. lane_loc 单独 backward 时 cls logits 梯度非零；
2. lane_loc 对 offset 也有有效梯度；
3. 正确曲线 prediction 的 smooth / curv loss 接近 0；
4. 将同一 GT 弯道预测成直线时 smooth / curv loss 明显增大；
5. invalid / no-lane Row 不参与几何差分；
6. nearest-grid + signed-offset target 仍保持现有定义；
7. Loss finite，无 NaN / Inf。

建议新增：

~~~text
tests/test_lane_loss_geometry.py
~~~

### 5.4 增加 CE + Exist + Offset 干净 baseline

需要一个明确可复现的模型正确性基线：

~~~text
lane_ce     > 0
lane_exist  > 0
lane_offset > 0

lane_loc    = 0
lane_smooth = 0
lane_curv   = 0
~~~

目的不是作为最终模型，而是隔离 Backbone / Fusion / Head / nearest-grid / signed-offset 是否本身可以稳定学习。

不要手工每次改 default.yaml。增加独立配置文件，例如：

~~~text
ultralytics/cfg/experiments/lane_core_baseline.yaml
~~~

或项目中同等级的明确 experiment config。

必须可以一条命令复现。

### 5.5 对 8x10 AdaptiveAvgPool 做 Head 消融

当前 640x640 输入经过 P4/P5 fusion 后仍被每个 branch 压到固定：

~~~text
8 x 10
~~~

这是从早期 256x320 LaneRobot 设计继承的设置，对当前 640 输入是否最优尚未验证。

本轮不要直接删除 8x10，而是增加三套可训练模型配置：

~~~text
8x10   control
10x10
16x16
~~~

要求：

- 除 feat_h / feat_w 外保持结构和训练参数一致；
- 不在同一次实验同时改 hidden_dim、reduce_channels 等参数；
- 确保对应 FC 输入维度自动正确；
- 三套都能 forward / backward / export；
- 最终用同一数据、同一 seed、同一训练预算比较。

主要指标：

~~~text
matched MAE / MAE px
Acc@3
Acc@5
Exist F1
弯道 / 远端 Row 子集
参数量
推理延迟
ONNX 大小
~~~

在实验结果出来前，默认 8x10 不直接宣判错误。

### 5.6 lane_label_smoothing 必须真正生效

当前 default.yaml 有：

~~~yaml
lane_label_smoothing: 0.02
~~~

但 LaneRobotLoss 没有读取它，属于死参数。

本轮选择“实现”，不要仅删除。

要求：

- 在 LaneRobotLoss.__init__ 读取 lane_label_smoothing；
- 范围校验建议为 [0, 1)；
- hard-label CE 分支使用该值；
- soft-label CE 分支也必须有明确语义，不能让参数表面存在但实际仍无效；
- 最简单可接受方案：对最终 soft target 做 uniform smoothing；
- no-lane 行仍按 no-lane 分类目标处理，不能被错误当作 visible soft target；
- 增加测试证明 0.0 和非 0 值产生不同 CE。

## 6. 六项修复后的 Loss 目标

完整修正版 baseline：

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

- inference decode 不变；
- loc 的训练位置估计和 inference decode 可以不同；
- smooth / curv 比较 GT 几何，不再比较 0；
- 四个 Independent branch 仍独立算 loss 后聚合。

## 7. 本轮明确不修改

本轮六项模型修复禁止顺手加入：

- Evidence Head；
- Visibility prediction head；
- task supervision mask；
- 比赛 / 大创混合训练；
- 动态 N 重构；
- VLM 文本语义；
- Transformer；
- GNN；
- BEV；
- Depth Branch；
- 时序网络；
- Completion Module；
- 修改当前 Row Anchor 数量；
- 修改当前 nearest-grid + signed-offset 推理解码定义。

## 8. 大创最终研究主线修正

此前旧文档中的 Evidence Head 方案作废。

TaskNav V1 大创研究主线：

~~~text
固定 LaneRobot 主体
+ 三个核心导航功能表示
+ Complete Geometry GT
+ Visibility Mask
+ Navigation Cue Dropout
+ Missing-region weighted loss
+ Visible / Missing robustness evaluation
+ RDK X5 monocular deployment
~~~

Visibility Mask 的作用：

- 数据标注元数据；
- Missing / Visible 分区；
- loss weighting；
- benchmark 切分；
- Cue Dropout 同步更新。

模型不需要输出 visibility / evidence logits。

核心原则：

> 看不见，不等于导航结构不存在。

因此未来 Geometry：

~~~text
x >= 0 : Geometry 存在
x = -1 : Geometry 真正不存在
x = -2 : ignore / 人工也无法可靠确定
~~~

Visibility：

~~~text
1  : 当前直接可见
0  : Geometry 存在，但视觉 cue 缺失
-1 : 不适用
-2 : ignore
~~~

## 9. Cue Dropout 与 Missing weighted loss

在六项 baseline 模型修复、重新训练并冻结之后，再进入研究增强。

Navigation Cue Dropout：

~~~text
图像 cue 被局部擦除 / 遮挡 / 弱化
Geometry GT 保持不变
Visibility 对应位置从 1 变 0
~~~

Missing weighted loss：

~~~text
Visible Geometry weight = 1
Missing Geometry weight = lambda_missing
~~~

lambda_missing 通过消融确定。

## 10. Benchmark

后续至少建立：

~~~text
Test-Normal
Test-Missing
Test-Sparse
Test-Mixed
Test-Distractor
Test-Occlusion
~~~

核心指标：

~~~text
Overall MAE
Visible MAE
Missing MAE
Acc@5 / Acc@10
Exist Precision / Recall / F1
False Completion Rate
~~~

需要保留 spline / polynomial classical baseline，用于证明连续恢复不是后处理拟合单独造成的。

## 11. 开发顺序

~~~text
Stage 0 - Model correctness
  1. 修 lane_loc 可导训练路径
  2. 修 smooth / curv GT-relative geometry
  3. 加 Loss tests
  4. 建 core baseline config
  5. 建 8x10 / 10x10 / 16x16 pool configs
  6. 实现 lane_label_smoothing

Stage 1 - Baseline experiments
  7. pytest
  8. 1~3 epoch smoke
  9. CE + Exist + Offset core baseline
 10. 完整修正版 Loss baseline
 11. Pool size ablation
 12. ONNX parity
 13. 从头正式训练
 14. 冻结 baseline

Stage 2 - TaskNav research
 15. Complete Geometry / -2 ignore
 16. Visibility Mask
 17. Navigation Cue Dropout
 18. Missing weighted loss
 19. Normal / Missing / Sparse / Mixed / Distractor / Occlusion benchmark
 20. spline / polynomial baseline

Stage 3 - Deployment
 21. ONNX
 22. RDK X5 INT8
 23. 控制接口
 24. 实车闭环
~~~

## 12. 本轮完成判据

六项模型修复阶段完成至少满足：

- lane_loc backward 后 cls logits 梯度非零；
- smooth / curv 对完美 GT geometry 近似 0；
- smooth / curv 不再天然偏好直线；
- core baseline 配置可以单独训练；
- 8x10 / 10x10 / 16x16 三个 Head 配置均可构建与训练；
- lane_label_smoothing 非 0 时确实改变分类 loss；
- 原有 protocol tests 通过；
- 原有 validator tests 通过；
- 新增 lane loss tests 通过；
- 1~3 epoch smoke 无 NaN；
- PyTorch 输出协议未变化；
- ONNX 两输出协议未变化；
- ORT parity 通过。

## 13. 一句话定义

> 当前 TaskNav 先把 Independent LaneRobotV2 的几何 Loss 和 Head 实验基线修正确，再进入 Complete Geometry + Visibility Mask + Cue Dropout + Missing weighting 的大创研究主线。
