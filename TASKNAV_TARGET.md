# TaskNav 目标设计文档

> 更新日期：2026-10-02  
> 当前状态：LaneRobot 四任务 baseline 正确性加固阶段；TaskNav Evidence 功能尚未正式实现。  
> 当前代码基线：main（2026-10-02，已统一核心 Row Anchor fallback）  
> 原则：先得到可复现、可验证、可导出的可信 baseline，再逐步加入 TaskNav 功能，避免同时修改数据、Head、Loss 与部署链路。

## 1. 项目目标

TaskNav 面向移动机器人在真实赛道中的：

- 断线；
- 遮挡；
- 稀疏锥桶；
- 颜色边界；
- 不同视觉载体混合。

目标不是单纯识别“白线 / 黄线 / 锥桶是什么”，而是恢复对机器人控制有意义的导航结构：

~~~text
reference_guide
left_boundary
right_boundary
~~~

核心问题：

> 看不见，不等于导航结构不存在。

因此 TaskNav V1 最终希望同时学习：

1. Geometry：导航结构在哪里；
2. Evidence：当前位置是否存在直接视觉证据。

## 2. 当前 baseline 已经实现的内容

当前结构：

~~~text
RGB
 ↓
YOLO Backbone
 ↓
P4 / P5 Fusion
 ↓
LaneRobotV2Independent
 ├── task 0
 ├── task 1
 ├── task 2
 └── task 3
~~~

当前默认协议：

~~~text
imgsz       = 640
x_grids     = 160
row_anchors = 56
num_lanes   = 4
~~~

当前输出：

~~~text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
~~~

当前 V2 使用：

~~~text
nearest-grid classification
+
signed sub-grid offset
~~~

当前解码：

~~~text
pred_x = argmax(cls) + offset
~~~

当前 manual 标签、Validator 和 ONNX 导出已完成一轮 baseline 加固。

## 3. 当前 manual Geometry 协议

当前 baseline 严格协议：

~~~text
lane_id x0 y0 x1 y1 ... x55 y55
~~~

Row Anchor：

~~~text
row 0  -> y=1.0
row 55 -> y=0.3333333333
order  -> bottom-to-top
~~~

公式：

~~~text
y(row) = 1.0 - row / 55 * (2/3)
~~~

当前 x 定义：

~~~text
x in [0,1] : 该 Row 存在点
x = -1     : 当前 Row 无点
~~~

当前 strict parser 不接受 x=-2。

因此本文后续提出的 TaskNav ignore 值 -2 属于“未来协议扩展”，在真正实现 TaskNav Dataset 前不能直接写入当前 baseline 标签，否则 strict parser 会拒绝。

当前 strict 校验已经覆盖：

- 56 对坐标；
- y 顺序与固定值；
- x=-1 / [0,1]；
- task_id；
- duplicate task；
- NaN / Inf；
- 缺失 txt。

## 4. 当前 Validator 已实现

当前 baseline 已有：

### Geometry / localization

~~~text
matched MAE
matched MAE px
Acc@1
Acc@3
Acc@5
~~~

### Structure existence

~~~text
Miss Rate
Precision
Recall
F1
Accuracy
~~~

matched MAE 只统计 GT 与 prediction 都存在的位置，不再将 pred_x=-1 当作普通坐标。

当前 tolerance accuracy 的分母仍是全部 valid GT，因此漏检会失败。

当前 fitness：

~~~text
Acc@3
+ 0.5 * Acc@5
- 0.003 * matched_MAE
+ 0.05 * Exist_F1
~~~

TaskNav 后续仍需要在 Evidence 标签引入后扩展 Visible / Missing 指标。

## 5. 当前 ONNX baseline 已实现

当前 export_onnx.py：

- 支持 CLI 参数；
- 自动读取 x_grids / row_anchors / num_lanes；
- 校验模型实际输出 shape；
- 默认导出单文件 ONNX；
- 可选 external data；
- 使用 onnx.checker；
- 默认执行 ONNX Runtime parity；
- 比较 PyTorch / ORT 数值。

当前输出：

~~~text
cls_logits
offset
~~~

TaskNav Evidence Head 尚未实现，因此当前不是三输出。

## 6. Baseline 仍需处理的事项

在 TaskNav 功能开发前，仍建议先完成：

### 6.1 Row Anchor legacy fallback 已清理

核心 Lane 模块的最终 fallback 已统一为正式协议：

~~~text
1.0 -> 0.3333333333
~~~

Dataset、Trainer、Validator 和 Plotting 的默认 Row Anchor 语义现在与 strict manual protocol 一致。

### 6.2 通用 lane 默认入口已统一

当前通用 lane 默认映射已经指向四任务 baseline：

~~~text
TASK2DATA["lane"]            -> lane-robot-4tasks.yaml
TASK2CALIBRATIONDATA["lane"] -> lane-robot-4tasks.yaml
TASK2MODEL["lane"]           -> yolo26s-lane-independent.yaml
~~~

lane-robot.yaml 仅作为兼容文件名保留，其内容已同步到当前四任务 strict manual 配置。

### 6.3 本机运行新增测试

当前已有：

~~~text
tests/test_tasknav_lane_protocol.py
tests/test_lane_validator_metrics.py
~~~

仓库暂无 GitHub Actions workflow，不能把“测试已提交”写成“CI 已通过”。

### 6.4 smoke run + 从头正式训练

旧规则下的 checkpoint 不用于判断最新 baseline 精度。

建议：

~~~text
1~3 epoch smoke
  ↓
确认数据 / loss / val / plot / checkpoint / ONNX
  ↓
从头正式训练
  ↓
冻结 baseline
~~~

## 7. TaskNav V1 不做的事情

V1 暂时不引入：

- 独立 Completion Module；
- Transformer；
- GNN；
- BEV 网络；
- 时序网络；
- RGB + Depth 联合训练；
- 大规模跨 Row 新网络。

也不重做当前 YOLO Backbone、P4/P5 Fusion 和 Row-Anchor 主体。

## 8. TaskNav V1 最小 Head 修改

在每个 SingleLaneRobotV2Branch 当前 512-d feature 后新增：

~~~text
                     512-d feature
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       cls_fc2         offset_fc      evidence_fc
          │               │               │
          ▼               ▼               ▼
      x / no-lane       offset       direct evidence?
~~~

目标：

~~~text
evidence_fc = Linear(512, 56)
~~~

TaskNav 目标输出：

~~~text
cls      [B, 161, 56, N]
offset   [B,   1, 56, N]
evidence [B,   1, 56, N]
~~~

默认 N=4。

Evidence 输出 logits，训练用 BCEWithLogitsLoss，推理后 sigmoid。

## 9. 动态 N 目标

当前 baseline 尚未实现动态 N。

当前任务数同时存在于：

~~~text
default.yaml
lane-robot-4tasks.yaml
yolo26*-lane-independent.yaml
~~~

当前均为 4，因此 baseline 一致。

TaskNav 的目标是：

~~~text
default.yaml / runtime args
        ↓
lane_num_lanes = N
        ↓
Model
Dataset
Loss
Validator
Predictor
Plotting
Export
全部使用同一个 N
~~~

目标要求：

1. 默认 N=4；
2. 只改 default.yaml 即可切换 N；
3. data YAML 不再作为任务数独立真值源；
4. model YAML 的 num_lanes 只作为兼容默认值；
5. Head 自动创建 N 个 branch；
6. task weights / task names 长度与 N 校验；
7. checkpoint Head 维度不匹配时明确报错；
8. ONNX 输出最后一维自动为 N。

默认功能名称计划：

~~~text
task 0 -> reference_guide
task 1 -> left_boundary
task 2 -> right_boundary
task 3 -> reserve_3
~~~

## 10. TaskNav 数据语义

TaskNav 必须区分四种情况。

### A. Geometry 存在且直接可见

~~~text
geometry x = 有效位置
evidence   = 1
~~~

### B. Geometry 存在但没有直接视觉证据

~~~text
geometry x = 有效位置
evidence   = 0
~~~

这是 TaskNav 最核心的区域。

例如：

- 白线被遮挡；
- 白线断开；
- 两个稀疏锥桶之间；
- 局部视觉载体消失，但整体几何仍可可靠确定。

### C. Geometry 真正不存在

未来 TaskNav Geometry 协议计划：

~~~text
geometry x = -1
evidence   = -1
~~~

### D. 人工也无法可靠确定

未来计划增加 ignore：

~~~text
geometry x = -2
evidence   = -2
~~~

注意：当前 baseline strict manual protocol 仍只接受 x=-1 或 [0,1]。实现 ignore 前必须先升级 protocol.py / Dataset / tests，不能提前混用。

## 11. Geometry 与 Evidence 标签设计

为尽量保持现有 Geometry 链路稳定，计划分开保存。

### Geometry

继续：

~~~text
labels/train/000123.txt
~~~

未来 TaskNav 语义：

~~~text
x >= 0 : Geometry 存在
x = -1 : Geometry 不存在
x = -2 : ignore
~~~

关键变化：

> 视觉上暂时看不见，不再自动等于 x=-1。

只要结构仍存在且人工可以可靠确定，就继续提供完整 x_gt。

### Evidence

新增：

~~~text
evidence/train/000123.txt
~~~

计划格式：

~~~text
task_id e0 e1 ... e55
~~~

计划值：

~~~text
 1 : 直接视觉证据存在
 0 : Geometry 存在，但直接证据缺失
-1 : Geometry 不存在，Evidence 不适用
-2 : ignore
~~~

## 12. Dataset 目标修改

目标文件：

~~~text
ultralytics/models/yolo/lane/dataset.py
~~~

未来增加：

1. Evidence 文件读取；
2. lane_evidence [56,N]；
3. x=-1 / x=-2 区分；
4. lane_ignore；
5. Geometry 与 Evidence 同步增强；
6. 配置与任务维度动态化。

目标 batch：

~~~text
img
lane
lane_x
lane_y
lane_evidence
lane_ignore
~~~

## 13. Loss 目标修改

当前已有：

~~~text
lane_ce
lane_loc
lane_exist
lane_smooth
lane_curv
lane_offset
~~~

TaskNav 增加：

~~~text
lane_evidence
~~~

总目标：

~~~text
L_total = L_lane_baseline + lambda_evidence * L_evidence
~~~

Evidence 第一版：

~~~text
BCEWithLogitsLoss
~~~

Mask 规则：

- evidence=1：Geometry 正常监督，Evidence target=1；
- evidence=0：Geometry 仍正常监督，Evidence target=0；
- geometry=-1：训练 no-lane，不计算位置 / offset / Evidence；
- geometry=-2：TaskNav 相关 loss 全部 ignore。

## 14. Navigation Cue Dropout

目标：主动生成“直接证据缺失，但 Geometry GT 保持不变”的样本。

可以模拟：

- 连续线局部擦除；
- 颜色边界局部弱化；
- 随机移除部分锥桶；
- 局部遮挡；
- 局部模糊；
- 地面纹理覆盖局部 cue。

发生后：

~~~text
Geometry GT -> 不变
Evidence    -> 对应位置变 0
~~~

需要多种遮挡形式，避免模型只学习固定遮挡外观。

## 15. TaskNav Validator 目标

在当前 baseline Validator 基础上继续增加：

### Geometry

~~~text
Overall MAE
Visible MAE
Missing MAE
Visible Acc
Missing Acc
~~~

定义：

~~~text
Visible = Geometry存在 AND Evidence=1
Missing = Geometry存在 AND Evidence=0
~~~

Missing Region 是核心指标。

### Structure existence

继续保留当前：

~~~text
Precision
Recall
F1
Miss Rate
Accuracy
~~~

后续可增加 False Positive Rate / False Completion。

### Evidence

在 Geometry 存在位置统计：

~~~text
Evidence Accuracy
Evidence Precision
Evidence Recall
Evidence F1
~~~

## 16. Predictor / 可视化目标

未来每个 Row 区分：

### Observed

~~~text
P_exist 高
P_evidence 高
~~~

### Inferred / Completed

~~~text
P_exist 高
P_evidence 低
~~~

### Absent

~~~text
P_exist 低
~~~

调试可视化必须把 Observed 与 Inferred 分开显示。

## 17. TaskNav ONNX / RDK X5 目标

当前 baseline：

~~~text
cls_logits [B,161,56,N]
offset     [B,1,56,N]
~~~

未来 TaskNav：

~~~text
cls_logits      [B,161,56,N]
offset          [B,1,56,N]
evidence_logits [B,1,56,N]
~~~

要求：

- 三输出名称固定；
- N 从运行时模型读取；
- PyTorch / ONNX Runtime parity；
- 后续再做 RDK X5 INT8；
- 检查算子兼容与 BPU / CPU 落点；
- 测 FPS、P50、P95；
- 比较 FP32 / ONNX / INT8 的 Visible / Missing 精度。

## 18. 推荐开发顺序

必须按阶段推进：

~~~text
A. Baseline correctness
   1. Row Anchor fallback 已统一（保持回归测试）
   2. 通用 lane 默认入口已统一（保持配置回归检查）
   3. 跑 protocol / validator tests
   4. smoke run
   5. ONNX parity
   6. 从头正式训练并冻结 baseline

B. TaskNav configuration
   7. 动态 N
   8. 明确 task names / weights

C. TaskNav data
   9. 完整 Geometry GT
  10. ignore mask
  11. Evidence 标签

D. TaskNav model
  12. Evidence Head
  13. Evidence Loss
  14. Visible / Missing / Evidence metrics

E. Robustness
  15. Navigation Cue Dropout
  16. 缺失比例实验
  17. 异质视觉载体实验
  18. 防幻觉实验

F. Deployment
  19. ONNX 三输出
  20. RDK X5 INT8
  21. 实车闭环
~~~

## 19. 实验路线

### Exp01：最新 LaneRobot baseline

使用最新协议、最新 Validator、最新 offset / decode，从头重新训练。

目的：得到可信比较基线。

### Exp02：动态 N / 功能任务语义

不加 Evidence Head，只完成配置解耦和功能 task 定义。

### Exp03：完整导航结构监督

对遮挡 / 断线 / 稀疏区域继续提供可靠 Geometry GT。

### Exp04：Evidence Head

显式学习 Observed / Inferred。

### Exp05：Navigation Cue Dropout

增强 Missing Region 恢复能力。

### Exp06：缺失比例

测试：

~~~text
0%
10%
30%
50%
70%
~~~

重点看 Missing Region 退化速度。

### Exp07：异质视觉载体

例如：

- 白线 -> 锥桶；
- 白线 + 锥桶；
- 左白线 / 右锥桶；
- 颜色边界与实体线切换。

### Exp08：防幻觉 / 干扰

例如：

- 地砖缝；
- 阴影；
- 胶带；
- 普通物体边缘；
- 线外障碍物；
- 真正不存在导航结构。

### Exp09：RDK X5

比较：

~~~text
PyTorch FP32
ONNX
RDK X5 INT8
~~~

### Exp10：实车闭环

覆盖：

- 完整线；
- 断线；
- 障碍遮挡；
- 稀疏锥桶；
- 白线 / 锥桶切换；
- 左右异质边界。

## 20. Depth 的定位

Depth 暂不进入 TaskNav V1 主视觉网络训练。

系统层可保持：

~~~text
RGB   -> TaskNav -> Guide / Left / Right
Depth -> Temporary Occupancy
~~~

当临时障碍物与 Guide 冲突时：

~~~text
Depth occupancy
      ↓
Guide conflict?
      ↓
Local Guide Correction
      ↓
绕障后恢复 Guide
~~~

Depth 属于后续安全约束与闭环扩展，不属于 TaskNav V1 主模型创新。

## 21. TaskNav V1 完成判据

至少满足：

- 最新 baseline 已从头训练并冻结；
- 默认四任务可以稳定训练和预测；
- 仅修改 runtime/default 配置即可切换任意 N；
- 完整 Geometry GT 支持遮挡 / 断线；
- ignore 机制工作；
- Evidence Head 能区分 Observed / Inferred；
- Missing Region 指标相对 baseline 有可重复改善；
- Cue Dropout 带来稳定收益；
- 可见区域不能明显退化；
- 真正不存在结构时不会无条件补线；
- ONNX 三输出与 PyTorch 一致；
- RDK X5 INT8 保持可用实时性；
- 实车完成目标场景闭环导航。

## 22. 一句话定义

> TaskNav V1 = 动态 N 类 LaneRobotV2Independent + 功能化导航任务表示 + 完整 Geometry 监督 + Evidence 辅助监督 + Navigation Cue Dropout。

其中最重要的原则仍然是：

> **“看不见”不等于“导航结构不存在”。**
