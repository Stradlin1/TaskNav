# TaskNav 目标设计文档

> 状态：设计目标，尚未开始 TaskNav 功能改造  
> 基线：`TaskNav/main` 初始提交完整复制自 `yolo26_lane_robotv2_ufld:lmm` 当前代码快照  
> 原则：尽可能复用 LaneRobotV2Independent，不新增独立的 Completion Module，不重做 Backbone / Fusion / Row-Anchor 主体。

---

## 1. 项目目标

TaskNav 面向移动机器人在真实赛道中遇到的 **断线、遮挡、稀疏锥桶、颜色边界以及不同视觉载体混合** 场景。

目标不是识别“白线 / 黄线 / 锥桶分别是什么”，而是直接恢复对机器人控制有意义的连续导航结构：

- `reference_guide`：参考引导结构；
- `left_boundary`：左侧不可越界边界；
- `right_boundary`：右侧不可越界边界。

核心问题：

> 当局部视觉证据缺失时，模型仍需要知道导航结构在哪里；同时还必须知道该位置是“直接观察到的”，还是“在缺失 / 遮挡条件下恢复得到的”，避免把障碍物本身错误学习成导航线特征。

因此 TaskNav 需要同时学习两类信息：

1. **Geometry：导航结构在哪里。**
2. **Evidence：该结构当前位置是否存在直接视觉证据。**

---

## 2. 当前 LaneRobot 基线

当前仓库初始代码来自 LMM 分支，主要结构为：

~~~text
RGB
 ↓
YOLO Backbone
 ↓
P4 / P5 Fusion
 ↓
LaneRobotV2Independent
 ├── task branch 0
 ├── task branch 1
 ├── task branch 2
 └── task branch 3
~~~

当前协议：

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

每个任务分支内部：

~~~text
input feature
  ↓
Conv1x1
  ↓
AdaptiveAvgPool2d
  ↓
Flatten
  ↓
Linear -> 512
  ↓
ReLU
  ├── cls_fc2
  └── offset_fc
~~~

TaskNav 将继续使用这套结构作为工程基础。

---

## 3. TaskNav 最终网络目标

### 3.1 不做的事情

TaskNav V1 暂时不引入：

- 独立 Completion Module；
- Transformer；
- GNN；
- 额外跨 Row 大网络；
- BEV 网络；
- 时序网络；
- RGB + Depth 联合训练。

也不重新设计 YOLO Backbone、P4/P5 Fusion 或 Row-Anchor 表示。

### 3.2 要做的最小网络修改

在每个独立 LaneRobotV2 分支已有的 512 维共享特征后增加一个很小的 `Evidence Head`：

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

单分支新增：

~~~text
evidence_fc = Linear(512, 56)
~~~

TaskNav 目标输出：

~~~text
cls      [B, 161, 56, 3]
offset   [B,   1, 56, 3]
evidence [B,   1, 56, 3]
~~~

其中：

- `cls + offset` 回答“导航结构在哪里”；
- `no-lane` 回答“导航结构是否存在”；
- `evidence` 回答“当前位置是否存在直接视觉证据”。

Evidence Head 输出 logits，训练时使用 `BCEWithLogitsLoss`，推理时再做 sigmoid。

---

## 4. 导航任务从 4 个通用槽位统一为 3 个功能槽位

当前 LMM 是 4 个独立 task。TaskNav 正式数据协议目标改为：

~~~text
task 0 -> reference_guide
task 1 -> left_boundary
task 2 -> right_boundary
~~~

即：

~~~text
num_lanes = 3
~~~

说明：

- 初始 LMM 四任务代码保留作为工程来源和回归基线；
- TaskNav 正式模型使用相同的 Independent Branch 思路，但只实例化 3 个导航功能分支；
- 白线、色块边界、锥桶等不再按“物体类型”决定 task，而按它们对机器人导航的作用归入 G / L / R。

---

## 5. 最重要的数据定义

TaskNav 必须把下面三种情况严格区分。

### A. 结构存在，并且直接可见

~~~text
geometry x = 有效位置
evidence   = 1
~~~

例如完整白线、清晰颜色边界、锥桶直接提供边界证据的位置。

### B. 结构存在，但没有直接视觉证据

~~~text
geometry x = 有效位置
evidence   = 0
~~~

例如：

- 白线被障碍物遮挡；
- 白线中间断开；
- 两个稀疏锥桶之间；
- 局部视觉载体消失，但前后关系足以确定导航结构。

这是 TaskNav 最核心的训练区域。

### C. 导航结构真的不存在

~~~text
geometry x = -1
evidence   = -1
~~~

此时继续使用 LaneRobot 的 `no-lane` 类进行训练。

### D. 人工也无法可靠确定的区域

建议新增明确的 ignore 状态：

~~~text
geometry x = -2
evidence   = -2
~~~

这种位置不参与位置、存在性或 Evidence Loss。

目的：避免在大面积遮挡、结构走向本身不确定时人为编造 GT。

---

## 6. 标签文件设计

为了尽量不破坏当前 LaneRobot 标签链路，Geometry 与 Evidence 分开保存。

### 6.1 Geometry 标签

继续使用：

~~~text
labels/train/000123.txt
~~~

格式继续保持：

~~~text
task_id x1 y1 x2 y2 ... x56 y56
~~~

但语义调整为：

~~~text
x >= 0 : 导航结构存在，并给出完整几何 GT
x = -1 : 导航结构不存在
x = -2 : ignore，不参与训练
~~~

关键变化：

> “看不见”不再自动写成 -1。

如果导航结构仍然存在，即使被遮挡，也需要继续给出正确的连续 `x_gt`。

### 6.2 Evidence 标签

新增：

~~~text
evidence/train/000123.txt
~~~

格式：

~~~text
task_id e1 e2 e3 ... e56
~~~

取值：

~~~text
 1 : 当前 Row 有直接视觉证据
 0 : 导航结构存在，但当前 Row 没有直接视觉证据
-1 : 导航结构不存在，Evidence 不适用
-2 : ignore / 无法可靠标注
~~~

示例：

~~~text
Geometry:
right_boundary:
40 42 44 46 48 50 52 54

Evidence:
1  1  1  0  0  0  1  1
~~~

表示中间三个点的导航结构仍然存在，但被遮挡或缺失。

---

## 7. Dataset 需要修改的内容

目标文件：

~~~text
ultralytics/models/yolo/lane/dataset.py
~~~

需要增加：

1. 读取对应 `evidence/*.txt`；
2. 将 Evidence 转成 `[56, num_lanes]`；
3. 区分 `x=-1` 与 `x=-2`；
4. 输出 `lane_ignore` 或等价 mask；
5. batch 中新增 `lane_evidence`；
6. collate 时同步堆叠 Evidence；
7. 保证 Geometry 与 Evidence 在所有几何增强下同步变化。

目标 batch 至少包含：

~~~text
img
lane
lane_x
lane_y
lane_evidence
lane_ignore
~~~

---

## 8. Head 需要修改的内容

目标文件：

~~~text
ultralytics/nn/modules/head.py
~~~

主要修改：

### SingleLaneRobotV2Branch

当前：

~~~text
cls_fc2
offset_fc
~~~

增加：

~~~text
evidence_fc
~~~

forward 从：

~~~text
return cls, offset
~~~

改为：

~~~text
return cls, offset, evidence
~~~

### LaneRobotV2Independent

对各任务的 Evidence 输出进行 stack。

训练 / PyTorch 输出目标：

~~~python
{
    "cls": cls,
    "offset": offset,
    "evidence": evidence,
}
~~~

---

## 9. Loss 需要修改的内容

目标文件：

~~~text
ultralytics/utils/loss.py
~~~

当前 LaneRobot Loss 保留：

~~~text
lane_ce
lane_loc
lane_exist
lane_smooth
lane_curv
lane_offset
~~~

新增：

~~~text
lane_evidence
~~~

总损失：

~~~text
L_total =
L_lane_original
+
lambda_evidence * L_evidence
~~~

第一版 Evidence Loss 使用：

~~~text
BCEWithLogitsLoss
~~~

### Loss Mask 规则

#### Evidence = 1

- 位置监督：计算；
- offset：计算；
- cls：计算；
- evidence：目标 1。

#### Evidence = 0

- 位置监督：仍然计算；
- offset：仍然计算；
- cls：仍然计算；
- evidence：目标 0。

这是 TaskNav 的核心：

> 虽然这里没有直接视觉证据，但仍然要求模型恢复正确的导航几何。

#### Geometry = -1

- 训练 `no-lane`；
- 不计算位置 / offset；
- Evidence Loss 不计算。

#### Geometry = -2

- 所有 TaskNav 相关 Loss 均 ignore。

---

## 10. Navigation Cue Dropout

在 Dataset / Augmentation 中新增专门的数据增强。

目的：

> 主动制造“视觉证据缺失，但导航结构 GT 不变”的训练样本。

可模拟：

- 连续白线局部擦除；
- 颜色边界局部弱化；
- 随机移除部分锥桶；
- 随机遮挡；
- 局部模糊；
- 使用附近地面纹理覆盖导航线索。

Cue Dropout 发生后：

~~~text
geometry GT  -> 不变
evidence     -> 对应区域置 0
~~~

### 防止 shortcut learning

不能永远使用一种黑色矩形遮挡。

遮挡形式应多样化，并且障碍物 / 遮挡物也必须出现在“没有导航结构”的区域中。

目标是避免模型学习：

~~~text
某种障碍物外观
  ->
这里一定有导航线
~~~

而应迫使模型依赖整体导航关系。

---

## 11. Validator 与核心指标

目标文件：

~~~text
ultralytics/models/yolo/lane/val.py
~~~

TaskNav 不再只报告一个总体误差。

至少新增：

### Geometry

~~~text
Overall MAE
Visible MAE
Missing MAE
Accuracy@5px
Accuracy@10px
~~~

其中：

~~~text
Visible Region = evidence == 1
Missing Region = evidence == 0
~~~

`Missing MAE` 是 TaskNav 的核心指标。

### Structure existence

继续评估：

~~~text
Precision
Recall
F1
False Positive Rate
~~~

用于判断模型会不会在真正没有导航结构时乱补。

### Evidence

在结构存在的位置统计：

~~~text
Evidence Accuracy
Evidence Precision
Evidence Recall
Evidence F1
~~~

---

## 12. Predictor / 可视化

目标文件：

~~~text
ultralytics/models/yolo/lane/predict.py
ultralytics/models/yolo/lane/plotting.py
infer.py
infer_onnx.py
~~~

推理后一个 Row 应能区分三种结果：

### Observed

~~~text
P_exist 高
P_evidence 高
~~~

含义：导航结构存在，而且有直接视觉证据。

### Inferred / Completed

~~~text
P_exist 高
P_evidence 低
~~~

含义：导航结构存在，但这里是模型恢复出的结构。

### Absent

~~~text
P_exist 低
~~~

含义：导航结构不存在。

调试可视化应明确区分 Observed 与 Inferred 点，便于检查模型到底是在“看见”还是“补全”。

---

## 13. ONNX / RDK X5 导出协议

目标文件：

~~~text
export_onnx.py
infer_onnx.py
~~~

目前 LMM ONNX 输出：

~~~text
cls_logits
offset
~~~

TaskNav V1 目标增加第三个输出：

~~~text
cls_logits      [B, 161, 56, 3]
offset          [B,   1, 56, 3]
evidence_logits [B,   1, 56, 3]
~~~

第一版优先保持三个独立输出，不急于合并。

后续再针对 RDK X5 INT8 做：

- 算子兼容检查；
- BPU / CPU 落点检查；
- FP32 / ONNX / INT8 一致性；
- FPS；
- P50 / P95 latency；
- Visible / Missing 精度变化。

Evidence Head 很小，不应显著改变实时性。

---

## 14. 当前基线必须先修复的问题

LMM 当前代码已经记录一个 Row Anchor 协议不一致：

训练数据配置：

~~~text
y_start = 0.333
y_end   = 1.0
~~~

而当前：

~~~text
infer.py
infer_onnx.py
~~~

存在：

~~~text
Y_START = 0.67
Y_END   = 1.0
~~~

在开始 TaskNav 正式实验之前，必须统一训练、验证、PyTorch 推理、ONNX 推理使用完全一致的 Row Anchor 定义。

该修复属于基线正确性修复，不属于 TaskNav 创新。

---

## 15. 配置文件目标

建议新建 TaskNav 自己的配置，而不是继续覆盖 LMM 文件。

建议：

~~~text
ultralytics/cfg/datasets/tasknav.yaml
ultralytics/cfg/models/26/yolo26s-tasknav.yaml
~~~

数据配置目标：

~~~yaml
x_grids: 160
row_anchors: 56
num_lanes: 3

names:
  0: reference_guide
  1: left_boundary
  2: right_boundary
~~~

同时在 `default.yaml` 增加：

~~~text
lane_evidence
tasknav_cue_dropout
tasknav_cue_dropout_prob
~~~

具体权重与概率通过实验确定，不在目标文档中提前写死。

---

## 16. 实验路线

### Exp01：LMM 原始 LaneRobot 回归基线

目的：

- 确认初始导入代码可正常训练 / 推理；
- 固化基线；
- 修复 Row Anchor 不一致后重新验证。

### Exp02：G / L / R Unified LaneRobot

模型仍是 Independent LaneRobot，不加 Evidence Head，不加 Cue Dropout。

目的：

- 建立 TaskNav 公平基线；
- 验证功能标签本身的效果。

### Exp03：完整导航结构监督

对遮挡 / 断线 / 稀疏区域继续提供完整 Geometry GT。

目的：

- 验证不修改主体网络时，LaneRobot 是否已经具备隐式结构恢复能力。

### Exp04：Evidence Head

加入 Evidence 辅助监督。

目的：

- 让模型显式区分 Observed 与 Inferred；
- 验证是否减少把遮挡物当作导航结构证据的错误。

### Exp05：Navigation Cue Dropout

在 Exp04 基础上加入专门缺失增强。

目的：

- 提高缺失 / 遮挡 / 稀疏条件下的恢复能力。

### Exp06：缺失比例实验

~~~text
0%
10%
30%
50%
70%
~~~

重点观察 Missing MAE 的退化速度。

### Exp07：异质视觉载体

测试：

- 白线 -> 锥桶；
- 白线 -> 白线 + 锥桶 -> 锥桶；
- 左白线 / 右锥桶；
- 颜色边界与实体线切换。

### Exp08：防幻觉 / 干扰实验

测试：

- 地砖缝；
- 阴影；
- 胶带；
- 普通物体边缘；
- 障碍物在线外；
- 真正不存在导航结构的区域。

重点统计 False Positive / False Completion。

### Exp09：RDK X5

比较：

~~~text
PyTorch FP32
ONNX
RDK X5 INT8
~~~

### Exp10：实车闭环

场景：

- 完整线；
- 断线；
- 障碍遮挡；
- 稀疏锥桶；
- 白线 / 锥桶切换；
- 左右异质边界。

---

## 17. Depth 的定位

Depth 不进入 TaskNav V1 的视觉网络训练。

后续系统层：

~~~text
RGB -> TaskNav -> Guide / Left / Right
Depth -> Temporary Occupancy
~~~

只有当临时障碍物与 Guide 冲突时：

~~~text
Depth occupancy
      ↓
Guide conflict?
      ↓ yes
Local Guide Correction
      ↓
绕障后恢复原 Guide
~~~

Depth 是安全约束和闭环系统扩展，不是 TaskNav 导航结构感知主模型的一部分。

---

## 18. 推荐代码修改顺序

1. **先保证当前 LMM baseline 可复现。**
2. **修复 Row Anchor 训练 / 推理不一致。**
3. **建立 G / L / R 三任务数据协议。**
4. **修改 Dataset 支持完整 Geometry GT + Evidence 标签 + ignore。**
5. **建立 Unified LaneRobot baseline。**
6. **增加 Evidence Head。**
7. **增加 Evidence Loss。**
8. **修改 Validator，加入 Visible / Missing / Evidence 指标。**
9. **增加 Navigation Cue Dropout。**
10. **修改 Predictor / Plotting。**
11. **修改 ONNX 导出与推理。**
12. **完成 RDK X5 量化和实车验证。**

必须按照这个顺序推进，避免同时修改数据、网络、损失和部署链路后无法定位问题。

---

## 19. TaskNav V1 完成判据

TaskNav V1 至少满足：

- G / L / R 三类导航功能结构可以正常训练和预测；
- 被遮挡区域仍输出连续正确的 Geometry；
- Evidence Head 能区分直接观测点和恢复点；
- 真正不存在结构时能够输出 no-lane，而不是无条件补线；
- Missing Region 指标明显优于 Unified LaneRobot baseline；
- Navigation Cue Dropout 带来可重复的 Missing Region 改善；
- 完整可见区域性能不能出现明显退化；
- ONNX 输出与 PyTorch 一致；
- RDK X5 INT8 保持可用实时性；
- 最终实车在断线、遮挡、稀疏锥桶及异质载体场景中完成闭环导航。

---

## 20. 一句话定义 TaskNav V1

> **TaskNav V1 = LaneRobotV2Independent + G/L/R 导航功能表示 + 完整导航结构监督 + Evidence 辅助监督 + Navigation Cue Dropout。**

其中最重要的任务定义是：

> **“看不见”不等于“导航结构不存在”。**

模型不仅要恢复导航结构的位置，还要知道该位置是否具有直接视觉证据。
