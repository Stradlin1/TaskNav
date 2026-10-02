# TaskNav 四任务 Independent LaneRobotV2 基线

> 更新日期：2026-10-02  
> 当前主线：main  
> 本文对应代码基线：main（2026-10-02，已统一核心 Row Anchor fallback）  
> 当前阶段：先闭环 LaneRobot 四任务基线，再进入 TaskNav Evidence / Completion 方向。

## 1. 当前模型结构

当前默认模型是：

~~~text
yolo26s-lane-independent.yaml
~~~

结构：

~~~text
RGB 640x640
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

四个任务只共享 Backbone / Fusion。进入 prediction head 后，每个任务都有独立的：

~~~text
Conv1x1
AdaptiveAvgPool2d
FC1
cls_fc2
offset_fc
~~~

当前四个 task 之间不共享 prediction branch 参数，也不存在 task 维度上的 Softmax 竞争。

## 2. 当前张量协议

默认配置：

~~~text
imgsz       = 640
x_grids     = 160
classes/row = 161
row_anchors = 56
num_lanes   = 4
~~~

分类状态：

~~~text
0..159 : 横向 grid
160    : no-lane
~~~

训练 / PyTorch 输出：

~~~text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
~~~

当前根目录 export_onnx.py 导出两个独立 ONNX 输出：

~~~text
cls_logits [B, 161, 56, 4]
offset     [B,   1, 56, 4]
~~~

LaneRobotV2Independent 内部仍保留 export=True 的拼接兼容逻辑，但部署时应以 export_onnx.py 的实际输出契约为准。

## 3. 当前 manual 标签协议

当前四任务训练使用：

~~~text
ultralytics/cfg/datasets/lane-robot-4tasks.yaml
~~~

标签文件每个非空行必须严格为：

~~~text
lane_id x0 y0 x1 y1 ... x55 y55
~~~

固定 Row Anchor 定义：

~~~text
row 0  -> y = 1.000000      图像底部
row 1  -> y ≈ 0.987879
...
row 55 -> y = 0.333333      图像高度 1/3 处
~~~

公式：

~~~text
y(row) = 1.0 - row / 55 * (2/3)
~~~

因此 56 个 Row Anchor 覆盖原图的下方 2/3，并且索引顺序是：

~~~text
bottom -> top
row 0  -> row 55
~~~

x 规则：

- x 为相对整张图片宽度的归一化坐标；
- 有效 x 必须位于 [0, 1]；
- x=-1 表示该固定 Row Anchor 没有点；
- 当前 strict manual 协议不接受其他负值；
- 某个 lane_id 整行不出现，表示该 task 在整张图不存在；
- 空标签文件表示全部 task 不存在。

当前 data YAML 已启用：

~~~yaml
strict_labels: true
~~~

同时 default.yaml 中：

~~~yaml
lane_strict_labels: True
~~~

严格模式会校验：

1. 每个非空标签行恰好包含 56 对 x/y；
2. y 必须匹配上述 56 个固定 Anchor；
3. y 顺序必须 bottom-to-top；
4. x 只能是 -1 或 [0,1]；
5. task_id 必须是整数且位于当前 num_lanes 范围；
6. 同一标签文件不能重复 task_id；
7. NaN / Inf 会直接报错；
8. strict 模式下缺失 txt 标签会直接报错。

相关实现：

~~~text
ultralytics/models/yolo/lane/protocol.py
ultralytics/models/yolo/lane/dataset.py
tests/test_tasknav_lane_protocol.py
~~~

## 4. Dataset 与训练语义

Dataset 对 TXT 标签按出现顺序映射：

~~~text
第 0 对 x/y  -> row 0
第 1 对 x/y  -> row 1
...
第 55 对 x/y -> row 55
~~~

训练目标：

~~~text
lane   [B, 56, 4]
lane_x [B, 56, 4]
~~~

其中 lane 是离散 grid class，lane_x 是连续 grid 坐标。

当前 x 映射：

~~~text
target_x = x_norm * (x_grids - 1)
         = x_norm * 159
~~~

离散分类目标取最近 grid：

~~~text
target_class = round(target_x)
~~~

V2 offset 定义为相对最近 grid 的有符号残差：

~~~text
offset_gt = target_x - target_class
~~~

理论范围约为：

~~~text
[-0.5, +0.5]
~~~

当前 V2 解码：

~~~text
pred_x = argmax(class_logits) + offset
~~~

当前 soft-label 也以离散 target_class 为中心，避免 continuous soft-label 与 offset 对同一小数部分重复补偿。

## 5. 当前默认训练配置

default.yaml 当前主要值：

~~~text
task       = lane
model      = yolo26s-lane-independent.yaml
data       = ultralytics/cfg/datasets/lane-robot-4tasks.yaml

epochs     = 120
patience   = 25
batch      = 16
imgsz      = 640

optimizer  = AdamW
pretrained = False
amp        = False
~~~

Lane 参数：

~~~text
lane_x_grids        = 160
lane_row_anchors    = 56
lane_num_lanes      = 4
lane_task_weights   = [1,1,1,1]

lane_y_start        = 1.0
lane_y_end          = 0.3333333333
lane_strict_labels  = True

lane_ce             = 1.0
lane_loc            = 2.0
lane_exist          = 1.0
lane_smooth         = 0.03
lane_offset         = 3.0
lane_curv           = 0.02

lane_soft_label     = True
lane_soft_sigma     = 1.2
lane_softargmax_topk= 5
~~~

标准入口：

~~~bash
python train.py
~~~

train.py 读取：

~~~text
ultralytics/cfg/default.yaml
~~~

当前 data YAML 路径为：

~~~yaml
path: datasets/datasets
train: images/train
val: images/valid
train_labels: labels_corrected/train
val_labels: labels_corrected/valid
~~~

这里的 datasets/datasets 是当前本地数据实际目录结构，不是误写。

## 6. 当前 Validator 定义

当前 Validator 已将“存在性”和“定位精度”拆开。

定位指标：

~~~text
lane_matched_mae
lane_matched_mae_px
Acc@1
Acc@3
Acc@5
~~~

存在性指标：

~~~text
lane_miss_rate
lane_exist_precision
lane_exist_recall
lane_exist_f1
lane_exist_acc
~~~

定义：

~~~text
valid      = GT x >= 0
pred_valid = pred x >= 0
matched    = valid AND pred_valid
~~~

matched MAE 只统计 matched 点，no-lane sentinel -1 不再作为 x 坐标参加 MAE。

Acc@1/3/5 的分母仍然是全部 GT valid 点，因此漏检会自动判为 tolerance failure。

存在性：

~~~text
TP = GT有点，预测有点
FP = GT没点，预测有点
FN = GT有点，预测没点
TN = GT没点，预测没点
~~~

当前 fitness：

~~~text
fitness =
    Acc@3
  + 0.5 * Acc@5
  - 0.003 * matched_MAE
  + 0.05 * Exist_F1
~~~

MAE_px 使用严格 grid-to-pixel 比例：

~~~text
(image_width - 1) / (x_grids - 1)
~~~

相关测试：

~~~text
tests/test_lane_validator_metrics.py
~~~

注意：仓库目前没有 GitHub Actions workflow，因此“测试文件存在”不等于“每次提交都已经自动通过 CI”。

## 7. ONNX 导出

当前 export_onnx.py 已移除旧机器绝对路径作为唯一入口。

默认 checkpoint：

~~~text
runs/lane/train/weights/best.pt
~~~

基本用法：

~~~bash
python export_onnx.py
~~~

指定输入输出：

~~~bash
python export_onnx.py \
  --weights runs/lane/train/weights/best.pt \
  --output runs/lane/train/weights/best.onnx
~~~

主要能力：

- --weights / --output；
- --imgsz；
- --opset；
- --device；
- 单文件 ONNX 默认策略；
- 可选 --external-data；
- 可选 --simplify；
- 自动读取实际 Head 的 x_grids / row_anchors / num_lanes；
- 自动校验 PyTorch 输出 shape；
- onnx.checker 检查；
- 默认执行 ONNX Runtime 数值一致性校验；
- 检查输出名称、shape、NaN / Inf；
- 使用 atol / rtol 比较 PyTorch 与 ONNX Runtime 输出。

默认输出仍为：

~~~text
cls_logits
offset
~~~

TaskNav Evidence Head 尚未实现，因此当前没有第三个 evidence 输出。

## 8. 当前测试

当前新增：

~~~text
tests/test_tasknav_lane_protocol.py
tests/test_lane_validator_metrics.py
~~~

建议在正式训练前执行：

~~~bash
pytest tests/test_tasknav_lane_protocol.py
pytest tests/test_lane_validator_metrics.py
~~~

并继续做短周期 smoke run，确认：

- 数据可完整加载；
- loss finite；
- 四任务 shape 正确；
- Validator 指标正常；
- best.pt / last.pt 正常保存；
- Row Anchor 可视化方向正确；
- ONNX 导出与 ORT parity 通过。

## 9. 当前已知未完成事项

### 9.1 核心 Row Anchor fallback 已统一

核心 Lane 模块的最终 fallback 已统一为正式 manual 协议：

~~~text
y_start = 1.0
y_end   = 0.3333333333
~~~

覆盖：

~~~text
ultralytics/models/yolo/lane/dataset.py
ultralytics/models/yolo/lane/train.py
ultralytics/models/yolo/lane/val.py
ultralytics/models/yolo/lane/plotting.py
~~~

因此标准配置和最终 fallback 现在都遵循同一语义：

~~~text
row 0  = image bottom
row 55 = image height 1/3
order  = bottom-to-top
~~~

### 9.2 通用 lane 默认入口已统一到四任务 baseline

Ultralytics 通用映射现在使用：

~~~text
TASK2DATA["lane"]            = lane-robot-4tasks.yaml
TASK2CALIBRATIONDATA["lane"] = lane-robot-4tasks.yaml
TASK2MODEL["lane"]           = yolo26s-lane-independent.yaml
~~~

旧文件名：

~~~text
ultralytics/cfg/datasets/lane-robot.yaml
~~~

仍保留用于兼容显式引用，但内容已同步为当前四任务 strict manual 配置，不再包含历史绝对路径、单任务 num_lanes=1 或 0.67 -> 1.0 几何。

### 9.3 动态 N 尚未实现

当前任务数仍同时写在：

~~~text
default.yaml
lane-robot-4tasks.yaml
yolo26*-lane-independent.yaml
~~~

当前三处均为 4，因此四任务训练一致。

但“只改 default.yaml 就切换任意 N”的 TaskNav 目标尚未实现。

### 9.4 独立推理辅助脚本仍有历史配置

infer.py / infer_onnx.py 以及部分旧 helper script 仍包含历史机器路径或固定参数。

当前部署基线应优先以：

~~~text
export_onnx.py
模型实际 Head 元数据
正式部署端配置
~~~

为准。

### 9.5 当前任务名称仍是 generic names

data YAML 目前仍为：

~~~text
lane_task_0
lane_task_1
lane_task_2
lane_task_3
~~~

TaskNav 正式功能语义需要后续明确映射到：

~~~text
reference_guide
left_boundary
right_boundary
reserve_3
~~~

## 10. 基线训练原则

由于近期已经修改：

- offset target；
- V2 decode；
- soft-label 中心；
- Validator / fitness；
- strict manual protocol；

历史旧规则下训练出的 checkpoint 不应直接用于判断当前规则精度。

推荐：

~~~text
最新代码
  ↓
协议测试
  ↓
短周期 smoke run
  ↓
从头正式训练
  ↓
冻结可信 baseline
~~~

完成可信 baseline 后，再进入 TaskNav 的 Geometry / Evidence / Cue Dropout 改造。
