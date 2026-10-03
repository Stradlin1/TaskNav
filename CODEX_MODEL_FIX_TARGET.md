# CODEX 修改目标：TaskNav 六项模型正确性修复

> 目标仓库：`Stradlin1/TaskNav`  
> 基线分支：`main`  
> 目标日期：2026-10-03  
> 任务性质：直接修改代码、配置、测试与文档。  
> 重要：不要重构无关模块，不要顺手加入大创后续功能。

---

## 0. 先读这些文件

开始修改前先完整阅读：

~~~text
README.md
TASKNAV_TARGET.md
INDEPENDENT_LANE_README.md
MODIFICATION_REPORT.md

ultralytics/utils/loss.py
ultralytics/nn/modules/head.py
ultralytics/models/yolo/lane/train.py
ultralytics/models/yolo/lane/val.py
ultralytics/models/yolo/lane/plotting.py
ultralytics/cfg/default.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
train.py

tests/test_tasknav_lane_protocol.py
tests/test_lane_validator_metrics.py
~~~

同时参考原始单任务仓库：

~~~text
https://github.com/TarochLee/ULTRALYTICS_LANE_ROBOT
~~~

只用于理解历史逻辑，不允许简单整体回滚。

---

# 1. 不允许改变的协议

这些已经修正确，必须保留。

## 1.1 连续 x

~~~text
target_x = x_norm * (x_grids - 1)
~~~

默认：

~~~text
x_grids = 160
target_x in [0,159]
~~~

## 1.2 nearest-grid 分类目标

~~~text
target_class = round(target_x)
~~~

不要改回 `floor(target_x)`。

## 1.3 signed offset

~~~text
offset_gt = target_x - target_class
~~~

范围约：

~~~text
[-0.5,+0.5]
~~~

不要改成旧的 `target_x - floor(target_x)`。

## 1.4 inference decode

V2 推理保持：

~~~text
pred_x = argmax(cls_logits over x classes) + offset
~~~

不要为了修 lane_loc 而把部署解码改回 soft-argmax。

## 1.5 输出协议

保持：

~~~text
cls    [B, X+1, R, N]
offset [B,   1, R, N]
~~~

默认：

~~~text
X=160
R=56
N=4
~~~

ONNX 仍然是两个输出：

~~~text
cls_logits
offset
~~~

---

# 2. 修改一：修 lane_loc 对 cls logits 不可导

目标文件：

~~~text
ultralytics/utils/loss.py
~~~

当前问题：

~~~text
pred_x = argmax(cls_logits) + offset
lane_loc = SmoothL1(pred_x, target_x)
~~~

hard argmax 不可导，所以 lane_loc 无法向 cls logits 提供几何梯度。

## 2.1 要求增加“训练专用可导位置解码”

建议新增一个私有函数，例如：

~~~python
def _decode_x_train(self, logits, offset=None):
    ...
~~~

行为：

1. 只使用 `0..x_grids-1` 的 x-grid logits；
2. 排除 no-lane class；
3. 对 x-grid logits 做 softmax；
4. 计算完整可导 expectation：

~~~text
soft_x = sum(prob_i * i)
~~~

5. 如果存在 offset：

~~~text
train_x = soft_x + clamp(offset,-0.5,+0.5)
~~~

6. 返回形状：

~~~text
[B,R,L]
~~~

不要在这个函数内调用：

~~~text
argmax
numpy
detach
item
~~~

## 2.2 保留 inference decode

现有 `_decode_x()` 如果用于 hard V2 inference / 语义检查，可以保留 hard argmax 行为。

关键要求：

~~~text
training geometry losses -> differentiable soft position
actual prediction decode -> hard argmax + signed offset
~~~

## 2.3 lane_loc 改用训练专用位置

在 `_compute_single_task_loss()` 内：

~~~text
pred_x_train = _decode_x_train(...)
~~~

然后：

~~~text
loc = SmoothL1(
    pred_x_train[valid] / (x_grids - 1),
    target_x[valid] / (x_grids - 1)
)
~~~

### 必须满足

单独只保留 lane_loc 进行 backward 时：

~~~text
cls logits grad != 0
offset grad != 0
~~~

至少 cls logits 的有效 x classes 上梯度绝对值和必须 > 0。

---

# 3. 修改二：smooth / curvature 改为 GT-relative geometry

目标文件：

~~~text
ultralytics/utils/loss.py
~~~

当前错误思想：

~~~text
d1_pred -> 0
d2_pred -> 0
~~~

这会天然偏向直线。

## 3.1 一阶几何

使用训练专用可导位置 `pred_x_train`。

预测：

~~~text
d1_pred = pred_x_train[:,1:] - pred_x_train[:,:-1]
~~~

GT：

~~~text
d1_gt = target_x[:,1:] - target_x[:,:-1]
~~~

有效 mask：

~~~text
valid_pair = valid[:,1:] & valid[:,:-1]
~~~

Loss：

~~~text
SmoothL1(
  d1_pred[valid_pair] / (x_grids - 1),
  d1_gt[valid_pair]   / (x_grids - 1)
)
~~~

## 3.2 二阶几何

预测：

~~~text
d2_pred =
    pred_x_train[:,2:]
  - 2 * pred_x_train[:,1:-1]
  + pred_x_train[:,:-2]
~~~

GT：

~~~text
d2_gt =
    target_x[:,2:]
  - 2 * target_x[:,1:-1]
  + target_x[:,:-2]
~~~

有效 mask：

~~~text
valid_triplet =
    valid[:,2:]
  & valid[:,1:-1]
  & valid[:,:-2]
~~~

Loss：

~~~text
SmoothL1(
  d2_pred[valid_triplet] / (x_grids - 1),
  d2_gt[valid_triplet]   / (x_grids - 1)
)
~~~

## 3.3 不允许

不要再保留：

~~~text
SmoothL1(d1_pred, zeros)
SmoothL1(d2_pred, zeros)
~~~

作为默认完整 Loss。

---

# 4. 修改三：增加 Loss 单元测试

新增：

~~~text
tests/test_lane_loss_geometry.py
~~~

建议使用 `unittest`，保持仓库现有测试风格。

可以建立最小 Dummy Model / Dummy Head，只提供：

~~~text
model.parameters()
model.model[-1].x_grids
model.model[-1].row_anchors
model.model[-1].num_lanes
model.args
~~~

不要为了测试启动完整训练器。

至少实现以下测试。

## 4.1 lane_loc 能训练 cls

构造：

~~~text
logits requires_grad=True
offset requires_grad=True
valid GT
~~~

仅计算可导 location loss，backward 后断言：

~~~text
logits.grad is not None
logits.grad[:, :x_grids].abs().sum() > 0
~~~

## 4.2 lane_loc 能训练 offset

断言：

~~~text
offset.grad is not None
offset.grad.abs().sum() > 0
~~~

## 4.3 完美几何的一阶 / 二阶 loss 接近 0

构造一条明显弯曲的 GT，例如：

~~~text
10, 12, 15, 19, 24, 30 ...
~~~

让 `pred_x_train == target_x`。

断言：

~~~text
smooth < 1e-6
curv   < 1e-6
~~~

容差可按实际浮点实现合理设置。

## 4.4 直线预测弯曲 GT 时几何 loss 增大

同一弯曲 GT：

~~~text
GT = curve
prediction = constant / linear wrong geometry
~~~

断言：

~~~text
smooth_wrong > smooth_perfect
curv_wrong   > curv_perfect
~~~

## 4.5 invalid Row 不参与差分

人为将中间 Row 设为 no-lane。

确认跨越 invalid Row 的 pair / triplet 不进入几何 loss。

## 4.6 signed offset 协议回归

至少覆盖：

~~~text
target_x = 100.8
target_class = 101
offset_gt = -0.2
~~~

以及：

~~~text
target_x = 100.2
target_class = 100
offset_gt = +0.2
~~~

确保未来不会回滚为 floor-offset。

## 4.7 Loss finite

四任务随机合法输入：

~~~text
loss finite
all six components finite
no NaN
no Inf
~~~

---

# 5. 修改四：增加 CE + Exist + Offset 干净 baseline

不要要求用户每次手改 `default.yaml`。

新增目录：

~~~text
ultralytics/cfg/experiments/
~~~

新增：

~~~text
lane_core_baseline.yaml
~~~

内容至少覆盖：

~~~yaml
lane_ce: 1.0
lane_loc: 0.0
lane_exist: 1.0
lane_smooth: 0.0
lane_curv: 0.0
lane_offset: 3.0
~~~

其他关键参数显式写清楚：

~~~yaml
model: ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
data: ultralytics/cfg/datasets/lane-robot-4tasks.yaml
imgsz: 640
batch: 16
epochs: 120
optimizer: AdamW
seed: 0
~~~

可以复用全局 default，其余没有必要重复全部 Ultralytics 参数。

## 5.1 修改根目录 train.py

当前：

~~~text
python train.py
~~~

继续兼容。

同时增加：

~~~bash
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
~~~

要求：

- `--cfg` 可选；
- 不传时仍使用 `ultralytics/cfg/default.yaml`；
- 使用指定 cfg 构建 model；
- 使用同一个 cfg 传给 `model.train()`；
- 不硬编码具体 model 名称。

## 5.2 core baseline 目的

这是诊断基线，不是最终默认 Loss。

需要文档明确：

~~~text
CE     -> nearest grid
Exist  -> lane / no-lane
Offset -> signed sub-grid residual
~~~

---

# 6. 修改五：8x10 / 10x10 / 16x16 AdaptiveAvgPool 消融

当前：

~~~text
SingleLaneRobotV2Branch
  -> AdaptiveAvgPool2d((feat_h, feat_w))
~~~

默认：

~~~text
8x10
~~~

## 6.1 保留 8x10 控制组

不要直接删除或替换当前：

~~~text
yolo26s-lane-independent.yaml
~~~

它作为 8x10 control。

## 6.2 新增模型 YAML

新增：

~~~text
ultralytics/cfg/models/26/yolo26s-lane-independent-10x10.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent-16x16.yaml
~~~

要求：

- Backbone 完全一致；
- P4/P5 Fusion 完全一致；
- x_grids 完全一致；
- row_anchors 完全一致；
- num_lanes 完全一致；
- reduce_channels 完全一致；
- hidden_dim 完全一致；
- 唯一结构差异是 feat_h / feat_w。

分别：

~~~text
8x10  -> feat_h=8,  feat_w=10
10x10 -> feat_h=10, feat_w=10
16x16 -> feat_h=16, feat_w=16
~~~

确认 FC flatten_dim 自动随之改变：

~~~text
reduce_channels * feat_h * feat_w
~~~

不要硬编码旧 640 flatten dim。

## 6.3 新增完整 Loss 实验配置

新增：

~~~text
ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
ultralytics/cfg/experiments/lane_full_loss_10x10.yaml
ultralytics/cfg/experiments/lane_full_loss_16x16.yaml
~~~

三份配置：

- 数据一致；
- seed 一致；
- epoch 一致；
- batch 一致；
- optimizer 一致；
- Loss weights 一致；
- 只切换 model YAML。

## 6.4 增加最小 Head shape test

在新测试文件或现有合适测试中验证三种模型都能：

~~~text
build
forward
cls shape == [B,161,56,4]
offset shape == [B,1,56,4]
~~~

如果测试完整 YOLO 模型成本过高，可以测试 `SingleLaneRobotV2Branch` 的三个 pool size。

---

# 7. 修改六：让 lane_label_smoothing 真正生效

目标：

~~~text
ultralytics/utils/loss.py
ultralytics/cfg/default.yaml
tests/test_lane_loss_geometry.py
~~~

当前配置：

~~~yaml
lane_label_smoothing: 0.02
~~~

目前是死参数。

## 7.1 LaneRobotLoss 读取参数

在 `__init__`：

~~~python
self.label_smoothing = float(getattr(args, "lane_label_smoothing", 0.0))
~~~

并验证：

~~~text
0.0 <= lane_label_smoothing < 1.0
~~~

非法值直接抛 `ValueError`。

## 7.2 hard-label CE

当：

~~~text
lane_soft_label = False
~~~

使用：

~~~python
F.cross_entropy(
    logits,
    target,
    reduction="mean",
    label_smoothing=self.label_smoothing,
)
~~~

## 7.3 soft-label CE

当：

~~~text
lane_soft_label = True
~~~

visible Row 当前已有 Gaussian soft target。

在 Gaussian target 归一化后，实现：

~~~text
soft_visible =
    (1-eps) * gaussian_target
  + eps * uniform_visible_grid
~~~

要求：

- uniform 只分布在 `0..x_grids-1`；
- visible Row 的 no-lane target 仍为 0；
- 最终 visible target sum=1。

invalid/no-lane Row：

继续使用 no-lane CE，并让 `label_smoothing` 参数真实生效。

## 7.4 测试

至少：

1. `eps=0` 与旧逻辑数值兼容；
2. `eps=0.1` 与 `eps=0` 的 CE 不相同；
3. soft target 归一化；
4. visible target 的 no-lane 概率仍为 0；
5. `eps<0` 和 `eps>=1` 抛 ValueError。

---

# 8. 完整修正版 Loss 的目标结构

最终默认完整 Loss：

~~~text
L_total =
    lambda_ce     * L_ce
  + lambda_loc    * L_loc
  + lambda_exist  * L_exist
  + lambda_smooth * L_d1_gt
  + lambda_curv   * L_d2_gt
  + lambda_offset * L_offset
~~~

其中：

~~~text
L_ce:
  nearest-grid / no-lane classification

L_loc:
  differentiable soft x + signed offset
  vs continuous target_x

L_exist:
  lane-exist logit vs valid mask

L_d1_gt:
  predicted first derivative vs GT first derivative

L_d2_gt:
  predicted second derivative vs GT second derivative

L_offset:
  signed residual vs target_x - round(target_x)
~~~

---

# 9. 文档必须同步

实现完成后更新：

~~~text
README.md
TASKNAV_TARGET.md
INDEPENDENT_LANE_README.md
MODIFICATION_REPORT.md
~~~

其中必须记录实际最终实现，而不是照抄本任务文档的“计划”措辞。

不要重新加入已经废弃的：

~~~text
Evidence Head
动态 N 必须实现
task supervision mask
~~~

作为本轮目标。

---

# 10. 本轮明确禁止的额外改动

除非为六项修改的必要兼容，不要改：

~~~text
dataset label format
row_anchors=56
x_grids=160
strict label protocol
validator metric definitions
fitness definition
ONNX output names
ONNX number of outputs
task names
task semantic strings
RDK code
Depth
Evidence
Visibility Head
Cue Dropout
Missing weighted loss
Transformer
GNN
BEV
temporal model
~~~

特别禁止：

- 不要把比赛和大创模型混在一起训练；
- 不要新增 task supervision mask；
- 不要把四任务改成文本类别学习；
- 不要改变 inference 的 hard argmax + signed offset。

---

# 11. 验证命令

至少执行：

~~~bash
pytest tests/test_tasknav_lane_protocol.py
pytest tests/test_lane_validator_metrics.py
pytest tests/test_lane_loss_geometry.py
~~~

如果时间允许，直接：

~~~bash
pytest tests
~~~

然后执行三类 smoke：

## 11.1 core baseline smoke

~~~bash
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
~~~

实际 smoke 可以临时 CLI 覆盖 epoch，或复制测试配置，但不要永久把正式配置 epochs 改成 1。

## 11.2 full loss 8x10 smoke

~~~bash
python train.py --cfg ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
~~~

## 11.3 10x10 / 16x16 build smoke

至少确保 model build + one forward + one backward 成功。

---

# 12. ONNX 回归门禁

本轮不改输出协议，所以修改后必须继续：

~~~text
cls_logits [B,161,56,4]
offset     [B,1,56,4]
~~~

使用现有：

~~~bash
python export_onnx.py ...
~~~

至少确认：

- export success；
- onnx.checker success；
- ORT session success；
- PyTorch / ORT parity success；
- output names 不变；
- output shape 不变。

对 10x10 / 16x16 模型，如果没有正式 checkpoint，至少确保其 Head 所用算子仍是当前 ONNX 可导出算子；后续训练出 checkpoint 后再做完整 parity。

---

# 13. 需要输出给用户的最终修改报告

Codex 完成后，请在回复中明确列出：

1. 修改了哪些文件；
2. lane_loc 如何做到可导；
3. smooth / curv 新公式；
4. label smoothing 如何作用于 hard / soft CE；
5. 新增了哪些测试；
6. core baseline 配置路径；
7. 8x10 / 10x10 / 16x16 模型与实验配置路径；
8. pytest 结果；
9. smoke run 结果；
10. ONNX parity 结果；
11. 是否存在任何未完成项。

如果某项没有实际运行，不要写“通过”，必须写“未运行”。

---

# 14. 验收标准

本任务只有全部满足以下条件才算完成：

- [ ] inference 仍为 hard argmax + signed offset；
- [ ] offset GT 仍为 target_x - round(target_x)；
- [ ] lane_loc 对 cls logits 可导；
- [ ] lane_loc 梯度测试通过；
- [ ] smooth 对 GT 一阶几何；
- [ ] curv 对 GT 二阶几何；
- [ ] invalid Row mask 正确；
- [ ] core baseline config 存在并可读取；
- [ ] train.py 支持 --cfg 且无参数行为保持兼容；
- [ ] 8x10 control 保留；
- [ ] 10x10 model config 存在；
- [ ] 16x16 model config 存在；
- [ ] 三套 full-loss experiment config 存在；
- [ ] lane_label_smoothing 被 Loss 读取；
- [ ] label smoothing 边界校验存在；
- [ ] soft-label smoothing 测试通过；
- [ ] 原 protocol tests 不回归；
- [ ] 原 validator tests 不回归；
- [ ] 新 lane loss tests 通过；
- [ ] 输出 tensor shape 不变；
- [ ] ONNX 两输出协议不变；
- [ ] 文档同步到实际实现。

---

# 15. 完成后不要立刻做的事情

六项修复完成以后，先重新训练和比较 baseline。

不要在同一个 PR / commit 继续做：

~~~text
Complete Geometry
Visibility Mask
Cue Dropout
Missing weighted loss
RDK X5
~~~

这些进入下一阶段，避免无法判断本轮模型正确性修复到底带来了什么变化。
