# CODEX 修改目标：TaskNav 六项模型正确性修复

> 目标仓库：`Stradlin1/TaskNav`  
> 基线分支：`main`  
> 目标日期：2026-10-03  
> 任务性质：直接修改代码、配置、测试与文档。  
> 重要：不要重构无关模块，不要顺手加入大创后续功能。

---

## 0. 先读这些文件

开始修改前先完整阅读：

```text
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
```

同时参考原始单任务仓库：

```text
https://github.com/TarochLee/ULTRALYTICS_LANE_ROBOT
```

只用于理解历史逻辑，不允许简单整体回滚。

---

# 1. 不允许改变的协议

这些已经修正确，必须保留。

## 1.1 连续 x

```text
target_x = x_norm * (x_grids - 1)
```

默认：

```text
x_grids = 160
target_x in [0,159]
```

## 1.2 nearest-grid 分类目标

```text
target_class = round(target_x)
```

不要改回 `floor(target_x)`。

## 1.3 signed offset

```text
offset_gt = target_x - target_class
```

范围约：

```text
[-0.5,+0.5]
```

不要改成旧的 `target_x - floor(target_x)`。

## 1.4 inference decode

V2 推理保持：

```text
pred_x = argmax(cls_logits over x classes) + offset
```

不要为了修 lane_loc 而把部署解码改回 soft-argmax。

## 1.5 输出协议

保持：

```text
cls    [B, X+1, R, N]
offset [B,   1, R, N]
```

默认：

```text
X=160
R=56
N=4
```

ONNX 仍然是两个输出：

```text
cls_logits
offset
```

---

# 2. 修改一：修 lane_loc 对 cls logits 不可导

目标文件：

```text
ultralytics/utils/loss.py
```

当前问题：

```text
pred_x = argmax(cls_logits) + offset
lane_loc = SmoothL1(pred_x, target_x)
```

hard argmax 不可导，所以 lane_loc 无法向 cls logits 提供几何梯度。

## 2.1 要求增加“训练专用可导位置解码”

建议新增一个私有函数，例如：

```python
def _decode_x_train(self, logits, offset=None):
    ...
```

行为：

1. 只使用 `0..x_grids-1` 的 x-grid logits；
2. 排除 no-lane class；
3. 对 x-grid logits 做 softmax；
4. 计算完整可导 expectation：

```text
soft_x = sum(prob_i * i)
```

5. 如果存在 offset：

```text
train_x = soft_x + clamp(offset,-0.5,+0.5)
```

6. 返回形状：

```text
[B,R,L]
```

不要在这个函数内调用：

```text
argmax
numpy
detach
item
```

## 2.2 保留 inference decode

现有 `_decode_x()` 如果用于 hard V2 inference / 语义检查，可以保留 hard argmax 行为。

关键要求：

```text
training geometry losses -> differentiable soft position
actual prediction decode -> hard argmax + signed offset
```

## 2.3 lane_loc 改用训练专用位置

在 `_compute_single_task_loss()` 内：

```text
pred_x_train = _decode_x_train(...)
```

然后：

```text
loc = SmoothL1(
    pred_x_train[valid] / (x_grids - 1),
    target_x[valid] / (x_grids - 1)
)
```

### 必须满足

单独只保留 lane_loc 进行 backward 时：

```text
cls logits grad != 0
offset grad != 0
```

至少 cls logits 的有效 x classes 上梯度绝对值和必须 > 0。

---

# 3. 修改二：smooth / curvature 改为 GT-relative geometry

目标文件：

```text
ultralytics/utils/loss.py
```

当前错误思想：

```text
d1_pred -> 0
d2_pred -> 0
```

这会天然偏向直线。

## 3.1 一阶几何

使用训练专用可导位置 `pred_x_train`。

预测：

```text
d1_pred = pred_x_train[:,1:] - pred_x_train[:,:-1]
```

GT：

```text
d1_gt = target_x[:,1:] - target_x[:,:-1]
```

有效 mask：

```text
valid_pair = valid[:,1:] & valid[:,:-1]
```

Loss：

```text
SmoothL1(
  d1_pred[valid_pair] / (x_grids - 1),
  d1_gt[valid_pair]   / (x_grids - 1)
)
```

## 3.2 二阶几何

预测：

```text
d2_pred =
    pred_x_train[:,2:]
  - 2 * pred_x_train[:,1:-1]
  + pred_x_train[:,:-2]
```

GT：

```text
d2_gt =
    target_x[:,2:]
  - 2 * target_x[:,1:-1]
  + target_x[:,:-2]
```

有效 mask：

```text
valid_triplet =
    valid[:,2:]
  & valid[:,1:-1]
  & valid[:,:-2]
```

Loss：

```text
SmoothL1(
  d2_pred[valid_triplet] / (x_grids - 1),
  d2_gt[valid_triplet]   / (x_grids - 1)
)
```

## 3.3 不允许

不要再保留：

```text
SmoothL1(d1_pred, zeros)
SmoothL1(d2_pred, zeros)
```

作为默认完整 Loss。

---

# 4. 修改三：增加 Loss 单元测试

新增：

```text
tests/test_lane_loss_geometry.py
```

建议使用 `unittest`，保持仓库现有测试风格。

至少实现以下测试：

1. lane_loc backward 后 `cls logits grad != 0`；
2. lane_loc backward 后 `offset grad != 0`；
3. 完美弯曲 GT 的 smooth / curv 近似 0；
4. 直线预测弯曲 GT 时 smooth / curv 明显增大；
5. invalid/no-lane Row 不参与 pair / triplet 差分；
6. signed offset 协议回归：
   - `100.8 -> class 101 -> offset -0.2`
   - `100.2 -> class 100 -> offset +0.2`
7. 四任务随机合法输入下 total loss 与所有分量均 finite，无 NaN / Inf。

不要为了测试启动完整训练器；优先用最小 Dummy Model / Dummy Head。

---

# 5. 修改四：增加 CE + Exist + Offset 干净 baseline

新增目录：

```text
ultralytics/cfg/experiments/
```

新增：

```text
lane_core_baseline.yaml
```

内容至少覆盖：

```yaml
lane_ce: 1.0
lane_loc: 0.0
lane_exist: 1.0
lane_smooth: 0.0
lane_curv: 0.0
lane_offset: 3.0

model: ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
data: ultralytics/cfg/datasets/lane-robot-4tasks.yaml
imgsz: 640
batch: 16
epochs: 120
optimizer: AdamW
seed: 0
```

## 5.1 修改根目录 train.py

继续兼容：

```bash
python train.py
```

同时增加：

```bash
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
```

要求：

- `--cfg` 可选；
- 不传时仍使用 `ultralytics/cfg/default.yaml`；
- 使用指定 cfg 构建 model；
- 使用同一个 cfg 传给 `model.train()`；
- 不硬编码具体 model 名称。

---

# 6. 修改五：8x10 / 10x10 / 16x16 AdaptiveAvgPool 消融

当前：

```text
SingleLaneRobotV2Branch
  -> AdaptiveAvgPool2d((feat_h, feat_w))
```

默认：

```text
8x10
```

## 6.1 保留 8x10 控制组

不要删除或覆盖：

```text
ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
```

## 6.2 新增模型 YAML

新增：

```text
ultralytics/cfg/models/26/yolo26s-lane-independent-10x10.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent-16x16.yaml
```

要求除了 `feat_h / feat_w` 外完全一致：

```text
8x10  -> feat_h=8,  feat_w=10
10x10 -> feat_h=10, feat_w=10
16x16 -> feat_h=16, feat_w=16
```

确认：

```text
flatten_dim = reduce_channels * feat_h * feat_w
```

自动变化，不硬编码旧 640。

## 6.3 新增完整 Loss 实验配置

新增：

```text
ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
ultralytics/cfg/experiments/lane_full_loss_10x10.yaml
ultralytics/cfg/experiments/lane_full_loss_16x16.yaml
```

要求：

- data 一致；
- seed 一致；
- epochs 一致；
- batch 一致；
- optimizer 一致；
- Loss weights 一致；
- 只切换 model YAML。

并增加最小 shape 测试，至少证明三种 pool size 的 branch 都能 forward，输出 shape 不变。

---

# 7. 修改六：让 lane_label_smoothing 真正生效

目标：

```text
ultralytics/utils/loss.py
ultralytics/cfg/default.yaml
tests/test_lane_loss_geometry.py
```

## 7.1 读取参数

```python
self.label_smoothing = float(getattr(args, "lane_label_smoothing", 0.0))
```

校验：

```text
0.0 <= lane_label_smoothing < 1.0
```

非法值抛 `ValueError`。

## 7.2 hard-label CE

当：

```text
lane_soft_label = False
```

使用：

```python
F.cross_entropy(
    logits,
    target,
    reduction="mean",
    label_smoothing=self.label_smoothing,
)
```

## 7.3 soft-label CE

visible Row 的 Gaussian target 归一化后：

```text
soft_visible =
    (1-eps) * gaussian_target
  + eps * uniform_visible_grid
```

要求：

- uniform 只在 `0..x_grids-1`；
- visible Row 的 no-lane target 始终为 0；
- visible target sum=1；
- invalid/no-lane Row 继续走 no-lane CE，并让 label smoothing 真实生效。

## 7.4 测试

至少覆盖：

- `eps=0` 与旧逻辑兼容；
- `eps=0.1` 与 `eps=0` CE 不同；
- soft target sum=1；
- visible no-lane target=0；
- `eps<0` 或 `eps>=1` 抛 ValueError。

---

# 8. 最终完整 Loss

```text
L_total =
    lambda_ce     * L_ce
  + lambda_loc    * L_loc
  + lambda_exist  * L_exist
  + lambda_smooth * L_d1_gt
  + lambda_curv   * L_d2_gt
  + lambda_offset * L_offset
```

其中：

```text
L_ce      = nearest-grid / no-lane classification
L_loc     = differentiable soft x + signed offset vs target_x
L_exist   = existence
L_d1_gt   = predicted first derivative vs GT first derivative
L_d2_gt   = predicted second derivative vs GT second derivative
L_offset  = signed residual vs target_x - round(target_x)
```

---

# 9. 本轮明确禁止的额外改动

除六项修改必要兼容外，不要改：

```text
dataset label format
row_anchors=56
x_grids=160
strict label protocol
validator metric definitions
fitness definition
ONNX output names
ONNX output count
task names
task semantic strings
RDK code
Depth
Evidence Head
Visibility prediction Head
task supervision mask
Cue Dropout
Missing weighted loss
Transformer
GNN
BEV
temporal model
```

特别禁止：

- 不要把比赛和大创混在一起训练；
- 不要新增 task supervision mask；
- 不要改变 inference 的 hard argmax + signed offset；
- 不要回滚为 floor-offset。

---

# 10. 文档同步

实现完成后同步：

```text
README.md
TASKNAV_TARGET.md
INDEPENDENT_LANE_README.md
MODIFICATION_REPORT.md
```

写实际实现结果，不要保留已经作废的 Evidence Head / 动态 N 作为当前必做目标。

---

# 11. 验证命令

至少执行：

```bash
pytest tests/test_tasknav_lane_protocol.py
pytest tests/test_lane_validator_metrics.py
pytest tests/test_lane_loss_geometry.py
```

条件允许直接：

```bash
pytest tests
```

然后做：

```bash
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
python train.py --cfg ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
```

10x10 / 16x16 至少做 build + one forward + one backward smoke。

---

# 12. ONNX 回归门禁

修改后必须继续保持：

```text
cls_logits [B,161,56,4]
offset     [B,1,56,4]
```

使用现有 `export_onnx.py` 确认：

- export success；
- `onnx.checker` success；
- ORT session success；
- PyTorch / ORT parity success；
- output names 不变；
- output shape 不变。

---

# 13. Codex 完成后必须汇报

必须明确列出：

1. 修改文件清单；
2. lane_loc 如何实现可导；
3. smooth / curv 新公式；
4. label smoothing 如何作用于 hard / soft CE；
5. 新增测试；
6. core baseline 配置路径；
7. 8x10 / 10x10 / 16x16 配置路径；
8. pytest 实际结果；
9. smoke run 实际结果；
10. ONNX parity 实际结果；
11. 未完成项。

没有实际运行的项目必须写“未运行”，不能写“通过”。

---

# 14. 验收清单

- [ ] inference 仍为 hard argmax + signed offset
- [ ] offset GT 仍为 `target_x - round(target_x)`
- [ ] lane_loc 对 cls logits 可导
- [ ] lane_loc gradient test 通过
- [ ] smooth 对 GT 一阶几何
- [ ] curv 对 GT 二阶几何
- [ ] invalid Row mask 正确
- [ ] core baseline config 存在
- [ ] `train.py --cfg` 可用且无参数行为兼容
- [ ] 8x10 control 保留
- [ ] 10x10 model config 存在
- [ ] 16x16 model config 存在
- [ ] 三套 full-loss experiment config 存在
- [ ] lane_label_smoothing 被 Loss 读取
- [ ] label smoothing 边界校验存在
- [ ] soft-label smoothing 测试通过
- [ ] 原 protocol tests 不回归
- [ ] 原 validator tests 不回归
- [ ] 新 lane loss tests 通过
- [ ] 输出 tensor shape 不变
- [ ] ONNX 两输出协议不变
- [ ] 文档同步到实际实现

---

# 15. 完成后先停

六项修复完成后先重新训练 baseline。

不要在同一轮继续做：

```text
Complete Geometry
Visibility Mask
Navigation Cue Dropout
Missing weighted loss
RDK X5
```

这些进入下一阶段，避免无法判断本轮模型正确性修复本身带来的影响。
