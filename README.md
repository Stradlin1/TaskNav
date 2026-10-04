# TaskNav

TaskNav 是基于 YOLO26 + LaneRobotV2 的单目导航结构感知项目。Row-Anchor 基线的六项模型正确性修复已经完成，当前已完成 3-epoch 全量数据 smoke、ONNX opset 11 导出与 ONNX Runtime parity 验证，下一阶段进入三组正式 baseline 训练与 Pool 消融；基线冻结后再进入大创研究阶段的 Complete Geometry、Visibility Mask、Navigation Cue Dropout、Missing-region weighted loss 与 RDK X5 部署。

> 文档状态：2026-10-04  
> 当前分支：`main`  
> 当前模型：共享 Backbone / P4+P5 Fusion + Independent LaneRobotV2 branches  
> 当前输出协议：`cls [B, X+1, R, N]` + `offset [B, 1, R, N]`  
> 当前正式候选：Core 8x10、Full 8x10、Full 10x10  
> 已舍弃正式候选：Full 16x16（ONNX opset 11 导出链不兼容）

## 可复现训练环境

当前正式训练主线只使用仓库相对路径，不依赖 `/home/xhm`、`/home/baater` 等个人目录。仓库根目录的 `environment.yml` 是当前可工作 Conda 环境的完整快照，而不是最小依赖列表。

当前快照中的主要版本：

- Python 3.10.20
- PyTorch 2.13.0
- torchvision 0.28.0
- pytest 9.1.1
- ONNX 1.22.0
- ONNX Runtime / ONNX Runtime GPU 1.23.2
- NVIDIA / CUDA 运行时相关 pip 包为 13.x 系列

推荐环境：Linux x86_64 + NVIDIA GPU。宿主机需要安装与当前 CUDA runtime 兼容的 NVIDIA Driver。由于 `environment.yml` 是现有环境快照，其中会包含训练主线之外的辅助包。

### 从零安装

```bash
git clone https://github.com/Stradlin1/TaskNav.git
cd TaskNav

conda env create -f environment.yml
conda activate tasknav

# 必须执行：使用本仓库修改后的 Ultralytics，而不是仅依赖 pip 中的官方版本
python -m pip install -e .
```

检查训练环境：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

运行单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
```

当前已验证：21 项测试全部通过。

## 数据集目录

正式训练配置 `ultralytics/cfg/datasets/lane-robot-4tasks.yaml` 使用仓库相对路径。另一台电脑只要把数据集保持为：

```text
TaskNav/
├── datasets/
│   └── datasets/
│       ├── images/
│       │   ├── train/
│       │   └── valid/
│       └── labels_corrected/
│           ├── train/
│           └── valid/
├── train.py
└── ultralytics/
```

对应配置：

```yaml
path: datasets/datasets
train: images/train
val: images/valid
train_labels: labels_corrected/train
val_labels: labels_corrected/valid
```

因此仓库和数据集都具备时，不需要修改任何个人绝对路径即可运行正式训练。

## 当前正式训练

现在只保留三组正式候选：

```text
A. Core Loss + 8x10
B. Full Loss + 8x10
C. Full Loss + 10x10
```

单独运行：

```bash
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
python train.py --cfg ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
python train.py --cfg ultralytics/cfg/experiments/lane_full_loss_10x10.yaml
```

顺序运行，前一组正常结束后自动开始下一组，任意一组失败则停止：

```bash
set -e

for CFG in \
  ultralytics/cfg/experiments/lane_core_baseline.yaml \
  ultralytics/cfg/experiments/lane_full_loss_8x10.yaml \
  ultralytics/cfg/experiments/lane_full_loss_10x10.yaml
do
    python train.py --cfg "${CFG}"
done
```

三组配置目前均为：

```text
epochs: 120
patience: 25
```

因此 120 是最大 epoch 数；若触发 early stopping，会提前结束并进入下一组。

### 16x16 状态

`lane_full_loss_16x16.yaml` 保留为历史消融配置，但不再进入正式训练候选。

原因不是 smoke 精度问题，而是部署约束：当前 Head 的输入特征为 40x40，16x16 AdaptiveAvgPool 在 PyTorch legacy ONNX exporter 的 opset 11 路径中无法导出，因为输出尺寸不是输入尺寸的整数因子。

当前部署候选因此限定为：

```text
40x40 -> 8x10
40x40 -> 10x10
```

两者均保留在正式实验中。

根目录保留当前训练/导出主线入口。历史统计、旧推理、旧预标注、旧可视化等脚本统一归档到 `tasknav/oldscripts/`，不属于当前 baseline 训练依赖。

## 两条训练线

比赛模型和大创研究模型分开训练，不使用 task supervision mask 混训。

- 比赛线可以使用四个独立任务输出；具体 task_id 语义由比赛数据和状态机定义，模型本身不理解名称。
- 大创研究线围绕三个核心导航功能展开：当前参考引导、左边界、右边界。是否在工程代码中保留额外 branch 不作为研究创新点。
- 不把 task 名称文本输入网络；这是普通视觉几何预测模型，不是 VLM。

## 当前已冻结的核心协议

```text
target_x     = x_norm * (x_grids - 1)
target_class = round(target_x)
offset_gt    = target_x - target_class
decode       = argmax(cls) + offset
```

其中 offset 是相对最近 grid 的有符号残差，范围约为 `[-0.5, +0.5]`。推理解码继续保持 hard argmax + signed offset，不回退到旧的 floor-offset 定义，也不使用旧的 TopK soft-argmax + offset 推理解码。

当前 strict manual Row Anchor 协议：

```text
x_grids     = 160
row_anchors = 56
row 0       = y 1.0
row 55      = y 0.3333333333
order       = bottom-to-top
num_lanes   = 4
```

默认模型输入为：

```text
[B, 3, 640, 640]
```

输出为：

```text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
```

## 已完成：模型正确性修复

已完成以下六项代码修改：

1. `lane_loc` 使用完整 x-grid softmax 期望与 signed offset 的训练专用可导位置；推理仍保持 hard argmax。
2. `lane_smooth` / `lane_curv` 改为预测一阶/二阶几何对齐 GT 一阶/二阶几何。
3. 新增 Loss 梯度、曲线、invalid mask、signed offset、finite 与 Head shape 测试。
4. 新增可复现的 `CE + Exist + Offset` 干净 baseline 配置。
5. 保留 8x10 控制组，并新增 10x10 Pool 消融；16x16 只保留历史配置，不再进入正式候选。
6. `lane_label_smoothing` 已作用于 hard CE、visible soft target 和 no-lane CE，并校验 `[0,1)`。

## Smoke 与导出状态

当前已经完成：

- pytest 21 项全部通过；
- Core 8x10、Full 8x10、Full 10x10 的全量数据 3-epoch smoke；
- PT -> ONNX 的输出协议检查；
- ONNX opset 强制固定为 11；
- opset 11 模型通过 ONNX checker；
- PyTorch CPU 与 ONNX Runtime CPU 数值 parity 通过；
- 8x10 / 10x10 作为当前正式部署候选保留；
- 16x16 因 opset 11 AdaptiveAvgPool 导出限制退出正式候选。

`export_onnx.py` 当前固定：

```text
opset = 11
input  = images
output = cls_logits, offset
```

并使用 legacy TorchScript ONNX exporter（`dynamo=False`）直接生成 opset 11，避免新版 exporter 先生成 opset 18 再降版本失败。

导出示例：

```bash
python export_onnx.py \
  --weights runs/lane/<run>/weights/best.pt \
  --output runs/lane/<run>/weights/best.onnx \
  --overwrite
```

导出成功应至少看到：

```text
ONNX opset: 11
ONNX Runtime parity: PASSED
```

## Baseline 选择原则

三组正式训练用于回答两个问题：

```text
Core 8x10 vs Full 8x10
-> corrected Full Loss 是否有效

Full 8x10 vs Full 10x10
-> 提高 Pool 空间分辨率是否带来稳定收益
```

最终模型不能只看单一 MAE，应同时比较：

```text
matched MAE      ↓
matched MAE px   ↓
Acc@1 / @3 / @5 ↑
Miss rate        ↓
Exist F1         ↑
```

同时记录模型参数量、ONNX 大小与后续 RDK X5 BPU latency/FPS。若 8x10 与 10x10 精度接近，优先选择更轻的 8x10；若 10x10 有明确稳定收益且部署开销可接受，再冻结 10x10。

Validator 的正式指标使用 raw hard-argmax + signed-offset 预测，不依赖 polynomial post-smooth。Post-smooth 只作为可视化或后续控制端可选后处理，不能用来替代正式模型指标。

## 当前明确不做

本轮 baseline 没有引入：

- Evidence Head；
- task supervision mask；
- 比赛 / 大创混合训练；
- VLM 文本语义；
- Transformer / GNN / BEV / 时序网络；
- Depth Branch；
- 独立 Completion Module；
- 修改现有 nearest-grid + signed-offset 推理协议。

大创 V1 的 Visibility 只作为标签、训练加权和评测元数据使用，不要求模型预测 Visibility。

## 后续顺序

```text
模型正确性六项修复                         ✅
  -> 单元测试                              ✅
  -> 全量数据 3-epoch smoke               ✅
  -> ONNX opset 11 + ORT parity           ✅
  -> RDK X5 BPU 编译 smoke                待完成
  -> Core 8x10 正式训练                    待完成
  -> Full 8x10 正式训练                    待完成
  -> Full 10x10 正式训练                   待完成
  -> 比较指标 / 参数量 / 部署代价
  -> 冻结可信 baseline
  -> Complete Geometry
  -> Visibility Mask
  -> Navigation Cue Dropout
  -> Missing-region weighted loss
  -> Normal / Missing / Sparse / Mixed / Distractor / Occlusion benchmark
  -> ONNX / RDK X5 INT8
  -> 实车闭环
```

## 常用文件

```text
environment.yml
train.py
export_onnx.py
ultralytics/utils/loss.py
ultralytics/nn/modules/head.py
ultralytics/cfg/default.yaml
ultralytics/cfg/datasets/lane-robot-4tasks.yaml
ultralytics/cfg/experiments/lane_core_baseline.yaml
ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
ultralytics/cfg/experiments/lane_full_loss_10x10.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent-10x10.yaml
ultralytics/models/yolo/lane/
tests/
```

当前历史 checkpoint 只能用于兼容性检查、迁移或 debug。由于 offset、decode、Validator、strict protocol 和本轮 Loss 已发生变化，新的模型精度结论必须来自修复完成后的从头训练。
