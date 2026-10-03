# TaskNav

TaskNav 是基于 YOLO26 + LaneRobotV2 的单目导航结构感知项目。Row-Anchor 基线的六项模型正确性修复已经落地，当前进入从头重训与 Pool 消融阶段；基线冻结后再进入大创研究阶段的完整 Geometry、Visibility Mask、Navigation Cue Dropout、Missing-region weighted loss 与 RDK X5 部署。

> 文档状态：2026-10-03  
> 当前分支：`main`  
> 当前模型：共享 Backbone / P4+P5 Fusion + Independent LaneRobotV2 branches  
> 当前输出协议：`cls [B, X+1, R, N]` + `offset [B, 1, R, N]`

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

其中 offset 是相对最近 grid 的有符号残差，范围约为 `[-0.5, +0.5]`。推理解码继续保持 hard argmax + signed offset，不回退到旧的 floor-offset 定义。

当前 strict manual Row Anchor 协议：

```text
row_anchors = 56
row 0       = y 1.0
row 55      = y 0.3333333333
order       = bottom-to-top
```

## 已完成：模型正确性修复

2026-10-03 已完成以下六项代码修改：

1. `lane_loc` 使用完整 x-grid softmax 期望与 signed offset 的训练专用可导位置；推理仍保持 hard argmax。
2. `lane_smooth` / `lane_curv` 改为预测一阶/二阶几何对齐 GT 一阶/二阶几何。
3. 新增 Loss 梯度、曲线、invalid mask、signed offset、finite 与 Head shape 测试。
4. 新增可复现的 `CE + Exist + Offset` 干净 baseline 配置。
5. 保留 8x10 控制组，并新增 10x10、16x16 AdaptiveAvgPool Head 消融配置。
6. `lane_label_smoothing` 已作用于 hard CE、visible soft target 和 no-lane CE，并校验 `[0,1)`。

训练入口：

```bash
python train.py
python train.py --cfg ultralytics/cfg/experiments/lane_core_baseline.yaml
python train.py --cfg ultralytics/cfg/experiments/lane_full_loss_8x10.yaml
```

当前代码验证：pytest 21 项全部通过；8x10、10x10、16x16 完整模型 build/forward/backward 通过；core 与 full-loss 8x10 均通过 1 图 train + 1 图 val、64×64、1 epoch CPU 微型训练；历史 8x10 `best.pt` 以及随机初始化 10x10 / 16x16 的双输出 ONNX checker 与 ORT parity 均通过。尚未运行全量数据 1–3 epoch smoke、三组正式消融或 120 epoch 从头重训。

详细实现规范见：

- [CODEX_MODEL_FIX_TARGET.md](CODEX_MODEL_FIX_TARGET.md)
- [INDEPENDENT_LANE_README.md](INDEPENDENT_LANE_README.md)
- [TASKNAV_TARGET.md](TASKNAV_TARGET.md)
- [MODIFICATION_REPORT.md](MODIFICATION_REPORT.md)

## 当前明确不做

本轮模型修复没有引入：

- Evidence Head；
- task supervision mask；
- 比赛 / 大创混合训练；
- VLM 文本语义；
- Transformer / GNN / BEV / 时序网络；
- Depth Branch；
- 独立 Completion Module；
- 修改现有 nearest-grid + signed-offset 推理协议。

大创 V1 的 Visibility 只作为标签、训练加权和评测元数据使用，不要求模型预测 Visibility。

## 训练与验证顺序

```text
模型正确性六项修复（已完成）
  -> 单元测试 / 配置级 loss smoke / ONNX parity（已完成）
  -> 1 epoch 微型数据训练 smoke（已完成）
  -> 全量数据 1~3 epoch smoke（待运行）
  -> CE + Exist + Offset core baseline
  -> 完整修正版 loss baseline
  -> Pool 8x10 / 10x10 / 16x16 消融
  -> 冻结可信模型基线
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
ultralytics/utils/loss.py
ultralytics/nn/modules/head.py
ultralytics/cfg/default.yaml
ultralytics/cfg/models/26/yolo26s-lane-independent.yaml
ultralytics/models/yolo/lane/
tests/
export_onnx.py
```

当前历史 checkpoint 只能用于兼容性检查、迁移或 debug。由于 offset、decode、Validator、strict protocol 和本轮 Loss 已发生变化，新的模型精度结论必须来自修复完成后的从头训练。
