# Legacy scripts

这里保存 TaskNav / LaneRobot 早期阶段遗留的顶层脚本。

这些文件不属于当前 baseline 训练依赖，可能保留旧 checkpoint 路径、旧数据目录、摄像头编号或历史实现假设，仅用于回溯和参考。

当前训练入口：

```bash
python train.py --cfg ultralytics/cfg/experiments/<experiment>.yaml
```

当前 ONNX 导出入口仍为仓库根目录的 `export_onnx.py`。

协议测试所依赖的规范标注脚本仍保留在：

```text
scripts/manual_fix_56anchors_v11_class1_789_update.py
```
