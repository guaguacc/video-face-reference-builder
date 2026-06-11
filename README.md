# Video Face Reference Builder

从非配合、近距离、局部遮挡的人脸视频中提取可追溯的人脸参考信息。

第一阶段目标是建立可复现的本地处理管线：

- 读取视频信息
- 抽帧
- 计算基础质量分
- 生成最佳帧参考板
- 输出处理报告

完整需求见 `需求规格.md`，阶段计划见 `实施计划.md`。

## 本地环境

本地项目目录包含 `.venv/`，用于运行当前第一阶段代码。该目录体积较大，不提交到 Git。

运行测试：

```bash
.venv/bin/python -m pytest -q
```

## 运行当前案例

案例视频放在本地 `案例/` 目录中。该目录体积较大，不提交到 Git。

```bash
.venv/bin/python scripts/run_case.py
```

输出目录：

```text
outputs/<run_id>/
  video_info.json
  frames/
  frame_scores.json
  best_frames/
  reference_boards/
  report.md
```

## 第一版边界

当前版本只完成基础证据整理流程：视频信息、抽帧、质量评分、最佳帧、参考板和报告。

尚未实现：

- 人脸关键点检测
- 嘴部/人中/鼻部区域裁切
- 脸型轮廓估计
- 近距离透视/镜头畸变校正
- CodeFormer/GFPGAN 自动调用
- 候选完整脸局部校正
