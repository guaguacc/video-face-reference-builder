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
- CodeFormer/GFPGAN 可选增强对照自动调用
- 候选完整脸局部校正

### Candidate 04 AI 复合记录

当前案例的完整脸候选 04 使用“强特征包驱动”流程：先从局部切片中挑出嘴、鼻、人中、眼、脸型等强参考，拼成 feature pack，再通过 `imagegen` skill 生成完整脸候选。

复现记录和完整 prompt：

```text
docs/candidate_04_feature_driven_flow.md
```

重建强特征包：

```bash
.venv/bin/python scripts/build_strong_feature_reference_pack_case.py
```
