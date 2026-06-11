# Video Face Reference Builder

从非配合、近距离、局部遮挡的人脸视频中提取可追溯的人脸参考信息。

第一阶段目标是建立可复现的本地处理管线：

- 读取视频信息
- 抽帧
- 计算基础质量分
- 生成最佳帧参考板
- 输出处理报告

完整需求见 `需求规格.md`，阶段计划见 `实施计划.md`。

## 本地测试

当前开发环境复用 CodeFormer 虚拟环境：

```bash
PYTHONPATH=src /Users/guagua/Downloads/CodeFormer/.venv/bin/python -m pytest -q
```
