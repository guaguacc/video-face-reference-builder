import json
from pathlib import Path
from typing import Any, Union

import cv2


PathLike = Union[str, Path]


def read_video_info(video_path: PathLike) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        capture.release()

    duration = frame_count / fps if fps > 0 else 0.0
    codec = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()

    return {
        "path": str(path),
        "file_name": path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "codec": codec,
    }


def write_video_info(info: dict[str, Any], output_path: PathLike) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
