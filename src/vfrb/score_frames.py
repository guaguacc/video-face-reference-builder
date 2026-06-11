import json
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def score_frame(image_path: PathLike) -> dict[str, Any]:
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    brightness_penalty = abs(brightness - 128.0) / 128.0
    brightness_score = max(0.0, 1.0 - brightness_penalty)
    sharpness_score = min(sharpness / 1000.0, 1.0)

    total_score = round((sharpness_score * 0.7 + brightness_score * 0.3) * 100.0, 4)

    return {
        "path": str(path),
        "file_name": path.name,
        "sharpness": round(sharpness, 4),
        "brightness": round(brightness, 4),
        "brightness_score": round(brightness_score, 4),
        "edge_distortion_risk": "unknown",
        "perspective_risk": "unknown",
        "total_score": total_score,
    }


def score_frames(frame_dir: PathLike) -> list[dict[str, Any]]:
    directory = Path(frame_dir)
    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return [score_frame(path) for path in image_paths]


def select_best_frames(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["total_score"], reverse=True)[:limit]


def write_scores(rows: list[dict[str, Any]], output_path: PathLike) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
