from pathlib import Path
import json
import sys

import cv2
from PIL import Image, ImageDraw
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import image as image_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
SELECTED_DIR = CASE_ROOT / "selected_keyframes"
OUTPUT_DIR = CASE_ROOT / "local_reference" / "detector_trials" / "mediapipe_facemesh"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
BOARDS_DIR = OUTPUT_DIR / "boards"
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


LANDMARK_GROUPS = {
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "philtrum": [1, 2, 4, 5, 19, 94, 164, 0, 37, 267, 61, 291],
    "nose": [1, 2, 4, 5, 6, 19, 45, 48, 98, 115, 168, 195, 197, 275, 278, 327, 344],
    "left_eye": [33, 133, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
    "right_eye": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 466],
    "face_shape": [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 172, 176, 234, 251, 284, 288, 297, 323, 332, 356, 361, 365, 377, 378, 379, 397, 400, 454],
}

PADDING = {
    "mouth": 0.45,
    "philtrum": 0.65,
    "nose": 0.45,
    "left_eye": 0.45,
    "right_eye": 0.45,
    "face_shape": 0.12,
}

COLORS = {
    "mouth": (220, 40, 80),
    "philtrum": (250, 140, 20),
    "nose": (30, 160, 80),
    "left_eye": (40, 100, 240),
    "right_eye": (40, 100, 240),
    "face_shape": (150, 70, 220),
}


def _landmark_box(landmarks, indices, width, height, padding):
    points = []
    for index in indices:
        point = landmarks[index]
        x = int(point.x * width)
        y = int(point.y * height)
        if -width <= x <= width * 2 and -height <= y <= height * 2:
            points.append((x, y))
    if not points:
        return None
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    box_width = max(8, x2 - x1)
    box_height = max(8, y2 - y1)
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def main() -> None:
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_region = {name: [] for name in LANDMARK_GROUPS}
    annotated_rows = []
    frame_rows = []

    base_options = base_options_module.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    with face_landmarker.FaceLandmarker.create_from_options(options) as landmarker:
        for source in sorted(SELECTED_DIR.glob("*.jpg")):
            image_bgr = cv2.imread(str(source))
            if image_bgr is None:
                continue
            height, width = image_bgr.shape[:2]
            task_image = image_module.Image.create_from_file(str(source))
            result = landmarker.detect(task_image)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(image_rgb).convert("RGB")
            draw = ImageDraw.Draw(pil)

            detected = bool(result.face_landmarks)
            regions = []
            if detected:
                landmarks = result.face_landmarks[0]
                for region_name, indices in LANDMARK_GROUPS.items():
                    box = _landmark_box(landmarks, indices, width, height, PADDING[region_name])
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    draw.rectangle(box, outline=COLORS[region_name], width=5)
                    draw.text((x1 + 4, max(0, y1 - 18)), region_name, fill=COLORS[region_name])
                    crop_dir = CROPS_DIR / region_name
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = crop_dir / source.name
                    pil.crop(box).save(crop_path, format="JPEG", quality=95)
                    row = {
                        "source": str(source),
                        "source_file": source.name,
                        "region": region_name,
                        "box": [x1, y1, x2, y2],
                        "crop_path": str(crop_path),
                    }
                    regions.append(row)
                    rows_by_region[region_name].append(row)

            annotated_path = ANNOTATED_DIR / source.name
            pil.save(annotated_path, format="JPEG", quality=94)
            annotated_rows.append({"path": str(annotated_path), "file_name": source.name, "total_score": len(regions)})
            frame_rows.append({"source_file": source.name, "detected": detected, "region_count": len(regions)})

    build_reference_board(
        annotated_rows,
        BOARDS_DIR / "annotated_facemesh.jpg",
        title="MediaPipe Face Mesh Annotated Detections",
        columns=4,
    )
    for region_name, rows in rows_by_region.items():
        if not rows:
            continue
        build_reference_board(
            [{"path": row["crop_path"], "file_name": row["source_file"], "total_score": region_name} for row in rows],
            BOARDS_DIR / f"{region_name}_crops.jpg",
            title=f"MediaPipe {region_name.replace('_', ' ').title()} Crops",
            columns=4,
            thumb_size=(180, 180),
        )

    summary = {
        "method": "MediaPipe Face Mesh on selected keyframes",
        "source_dir": str(SELECTED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "detected_frame_count": sum(1 for row in frame_rows if row["detected"]),
        "frame_count": len(frame_rows),
        "region_counts": {name: len(rows) for name, rows in rows_by_region.items()},
        "frames": frame_rows,
        "assessment": "visual_review_required",
    }
    (OUTPUT_DIR / "mediapipe_facemesh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# MediaPipe Face Mesh Trial",
        "",
        f"- detected_frame_count: {summary['detected_frame_count']}",
        f"- frame_count: {summary['frame_count']}",
        "",
        "## Region Counts",
        "",
    ]
    for name, count in summary["region_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Frames", ""])
    for row in frame_rows:
        lines.append(f"- {row['source_file']}: detected={row['detected']}, region_count={row['region_count']}")
    (OUTPUT_DIR / "mediapipe_facemesh_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "mediapipe_facemesh_summary.md")
    print(BOARDS_DIR / "annotated_facemesh.jpg")


if __name__ == "__main__":
    main()
