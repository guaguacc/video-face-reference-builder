from pathlib import Path
import json
import sys

import cv2
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
SELECTED_DIR = CASE_ROOT / "selected_keyframes"
OUTPUT_DIR = CASE_ROOT / "local_reference" / "detector_trials" / "insightface"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
BOARDS_DIR = OUTPUT_DIR / "boards"
MODEL_ROOT = PROJECT_ROOT / "models" / "insightface"


def _clip_box(box, width, height, padding=0.15):
    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _landmark_box(points, width, height, indices, padding=0.65):
    selected = [points[index] for index in indices if index < len(points)]
    if not selected:
        return None
    x_values = [int(point[0]) for point in selected]
    y_values = [int(point[1]) for point in selected]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    box_width = max(10, x2 - x1)
    box_height = max(10, y2 - y1)
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
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", root=str(MODEL_ROOT), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    annotated_rows = []
    face_rows = []
    landmark_rows = {"eyes": [], "nose": [], "mouth_proxy": []}

    for source in sorted(SELECTED_DIR.glob("*.jpg")):
        image_bgr = cv2.imread(str(source))
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        faces = app.get(image_bgr)
        pil = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(pil)

        if faces:
            face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            face_box = _clip_box(face.bbox, width, height)
            draw.rectangle(face_box, outline=(255, 60, 60), width=5)
            face_dir = CROPS_DIR / "face"
            face_dir.mkdir(parents=True, exist_ok=True)
            face_crop = face_dir / source.name
            pil.crop(face_box).save(face_crop, format="JPEG", quality=95)
            face_rows.append({"path": str(face_crop), "file_name": source.name, "total_score": "face"})

            if hasattr(face, "kps") and face.kps is not None:
                points = face.kps
                for point in points:
                    x, y = int(point[0]), int(point[1])
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(40, 120, 255))
                proxy_specs = {
                    "eyes": ([0, 1], 0.80),
                    "nose": ([2], 2.80),
                    "mouth_proxy": ([3, 4], 0.90),
                }
                for region, (indices, padding) in proxy_specs.items():
                    box = _landmark_box(points, width, height, indices, padding=padding)
                    if box is None:
                        continue
                    draw.rectangle(box, outline=(40, 180, 80), width=4)
                    region_dir = CROPS_DIR / region
                    region_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = region_dir / source.name
                    pil.crop(box).save(crop_path, format="JPEG", quality=95)
                    landmark_rows[region].append({"path": str(crop_path), "file_name": source.name, "total_score": region})

        annotated_path = ANNOTATED_DIR / source.name
        pil.save(annotated_path, format="JPEG", quality=94)
        annotated_rows.append({"path": str(annotated_path), "file_name": source.name, "total_score": len(faces)})

    build_reference_board(
        annotated_rows,
        BOARDS_DIR / "annotated_insightface.jpg",
        title="InsightFace Annotated Detections",
        columns=4,
    )
    if face_rows:
        build_reference_board(face_rows, BOARDS_DIR / "face_crops.jpg", title="InsightFace Face Crops", columns=4)
    for region, rows in landmark_rows.items():
        if rows:
            build_reference_board(rows, BOARDS_DIR / f"{region}_crops.jpg", title=f"InsightFace {region} Crops", columns=4)

    summary = {
        "method": "InsightFace FaceAnalysis buffalo_l on selected keyframes",
        "source_dir": str(SELECTED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "detected_frame_count": sum(1 for row in annotated_rows if row["total_score"] > 0),
        "frame_count": len(annotated_rows),
        "face_crop_count": len(face_rows),
        "landmark_counts": {region: len(rows) for region, rows in landmark_rows.items()},
        "assessment": "visual_review_required",
    }
    (OUTPUT_DIR / "insightface_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# InsightFace Trial",
        "",
        f"- detected_frame_count: {summary['detected_frame_count']}",
        f"- frame_count: {summary['frame_count']}",
        f"- face_crop_count: {summary['face_crop_count']}",
        "",
        "## Landmark Counts",
        "",
    ]
    for region, count in summary["landmark_counts"].items():
        lines.append(f"- {region}: {count}")
    (OUTPUT_DIR / "insightface_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "insightface_summary.md")
    print(BOARDS_DIR / "annotated_insightface.jpg")


if __name__ == "__main__":
    main()
