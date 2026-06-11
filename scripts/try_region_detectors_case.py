from pathlib import Path
import json
import sys

import cv2
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
SELECTED_DIR = CASE_ROOT / "selected_keyframes"
OUTPUT_DIR = CASE_ROOT / "local_reference" / "detector_trials" / "opencv_haar"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
BOARDS_DIR = OUTPUT_DIR / "boards"


def _cascade(name: str) -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / name
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"Could not load cascade: {path}")
    return detector


def main() -> None:
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)

    detectors = {
        "frontal_face": _cascade("haarcascade_frontalface_alt2.xml"),
        "profile_face": _cascade("haarcascade_profileface.xml"),
        "eye": _cascade("haarcascade_eye_tree_eyeglasses.xml"),
        "smile": _cascade("haarcascade_smile.xml"),
    }
    colors = {
        "frontal_face": (255, 40, 40),
        "profile_face": (255, 140, 0),
        "eye": (30, 120, 255),
        "smile": (20, 170, 70),
    }

    all_rows = []
    rows_by_detector = {name: [] for name in detectors}
    annotated_rows = []

    for source in sorted(SELECTED_DIR.glob("*.jpg")):
        image_bgr = cv2.imread(str(source))
        if image_bgr is None:
            continue
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pil = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(pil)
        image_rows = []

        for detector_name, detector in detectors.items():
            boxes = detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4 if detector_name != "smile" else 10,
                minSize=(24, 24),
            )
            for index, (x, y, w, h) in enumerate(boxes, start=1):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                draw.rectangle((x1, y1, x2, y2), outline=colors[detector_name], width=5)
                draw.text((x1 + 4, max(0, y1 - 18)), detector_name, fill=colors[detector_name])
                crop_dir = CROPS_DIR / detector_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{source.stem}_{index:02d}.jpg"
                Image.open(source).convert("RGB").crop((x1, y1, x2, y2)).save(crop_path, format="JPEG", quality=94)
                row = {
                    "source": str(source),
                    "source_file": source.name,
                    "detector": detector_name,
                    "box": [x1, y1, x2, y2],
                    "crop_path": str(crop_path),
                }
                image_rows.append(row)
                all_rows.append(row)
                rows_by_detector[detector_name].append(row)

        annotated_path = ANNOTATED_DIR / source.name
        pil.save(annotated_path, format="JPEG", quality=94)
        annotated_rows.append({"path": str(annotated_path), "file_name": source.name, "total_score": len(image_rows)})

    if annotated_rows:
        build_reference_board(
            annotated_rows,
            BOARDS_DIR / "annotated_detections.jpg",
            title="OpenCV Haar Annotated Detections",
            columns=4,
        )

    for detector_name, rows in rows_by_detector.items():
        if not rows:
            continue
        build_reference_board(
            [
                {"path": row["crop_path"], "file_name": row["source_file"], "total_score": detector_name}
                for row in rows
            ],
            BOARDS_DIR / f"{detector_name}_crops.jpg",
            title=f"OpenCV Haar {detector_name} Crops",
            columns=4,
            thumb_size=(180, 180),
        )

    summary = {
        "method": "OpenCV Haar cascades on selected keyframes",
        "source_dir": str(SELECTED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "counts": {name: len(rows) for name, rows in rows_by_detector.items()},
        "rows": all_rows,
        "assessment": "visual_review_required",
    }
    (OUTPUT_DIR / "opencv_haar_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# OpenCV Haar Detector Trial",
        "",
        f"- source_dir: {summary['source_dir']}",
        f"- output_dir: {summary['output_dir']}",
        "",
        "## Counts",
        "",
    ]
    for name, count in summary["counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Assessment", "", "- visual_review_required"])
    (OUTPUT_DIR / "opencv_haar_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "opencv_haar_summary.md")
    print(BOARDS_DIR / "annotated_detections.jpg")


if __name__ == "__main__":
    main()
