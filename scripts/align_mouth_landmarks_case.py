from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
CROPS_ROOT = CASE_ROOT / "local_reference" / "ai_curated_crops"
SUMMARY_PATH = CROPS_ROOT / "ai_curated_crops_summary.json"
OUTPUT_DIR = CASE_ROOT / "local_reference" / "opencv_alignment" / "mouth"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
BOARDS_DIR = OUTPUT_DIR / "boards"

MOUTH_POINTS = [
    "left_mouth_corner",
    "right_mouth_corner",
    "upper_lip_center",
    "lower_lip_center",
    "philtrum_center",
]
REFERENCE_FILE = "selected_003_t_00039.000.jpg"


def main() -> None:
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    mouth_rows = [
        row
        for row in summary["rows"]
        if row["region"] == "mouth" and row["usable_for_alignment"]
    ]
    reference = next(row for row in mouth_rows if row["file_name"] == REFERENCE_FILE)
    ref_image = cv2.imread(reference["path"])
    ref_points = _points_array(reference)
    height, width = ref_image.shape[:2]

    output_rows = []
    alignment_rows = []
    for row in mouth_rows:
        source_image = cv2.imread(row["path"])
        source_points = _points_array(row)
        transform, inliers = cv2.estimateAffinePartial2D(
            source_points,
            ref_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=12.0,
        )
        if transform is None:
            continue
        aligned = cv2.warpAffine(
            source_image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        annotated = _draw_points(aligned, ref_points)
        destination = ALIGNED_DIR / row["file_name"]
        cv2.imwrite(str(destination), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        output_rows.append({"path": str(destination), "file_name": row["file_name"], "total_score": f"inliers={inlier_count}"})
        alignment_rows.append(
            {
                "source_file": row["file_name"],
                "source_crop": row["path"],
                "aligned_path": str(destination),
                "reference_file": REFERENCE_FILE,
                "transform": transform.tolist(),
                "inlier_count": inlier_count,
            }
        )

    if output_rows:
        build_reference_board(
            output_rows,
            BOARDS_DIR / "mouth_aligned_to_selected_003.jpg",
            title="Mouth Crops Aligned To Selected 003",
            columns=4,
            thumb_size=(220, 180),
        )

    result = {
        "method": "cv2.estimateAffinePartial2D mouth landmark alignment",
        "reference_file": REFERENCE_FILE,
        "point_names": MOUTH_POINTS,
        "input_count": len(mouth_rows),
        "aligned_count": len(alignment_rows),
        "board": str(BOARDS_DIR / "mouth_aligned_to_selected_003.jpg"),
        "rows": alignment_rows,
        "assessment": "visual_review_required_before_blending",
    }
    (OUTPUT_DIR / "mouth_alignment_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Mouth Alignment Summary",
        "",
        f"- method: {result['method']}",
        f"- reference_file: {result['reference_file']}",
        f"- input_count: {result['input_count']}",
        f"- aligned_count: {result['aligned_count']}",
        f"- board: {result['board']}",
        f"- assessment: {result['assessment']}",
    ]
    (OUTPUT_DIR / "mouth_alignment_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "mouth_alignment_summary.md")
    print(BOARDS_DIR / "mouth_aligned_to_selected_003.jpg")


def _points_array(row):
    points = row["points"]
    return np.array([points[name] for name in MOUTH_POINTS], dtype=np.float32)


def _draw_points(image_bgr, points):
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    import PIL.ImageDraw

    draw = PIL.ImageDraw.Draw(image)
    for point in points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 230, 0), outline=(0, 0, 0))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
