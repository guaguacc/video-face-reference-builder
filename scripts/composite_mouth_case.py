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
BASE_FRAME = CASE_ROOT / "selected_keyframes" / "selected_003_t_00039.000.jpg"
OUTPUT_DIR = CASE_ROOT / "candidate_guided_composite" / "mouth_round_01"
PATCH_DIR = OUTPUT_DIR / "patches"
COMPOSITE_DIR = OUTPUT_DIR / "composites"
BOARDS_DIR = OUTPUT_DIR / "boards"

REFERENCE_FILE = "selected_003_t_00039.000.jpg"
SOURCE_FILES = [
    "selected_011_t_00039.000.jpg",
    "selected_006_t_00048.500.jpg",
    "selected_007_t_00048.500.jpg",
    "selected_001_t_00048.000.jpg",
    "selected_005_t_00048.000.jpg",
]
POINT_NAMES = [
    "left_mouth_corner",
    "right_mouth_corner",
    "upper_lip_center",
    "lower_lip_center",
    "philtrum_center",
]


def main() -> None:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    mouth_rows = {
        row["file_name"]: row
        for row in summary["rows"]
        if row["region"] == "mouth" and row["usable_for_alignment"]
    }
    reference = mouth_rows[REFERENCE_FILE]
    ref_points = _points_array(reference)
    ref_box = tuple(reference["box"])
    ref_width = ref_box[2] - ref_box[0]
    ref_height = ref_box[3] - ref_box[1]

    base = cv2.imread(str(BASE_FRAME))
    if base is None:
        raise ValueError(f"Could not read base frame: {BASE_FRAME}")

    rows = []
    for source_file in SOURCE_FILES:
        row = mouth_rows.get(source_file)
        if not row:
            continue
        source_crop = cv2.imread(row["path"])
        if source_crop is None:
            continue
        transform, inliers = cv2.estimateAffinePartial2D(
            _points_array(row),
            ref_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=12.0,
        )
        if transform is None:
            continue
        aligned = cv2.warpAffine(
            source_crop,
            transform,
            (ref_width, ref_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        patch_path = PATCH_DIR / f"{Path(source_file).stem}_aligned_patch.jpg"
        cv2.imwrite(str(patch_path), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        feather = _feather_composite(base, aligned, ref_box)
        feather_path = COMPOSITE_DIR / f"{Path(source_file).stem}_feather.jpg"
        cv2.imwrite(str(feather_path), feather, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        seamless = _seamless_composite(base, aligned, ref_box)
        seamless_path = COMPOSITE_DIR / f"{Path(source_file).stem}_seamless.jpg"
        cv2.imwrite(str(seamless_path), seamless, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        rows.append(
            {
                "source_file": source_file,
                "aligned_patch": str(patch_path),
                "feather_path": str(feather_path),
                "seamless_path": str(seamless_path),
                "inlier_count": inlier_count,
                "transform": transform.tolist(),
            }
        )

    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "path": row["feather_path"],
                "file_name": f"feather {row['source_file']}",
                "total_score": f"inliers={row['inlier_count']}",
            }
        )
        review_rows.append(
            {
                "path": row["seamless_path"],
                "file_name": f"seamless {row['source_file']}",
                "total_score": f"inliers={row['inlier_count']}",
            }
        )
    if review_rows:
        build_reference_board(
            review_rows,
            BOARDS_DIR / "mouth_composite_candidates.jpg",
            title="Mouth Composite Candidates On Selected 003",
            columns=2,
            thumb_size=(260, 420),
        )

    result = {
        "method": "Mouth-only local affine alignment plus feather/seamless composite",
        "base_frame": str(BASE_FRAME),
        "reference_file": REFERENCE_FILE,
        "reference_box": ref_box,
        "source_files": SOURCE_FILES,
        "candidate_count": len(review_rows),
        "review_sheet": str(BOARDS_DIR / "mouth_composite_candidates.jpg"),
        "rows": rows,
        "ai_review": {
            "scope": "mouth region only",
            "do_not_treat_as_final_identity": True,
            "requires_visual_selection_before_next_region": True,
        },
    }
    (OUTPUT_DIR / "mouth_composite_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Mouth Composite Round 01",
        "",
        f"- base_frame: {result['base_frame']}",
        f"- reference_file: {result['reference_file']}",
        f"- review_sheet: {result['review_sheet']}",
        f"- candidate_count: {result['candidate_count']}",
        "",
        "## Notes",
        "",
        "- This is a mouth-only local composite trial.",
        "- It does not reconstruct a full face.",
        "- Select the least distorted candidate before moving to philtrum/nose.",
    ]
    (OUTPUT_DIR / "mouth_composite_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "mouth_composite_summary.md")
    print(BOARDS_DIR / "mouth_composite_candidates.jpg")


def _points_array(row):
    return np.array([row["points"][name] for name in POINT_NAMES], dtype=np.float32)


def _feather_composite(base, patch, box):
    output = base.copy()
    x1, y1, x2, y2 = box
    roi = output[y1:y2, x1:x2].astype(np.float32)
    patch = cv2.resize(patch, (x2 - x1, y2 - y1)).astype(np.float32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    center = ((x2 - x1) // 2, int((y2 - y1) * 0.58))
    axes = (int((x2 - x1) * 0.42), int((y2 - y1) * 0.34))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    mask_3 = mask[:, :, None]
    blended = patch * mask_3 + roi * (1.0 - mask_3)
    output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def _seamless_composite(base, patch, box):
    x1, y1, x2, y2 = box
    patch = cv2.resize(patch, (x2 - x1, y2 - y1))
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    center_local = ((x2 - x1) // 2, int((y2 - y1) * 0.58))
    axes = (int((x2 - x1) * 0.40), int((y2 - y1) * 0.32))
    cv2.ellipse(mask, center_local, axes, 0, 0, 360, 255, -1)
    center = (x1 + center_local[0], y1 + center_local[1])
    return cv2.seamlessClone(patch, base, mask, center, cv2.NORMAL_CLONE)


if __name__ == "__main__":
    main()
