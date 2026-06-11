from pathlib import Path
import json
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
SELECTED_DIR = CASE_ROOT / "selected_keyframes"
OUTPUT_DIR = CASE_ROOT / "local_reference" / "ai_curated_crops"
BOARDS_DIR = OUTPUT_DIR / "boards"


# Pixel boxes are x1, y1, x2, y2 on the original 592x1280 keyframes.
# Landmark points are in crop-local coordinates. These are AI-curated from
# the selected-keyframes sheet and original-frame spot checks.
CROP_MANIFEST = {
    "selected_001_t_00048.000.jpg": {
        "nose": {"box": (0, 430, 430, 940), "points": {"nose_tip": (188, 355), "left_nostril": (75, 338), "right_nostril": (358, 320), "nose_bridge": (178, 92)}, "usable": True},
        "philtrum": {"box": (0, 700, 430, 1120), "points": {"philtrum_center": (194, 242), "upper_lip_center": (210, 330), "nose_base": (190, 150)}, "usable": True},
        "mouth": {"box": (0, 830, 390, 1270), "points": {"left_mouth_corner": (30, 278), "right_mouth_corner": (302, 222), "upper_lip_center": (165, 190), "lower_lip_center": (180, 310), "philtrum_center": (175, 94)}, "usable": True},
        "cheek_skin": {"box": (230, 470, 592, 1020), "points": {}, "usable": True},
    },
    "selected_002_t_00047.500.jpg": {
        "eye": {"box": (0, 300, 280, 610), "points": {"inner_eye_corner": (250, 160), "outer_eye_corner": (42, 154), "upper_lid_center": (145, 128), "lower_lid_center": (145, 184)}, "usable": True},
        "nose": {"box": (80, 500, 480, 990), "points": {"nose_tip": (242, 292), "left_nostril": (106, 268), "right_nostril": (370, 335), "nose_bridge": (245, 40)}, "usable": True},
        "philtrum": {"box": (70, 730, 455, 1160), "points": {"philtrum_center": (230, 180), "upper_lip_center": (258, 325), "nose_base": (214, 90)}, "usable": True},
        "mouth": {"box": (100, 920, 465, 1280), "points": {"left_mouth_corner": (72, 300), "right_mouth_corner": (320, 152), "upper_lip_center": (218, 210), "lower_lip_center": (246, 304), "philtrum_center": (215, 72)}, "usable": False, "warning": "right side partly covered by overlay"},
        "cheek_skin": {"box": (250, 410, 592, 950), "points": {}, "usable": True},
    },
    "selected_003_t_00039.000.jpg": {
        "eye": (50, 160, 420, 520),
        "nose": (0, 470, 330, 940),
        "philtrum": (0, 700, 350, 1030),
        "mouth": (0, 850, 390, 1160),
        "cheek_skin": (240, 360, 540, 980),
        "face_shape": (120, 160, 560, 1120),
    },
    "selected_004_t_00038.000.jpg": {
        "mouth": (0, 680, 250, 1040),
        "cheek_skin": (0, 310, 330, 940),
        "face_shape": (0, 220, 360, 1110),
    },
    "selected_005_t_00048.000.jpg": {
        "nose": (0, 430, 430, 940),
        "philtrum": (0, 700, 430, 1120),
        "mouth": (0, 830, 390, 1270),
        "cheek_skin": (230, 470, 592, 1020),
    },
    "selected_006_t_00048.500.jpg": {
        "eye": (250, 270, 592, 570),
        "nose": (0, 520, 460, 1030),
        "philtrum": (0, 760, 455, 1130),
        "mouth": (0, 900, 385, 1270),
        "cheek_skin": (250, 440, 592, 1030),
    },
    "selected_007_t_00048.500.jpg": {
        "eye": (250, 270, 592, 570),
        "nose": (0, 520, 460, 1030),
        "philtrum": (0, 760, 455, 1130),
        "mouth": (0, 900, 385, 1270),
        "cheek_skin": (250, 440, 592, 1030),
    },
    "selected_008_t_00039.500.jpg": {
        "eye": (310, 300, 592, 580),
        "nose": (0, 490, 500, 980),
        "philtrum": (0, 730, 520, 1120),
        "mouth": (70, 890, 520, 1240),
        "cheek_skin": (260, 420, 592, 1040),
    },
    "selected_009_t_00039.500.jpg": {
        "eye": (310, 300, 592, 580),
        "nose": (0, 490, 500, 980),
        "philtrum": (0, 730, 520, 1120),
        "mouth": (70, 890, 520, 1240),
        "cheek_skin": (260, 420, 592, 1040),
    },
    "selected_010_t_00047.500.jpg": {
        "eye": (0, 300, 280, 610),
        "nose": (80, 500, 480, 990),
        "philtrum": (70, 730, 455, 1160),
        "mouth": (100, 920, 465, 1280),
        "cheek_skin": (250, 410, 592, 950),
    },
    "selected_011_t_00039.000.jpg": {
        "eye": (50, 160, 420, 520),
        "nose": (0, 470, 330, 940),
        "philtrum": (0, 700, 350, 1030),
        "mouth": (0, 850, 390, 1160),
        "cheek_skin": (240, 360, 540, 980),
        "face_shape": (120, 160, 560, 1120),
    },
    "selected_012_t_00040.000.jpg": {
        "eye": (220, 290, 592, 570),
        "nose": (0, 500, 430, 1030),
        "philtrum": (0, 760, 440, 1120),
        "cheek_skin": (210, 430, 592, 1020),
    },
    "selected_013_t_00040.000.jpg": {
        "eye": (220, 290, 592, 570),
        "nose": (0, 500, 430, 1030),
        "philtrum": (0, 760, 440, 1120),
        "cheek_skin": (210, 430, 592, 1020),
    },
    "selected_014_t_00038.500.jpg": {
        "mouth": (0, 690, 270, 1030),
        "cheek_skin": (0, 290, 340, 940),
        "face_shape": (0, 180, 380, 1120),
    },
    "selected_015_t_00038.500.jpg": {
        "mouth": (0, 690, 270, 1030),
        "cheek_skin": (0, 290, 340, 940),
        "face_shape": (0, 180, 380, 1120),
    },
    "selected_016_t_00015.500.jpg": {
        "nose": (0, 490, 300, 840),
        "philtrum": (0, 620, 300, 940),
        "mouth": (0, 690, 330, 1040),
        "cheek_skin": (0, 390, 410, 1040),
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_region = {}
    manifest_rows = []
    review_rows = []

    for file_name, regions in CROP_MANIFEST.items():
        source = SELECTED_DIR / file_name
        with Image.open(source) as image:
            image = image.convert("RGB")
            for region, raw_spec in regions.items():
                spec = _normalize_spec(region, raw_spec)
                box = spec["box"]
                region_dir = OUTPUT_DIR / region
                region_dir.mkdir(parents=True, exist_ok=True)
                destination = region_dir / file_name
                crop = image.crop(box)
                crop.save(destination, format="JPEG", quality=95)
                review_path = region_dir / f"{Path(file_name).stem}_review.jpg"
                _draw_landmark_review(crop, spec, review_path)
                row = {
                    "source": str(source),
                    "path": str(destination),
                    "review_path": str(review_path),
                    "file_name": file_name,
                    "region": region,
                    "box": box,
                    "points": spec["points"],
                    "usable_for_alignment": spec["usable"],
                    "landmark_source": spec["landmark_source"],
                    "warning": spec.get("warning", ""),
                }
                rows_by_region.setdefault(region, []).append(row)
                manifest_rows.append(row)
                review_rows.append(row)

    for region, rows in sorted(rows_by_region.items()):
        build_reference_board(
            [
                {
                    "path": row["path"],
                    "file_name": row["file_name"],
                    "total_score": "ai-box",
                }
                for row in rows
            ],
            BOARDS_DIR / f"{region}.jpg",
            title=f"AI Curated {region.replace('_', ' ').title()} Crops",
            columns=4,
            thumb_size=(220, 180),
        )
        build_reference_board(
            [
                {
                    "path": row["review_path"],
                    "file_name": row["file_name"],
                    "total_score": "accept" if row["usable_for_alignment"] else "reject",
                }
                for row in rows
            ],
            BOARDS_DIR / f"{region}_landmark_review.jpg",
            title=f"AI Reviewed {region.replace('_', ' ').title()} Landmarks",
            columns=4,
            thumb_size=(220, 180),
        )

    build_reference_board(
        [
            {
                "path": row["review_path"],
                "file_name": f"{row['region']} {row['file_name']}",
                "total_score": "accept" if row["usable_for_alignment"] else "reject",
            }
            for row in review_rows
        ],
        BOARDS_DIR / "all_landmark_review.jpg",
        title="AI Overall Landmark Review",
        columns=4,
        thumb_size=(220, 180),
    )
    accepted_alignment_rows = [
        row
        for row in review_rows
        if row["usable_for_alignment"] and row["region"] not in {"cheek_skin", "face_shape"}
    ]
    build_reference_board(
        [
            {
                "path": row["review_path"],
                "file_name": f"{row['region']} {row['file_name']}",
                "total_score": row["landmark_source"],
            }
            for row in accepted_alignment_rows
        ],
        BOARDS_DIR / "accepted_alignment_landmark_review.jpg",
        title="AI Accepted Alignment Landmark Review",
        columns=4,
        thumb_size=(220, 180),
    )

    summary = {
        "source_sheet": str(CASE_ROOT / "contact_sheets" / "selected_keyframes.jpg"),
        "source_keyframes": str(SELECTED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "method": "AI visual box and landmark selection from selected-keyframes sheet; crops taken from original keyframe images.",
        "regions": {region: len(rows) for region, rows in sorted(rows_by_region.items())},
        "accepted_for_alignment": sum(1 for row in manifest_rows if row["usable_for_alignment"]),
        "rejected_for_alignment": sum(1 for row in manifest_rows if not row["usable_for_alignment"]),
        "overall_review_sheet": str(BOARDS_DIR / "all_landmark_review.jpg"),
        "accepted_alignment_review_sheet": str(BOARDS_DIR / "accepted_alignment_landmark_review.jpg"),
        "ai_audit": {
            "decision": "use_accepted_alignment_crops_for_local_opencv_trials_only",
            "excluded_regions": ["cheek_skin", "face_shape"],
            "known_rejections": ["mouth:selected_002_t_00047.500.jpg"],
            "warnings": [
                "Most points are AI-estimated from visual crops, not model landmarks.",
                "Perspective risk remains high for nose and mouth close-ups.",
                "Do not run whole-face homography from these points.",
            ],
        },
        "rows": manifest_rows,
    }
    (OUTPUT_DIR / "ai_curated_crops_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# AI Curated Crops Summary",
        "",
        f"- source_sheet: {summary['source_sheet']}",
        f"- source_keyframes: {summary['source_keyframes']}",
        f"- output_dir: {summary['output_dir']}",
        f"- method: {summary['method']}",
        f"- accepted_for_alignment: {summary['accepted_for_alignment']}",
        f"- rejected_for_alignment: {summary['rejected_for_alignment']}",
        f"- overall_review_sheet: {summary['overall_review_sheet']}",
        f"- accepted_alignment_review_sheet: {summary['accepted_alignment_review_sheet']}",
        "",
        "## Regions",
        "",
    ]
    for region, count in summary["regions"].items():
        lines.append(f"- {region}: {count}")
    lines.extend(["", "## AI Audit", ""])
    lines.append(f"- decision: {summary['ai_audit']['decision']}")
    for warning in summary["ai_audit"]["warnings"]:
        lines.append(f"- warning: {warning}")
    (OUTPUT_DIR / "ai_curated_crops_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR / "ai_curated_crops_summary.md")
    print(BOARDS_DIR)


def _normalize_spec(region, raw_spec):
    if isinstance(raw_spec, dict):
        box = tuple(raw_spec["box"])
        points = raw_spec.get("points", {})
        usable = bool(raw_spec.get("usable", True))
        landmark_source = "ai_visual"
        warning = raw_spec.get("warning", "")
    else:
        box = tuple(raw_spec)
        points = _default_points(region, box)
        usable = bool(points) and region not in {"face_shape", "cheek_skin"}
        landmark_source = "ai_visual_estimated"
        warning = "estimated points require lower confidence"
    return {
        "box": box,
        "points": points,
        "usable": usable,
        "landmark_source": landmark_source,
        "warning": warning,
    }


def _default_points(region, box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    if region == "mouth":
        return {
            "left_mouth_corner": (int(width * 0.18), int(height * 0.62)),
            "right_mouth_corner": (int(width * 0.82), int(height * 0.56)),
            "upper_lip_center": (int(width * 0.50), int(height * 0.42)),
            "lower_lip_center": (int(width * 0.52), int(height * 0.72)),
            "philtrum_center": (int(width * 0.48), int(height * 0.24)),
        }
    if region == "philtrum":
        return {
            "nose_base": (int(width * 0.46), int(height * 0.26)),
            "philtrum_center": (int(width * 0.48), int(height * 0.50)),
            "upper_lip_center": (int(width * 0.52), int(height * 0.76)),
        }
    if region == "nose":
        return {
            "nose_bridge": (int(width * 0.50), int(height * 0.18)),
            "nose_tip": (int(width * 0.48), int(height * 0.58)),
            "left_nostril": (int(width * 0.32), int(height * 0.68)),
            "right_nostril": (int(width * 0.68), int(height * 0.68)),
        }
    if region == "eye":
        return {
            "outer_eye_corner": (int(width * 0.18), int(height * 0.50)),
            "inner_eye_corner": (int(width * 0.82), int(height * 0.48)),
            "upper_lid_center": (int(width * 0.50), int(height * 0.38)),
            "lower_lid_center": (int(width * 0.50), int(height * 0.62)),
        }
    return {}


def _draw_landmark_review(crop, spec, output_path):
    image = crop.copy()
    draw = ImageDraw.Draw(image)
    colors = {
        "usable": (20, 180, 80),
        "reject": (230, 40, 50),
        "point": (255, 230, 0),
        "text": (255, 255, 255),
    }
    outline = colors["usable"] if spec["usable"] else colors["reject"]
    draw.rectangle((2, 2, image.width - 3, image.height - 3), outline=outline, width=5)
    for name, point in spec["points"].items():
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors["point"], outline=(0, 0, 0))
        draw.text((x + 7, max(0, y - 9)), _short_label(name), fill=colors["text"], stroke_width=2, stroke_fill=(0, 0, 0))
    status = "ACCEPT" if spec["usable"] else "REJECT"
    draw.text((8, 8), status, fill=outline, stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(output_path, format="JPEG", quality=95)


def _short_label(name):
    return {
        "left_mouth_corner": "LMC",
        "right_mouth_corner": "RMC",
        "upper_lip_center": "UL",
        "lower_lip_center": "LL",
        "philtrum_center": "PH",
        "nose_tip": "NT",
        "left_nostril": "LN",
        "right_nostril": "RN",
        "nose_bridge": "NB",
        "nose_base": "BASE",
        "inner_eye_corner": "IN",
        "outer_eye_corner": "OUT",
        "upper_lid_center": "UP",
        "lower_lid_center": "LOW",
    }.get(name, name[:4].upper())


if __name__ == "__main__":
    main()
