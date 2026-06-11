from pathlib import Path
import json
import shutil
import sys

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
AI_CROPS_ROOT = CASE_ROOT / "local_reference" / "ai_curated_crops"
OUTPUT_DIR = CASE_ROOT / "candidate_guided_composite"
CANDIDATE_SOURCE = CASE_ROOT / "selected_keyframes" / "selected_003_t_00039.000.jpg"

REGION_BOARD_PATHS = {
    "mouth": AI_CROPS_ROOT / "boards" / "mouth.jpg",
    "philtrum": AI_CROPS_ROOT / "boards" / "philtrum.jpg",
    "nose": AI_CROPS_ROOT / "boards" / "nose.jpg",
    "eye": AI_CROPS_ROOT / "boards" / "eye.jpg",
    "cheek_skin": AI_CROPS_ROOT / "boards" / "cheek_skin.jpg",
    "face_shape": AI_CROPS_ROOT / "boards" / "face_shape.jpg",
}

REGION_BOXES_ON_CANDIDATE = {
    "eye": (50, 165, 420, 515),
    "nose": (0, 455, 350, 905),
    "philtrum": (0, 700, 365, 1035),
    "mouth": (0, 845, 405, 1170),
    "cheek_skin": (235, 355, 545, 995),
    "face_shape": (115, 145, 570, 1135),
}

REGION_COLORS = {
    "eye": (50, 110, 255),
    "nose": (40, 170, 90),
    "philtrum": (255, 150, 30),
    "mouth": (230, 50, 100),
    "cheek_skin": (170, 80, 230),
    "face_shape": (80, 80, 80),
}

AUDIT_NOTES = {
    "mouth": "Use selected_003/011 as strongest; 006/007 only close-up auxiliary; reject overlay-obscured crops.",
    "philtrum": "Use to check vertical color/shadow between nose base and upper lip; high perspective risk.",
    "nose": "Use for nose base and nostril shading, not full true nose width due close lens distortion.",
    "eye": "Haar/MediaPipe eye crops are useful for eye shape reference; coverage is limited.",
    "cheek_skin": "Use for skin tone/color consistency only, not geometry.",
    "face_shape": "Low confidence; candidate face outline is mostly AI/global scaffold.",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = OUTPUT_DIR / "candidate_full_face_reference.png"
    if CANDIDATE_SOURCE.exists():
        shutil.copy2(CANDIDATE_SOURCE, candidate_path)

    candidate = Image.open(candidate_path).convert("RGB")
    annotated = candidate.copy()
    draw = ImageDraw.Draw(annotated)
    for region, box in REGION_BOXES_ON_CANDIDATE.items():
        color = REGION_COLORS[region]
        draw.rectangle(box, outline=color, width=6)
        draw.text((box[0] + 8, max(0, box[1] - 28)), region, fill=color)
    annotated_path = OUTPUT_DIR / "candidate_region_overlay.jpg"
    annotated.save(annotated_path, format="JPEG", quality=94)

    sheet_path = OUTPUT_DIR / "candidate_guided_review_sheet.jpg"
    _build_review_sheet(annotated_path, sheet_path)

    summary = {
        "candidate_source": str(CANDIDATE_SOURCE),
        "candidate_reference": str(candidate_path),
        "candidate_region_overlay": str(annotated_path),
        "review_sheet": str(sheet_path),
        "method": "Most complete original selected keyframe used as local scaffold; AI crops used as repeated local evidence checks.",
        "region_boxes_on_candidate": REGION_BOXES_ON_CANDIDATE,
        "audit_notes": AUDIT_NOTES,
        "next_stage": [
            "Do not paste all crops directly.",
            "Use the most complete original keyframe as coordinate scaffold.",
            "Run region-by-region correction: mouth, philtrum/nose base, eye, skin tone.",
            "After each region correction, regenerate review sheet against source evidence.",
        ],
    }
    (OUTPUT_DIR / "candidate_guided_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Candidate Guided Review Summary",
        "",
        f"- candidate_reference: {summary['candidate_reference']}",
        f"- review_sheet: {summary['review_sheet']}",
        f"- method: {summary['method']}",
        "",
        "## Region Audit Notes",
        "",
    ]
    for region, note in AUDIT_NOTES.items():
        lines.append(f"- {region}: {note}")
    lines.extend(["", "## Next Stage", ""])
    for item in summary["next_stage"]:
        lines.append(f"- {item}")
    (OUTPUT_DIR / "candidate_guided_review_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(sheet_path)
    print(OUTPUT_DIR / "candidate_guided_review_summary.md")


def _build_review_sheet(candidate_overlay_path: Path, output_path: Path) -> None:
    padding = 22
    title_height = 50
    candidate_width = 430
    board_width = 430
    row_height = 280
    sheet_width = padding * 3 + candidate_width + board_width
    sheet_height = title_height + padding + row_height * 6 + padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((padding, 16), "Candidate-Guided Composite Review", fill=(20, 20, 20))

    with Image.open(candidate_overlay_path) as candidate:
        candidate_thumb = ImageOps.contain(candidate.convert("RGB"), (candidate_width, row_height * 2))
    sheet.paste(candidate_thumb, (padding, title_height + padding))
    draw.text((padding, title_height + padding + candidate_thumb.height + 8), "Candidate scaffold with target regions", fill=(40, 40, 40))

    x_right = padding * 2 + candidate_width
    y = title_height + padding
    for region, board_path in REGION_BOARD_PATHS.items():
        with Image.open(board_path) as board:
            board_thumb = ImageOps.contain(board.convert("RGB"), (board_width, row_height - 58))
        sheet.paste(board_thumb, (x_right, y))
        draw.text((x_right, y + board_thumb.height + 6), f"{region}: {AUDIT_NOTES[region][:86]}", fill=(40, 40, 40))
        y += row_height
    sheet.save(output_path, format="JPEG", quality=92)


if __name__ == "__main__":
    main()
