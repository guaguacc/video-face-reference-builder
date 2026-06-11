from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CASE_ROOT = Path("outputs/keyframe_selection_case")
AI_CROPS_ROOT = CASE_ROOT / "local_reference" / "ai_curated_crops"
OUT_DIR = CASE_ROOT / "candidate_guided_composite" / "ai_assembly_round_01"


FEATURES = [
    ("FACE SHAPE anchor", AI_CROPS_ROOT / "face_shape" / "selected_003_t_00039.000.jpg"),
    ("FACE SHAPE side", AI_CROPS_ROOT / "face_shape" / "selected_004_t_00038.000.jpg"),
    ("EYE anchor", AI_CROPS_ROOT / "eye" / "selected_003_t_00039.000.jpg"),
    ("EYE soft lid", AI_CROPS_ROOT / "eye" / "selected_011_t_00039.000.jpg"),
    ("NOSE anchor", AI_CROPS_ROOT / "nose" / "selected_003_t_00039.000.jpg"),
    ("NOSE nostril/side", AI_CROPS_ROOT / "nose" / "selected_008_t_00039.500.jpg"),
    ("PHILTRUM anchor", AI_CROPS_ROOT / "philtrum" / "selected_003_t_00039.000.jpg"),
    ("PHILTRUM alt", AI_CROPS_ROOT / "philtrum" / "selected_008_t_00039.500.jpg"),
    ("MOUTH anchor", AI_CROPS_ROOT / "mouth" / "selected_003_t_00039.000.jpg"),
    ("MOUTH frontal", AI_CROPS_ROOT / "mouth" / "selected_008_t_00039.500.jpg"),
    ("MOUTH side alt", AI_CROPS_ROOT / "mouth" / "selected_006_t_00048.500.jpg"),
    ("CHEEK/SKIN", AI_CROPS_ROOT / "cheek_skin" / "selected_003_t_00039.000.jpg"),
]


def build_feature_pack(output_path: Path) -> Path:
    missing = [path for _, path in FEATURES if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing feature crop(s):\n{missing_text}")

    cols = 3
    cell_w, cell_h = 360, 330
    pad = 24
    label_h = 28
    rows = (len(FEATURES) + cols - 1) // cols

    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (label, path) in enumerate(FEATURES):
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image.thumbnail((cell_w - 20, cell_h - label_h - 20), Image.Resampling.LANCZOS)

        row, col = divmod(idx, cols)
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)

        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(210, 210, 210), width=1)
        draw.text((x + 10, y + 8), label, fill=(20, 20, 20), font=font)
        paste_x = x + (cell_w - image.width) // 2
        paste_y = y + label_h + (cell_h - label_h - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def main() -> None:
    output_path = OUT_DIR / "strong_feature_reference_pack.jpg"
    result = build_feature_pack(output_path)
    print(result)


if __name__ == "__main__":
    main()
