import json
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

from vfrb.build_reference_boards import build_reference_board


PathLike = Union[str, Path]

REGION_GROUPS = {
    "mouth": {"mouth", "lower_face"},
    "philtrum": {"philtrum", "mouth"},
    "nose": {"nose", "philtrum"},
    "eye": {"eye"},
    "cheek_skin": {"cheek", "face_side"},
    "face_shape": {"face_side", "lower_face", "cheek"},
}

REGION_CROP_BOXES = {
    "mouth": (0.00, 0.38, 0.92, 0.86),
    "philtrum": (0.00, 0.24, 0.92, 0.72),
    "nose": (0.00, 0.12, 0.92, 0.62),
    "eye": (0.00, 0.00, 0.92, 0.42),
    "cheek_skin": (0.00, 0.14, 1.00, 0.82),
    "face_shape": (0.00, 0.00, 1.00, 1.00),
}


def build_local_reference_package(
    keyframe_selection_path: PathLike,
    selected_keyframes_dir: PathLike,
    output_dir: PathLike,
    optional_enhancement_summary_path: Optional[PathLike] = None,
) -> dict[str, Any]:
    selection_path = Path(keyframe_selection_path)
    selected_dir = Path(selected_keyframes_dir)
    output = Path(output_dir)
    regions_dir = output / "regions"
    crops_dir = output / "crops"
    boards_dir = output / "reference_boards"
    regions_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = _attach_selected_files(selection.get("selected", []), selected_dir)
    optional_enhancement = _load_optional_enhancement(optional_enhancement_summary_path)

    region_results = {}
    for region_name, accepted_labels in REGION_GROUPS.items():
        region_rows = [
            row
            for row in selected
            if accepted_labels.intersection(set(row.get("visible_regions", [])))
        ]
        region_rows = sorted(region_rows, key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
        region_dir = regions_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        for row in region_rows:
            source = Path(row["selected_path"])
            shutil.copy2(source, region_dir / source.name)

        board_path = boards_dir / f"{region_name}.jpg"
        if region_rows:
            build_reference_board(
                [_to_board_row(row) for row in region_rows],
                board_path,
                title=f"{region_name.replace('_', ' ').title()} Reference",
                columns=4,
            )

        crop_rows = _build_region_crops(region_name, region_rows, crops_dir / region_name)
        crop_board_path = boards_dir / f"{region_name}_crops.jpg"
        if crop_rows:
            build_reference_board(
                crop_rows,
                crop_board_path,
                title=f"{region_name.replace('_', ' ').title()} Rough Crops",
                columns=4,
                thumb_size=(220, 160),
            )

        region_results[region_name] = {
            "count": len(region_rows),
            "board_path": str(board_path) if region_rows else "",
            "crop_count": len(crop_rows),
            "crop_board_path": str(crop_board_path) if crop_rows else "",
            "frames": [_to_report_row(row) for row in region_rows],
        }

    summary = {
        "source_keyframe_selection": str(selection_path),
        "selected_keyframe_count": len(selected),
        "reference_boards_dir": str(boards_dir),
        "regions": region_results,
        "optional_codeformer": optional_enhancement,
        "evidence_boundary": [
            "Region boards are grouped from original selected keyframes.",
            "Rough crops use fixed relative boxes for inspection and must be verified before reconstruction.",
            "CodeFormer outputs, when present, are optional references and do not replace source frames.",
            "No face shape or missing-region inference is performed in this stage.",
        ],
    }
    (output / "local_reference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(summary, output / "local_reference_summary.md")
    return summary


def _attach_selected_files(selected_rows: list[dict[str, Any]], selected_dir: Path) -> list[dict[str, Any]]:
    selected_files = sorted(selected_dir.glob("selected_*"))
    rows = []
    for row, selected_file in zip(selected_rows, selected_files):
        item = dict(row)
        item["selected_path"] = str(selected_file)
        item["selected_file_name"] = selected_file.name
        rows.append(item)
    return rows


def _load_optional_enhancement(path: Optional[PathLike]) -> dict[str, Any]:
    if not path:
        return {"available": False}
    source = Path(path)
    if not source.exists():
        return {"available": False}
    data = json.loads(source.read_text(encoding="utf-8"))
    return {
        "available": True,
        "summary_path": str(source),
        "codeformer_input_count": data.get("codeformer_input_count", 0),
        "optional_enhanced_count": data.get("optional_enhanced_count", 0),
        "original_keyframes_remain_primary": data.get("original_keyframes_remain_primary", True),
    }


def _to_board_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["selected_path"],
        "file_name": row["selected_file_name"],
        "total_score": row.get("final_score", ""),
    }


def _to_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": row.get("selected_file_name", ""),
        "timestamp": row.get("timestamp", ""),
        "score": row.get("final_score", ""),
        "visible_regions": row.get("visible_regions", []),
        "reason": row.get("reason", ""),
    }


def _build_region_crops(region_name: str, rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    crop_box = REGION_CROP_BOXES[region_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_rows = []
    for row in rows:
        source = Path(row["selected_path"])
        destination = output_dir / source.name
        with Image.open(source) as image:
            width, height = image.size
            left = int(width * crop_box[0])
            top = int(height * crop_box[1])
            right = int(width * crop_box[2])
            bottom = int(height * crop_box[3])
            image.convert("RGB").crop((left, top, right, bottom)).save(destination, format="JPEG", quality=94)
        crop_rows.append(
            {
                "path": str(destination),
                "file_name": destination.name,
                "total_score": row.get("final_score", ""),
            }
        )
    return crop_rows


def _write_markdown_summary(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Local Reference Summary",
        "",
        f"- selected_keyframe_count: {summary['selected_keyframe_count']}",
        f"- reference_boards_dir: {summary['reference_boards_dir']}",
        "",
        "## Region Boards",
        "",
    ]
    for region_name, region in summary["regions"].items():
        lines.append(
            f"- {region_name}: count={region['count']}, board={region['board_path']}, "
            f"crop_count={region['crop_count']}, crop_board={region['crop_board_path']}"
        )

    lines.extend(["", "## Optional CodeFormer", ""])
    optional = summary["optional_codeformer"]
    if optional.get("available"):
        lines.append(f"- codeformer_input_count: {optional.get('codeformer_input_count', 0)}")
        lines.append(f"- optional_enhanced_count: {optional.get('optional_enhanced_count', 0)}")
        lines.append(f"- original_keyframes_remain_primary: {optional.get('original_keyframes_remain_primary', True)}")
    else:
        lines.append("- available: false")

    lines.extend(["", "## Evidence Boundary", ""])
    for item in summary["evidence_boundary"]:
        lines.append(f"- {item}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
