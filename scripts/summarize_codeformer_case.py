from pathlib import Path
import json
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.auxiliary_enhancement import summarize_codeformer_results
from vfrb.build_reference_boards import build_reference_board


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"


def main() -> None:
    codeformer_dir = CASE_ROOT / "codeformer_candidates"
    selected_original_dir = CASE_ROOT / "selected_keyframes"
    final_original_dir = CASE_ROOT / "selected_keyframes_final" / "original"
    optional_enhanced_dir = CASE_ROOT / "selected_keyframes_final" / "enhanced_optional"
    sheets_dir = CASE_ROOT / "contact_sheets"
    final_original_dir.mkdir(parents=True, exist_ok=True)
    optional_enhanced_dir.mkdir(parents=True, exist_ok=True)

    original_paths = sorted(selected_original_dir.glob("*.jpg"))
    for original in original_paths:
        shutil.copy2(original, final_original_dir / original.name)

    rows = summarize_codeformer_results(codeformer_dir)
    enhanced_rows = []
    original_only_rows = []
    for row in rows:
        final_path = Path(row["final_result_path"])
        if row["enhanced_success"]:
            destination = optional_enhanced_dir / final_path.name
            shutil.copy2(final_path, destination)
            enhanced_rows.append({"path": str(destination), "file_name": destination.name, "total_score": "optional"})
        else:
            original_only_rows.append({"path": str(final_path), "file_name": final_path.name, "total_score": "fallback"})

    if enhanced_rows:
        build_reference_board(
            enhanced_rows,
            sheets_dir / "codeformer_optional_success.jpg",
            title="CodeFormer Optional Success",
            columns=4,
        )
    if original_only_rows:
        build_reference_board(
            original_only_rows,
            sheets_dir / "codeformer_original_only.jpg",
            title="CodeFormer Original Only",
            columns=4,
        )

    summary = {
        "original_keyframe_count": len(original_paths),
        "codeformer_input_count": len(rows),
        "optional_enhanced_count": len(enhanced_rows),
        "original_keyframes_remain_primary": True,
        "rows": rows,
    }
    (CASE_ROOT / "codeformer_auxiliary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# CodeFormer Auxiliary Summary",
        "",
        "CodeFormer is an optional reference branch. Original selected keyframes remain the primary evidence.",
        "",
        f"- original_keyframe_count: {summary['original_keyframe_count']}",
        f"- codeformer_input_count: {summary['codeformer_input_count']}",
        f"- optional_enhanced_count: {summary['optional_enhanced_count']}",
        f"- original_keyframes_remain_primary: {summary['original_keyframes_remain_primary']}",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        report_lines.append(
            f"- {row['stem']}: detected_faces={row['detected_faces']}, "
            f"use_for_next_stage={row['use_for_next_stage']}"
        )
    (CASE_ROOT / "codeformer_auxiliary_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(CASE_ROOT / "codeformer_auxiliary_summary.md")


if __name__ == "__main__":
    main()
