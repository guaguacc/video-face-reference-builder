from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.local_reference import build_local_reference_package


CASE_ROOT = PROJECT_ROOT / "outputs" / "keyframe_selection_case"


def main() -> None:
    summary = build_local_reference_package(
        keyframe_selection_path=CASE_ROOT / "keyframe_selection.json",
        selected_keyframes_dir=CASE_ROOT / "selected_keyframes",
        output_dir=CASE_ROOT / "local_reference",
        optional_enhancement_summary_path=CASE_ROOT / "codeformer_auxiliary_summary.json",
    )
    print(CASE_ROOT / "local_reference" / "local_reference_summary.md")
    print(summary["reference_boards_dir"])


if __name__ == "__main__":
    main()
