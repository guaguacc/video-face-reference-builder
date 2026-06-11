from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.keyframe_selection import run_keyframe_selection


CASE_VIDEO = PROJECT_ROOT / "案例" / "视频人脸拟合文件" / "原视频" / "d6b785016780fcc8b3e32a51954b11ec.mp4"


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "keyframe_selection_case"
    ai_judgement_path = PROJECT_ROOT / "outputs" / "ai_global_judgement_case.json"
    result = run_keyframe_selection(
        video_path=CASE_VIDEO,
        output_dir=output_dir,
        ai_judgement_path=ai_judgement_path if ai_judgement_path.exists() else None,
        global_step_seconds=0.5,
        top_global_count=36,
        max_refined_windows=4,
        refined_step_seconds=0.25,
        final_count=16,
    )
    print(output_dir)
    print(result["windows"])


if __name__ == "__main__":
    main()
