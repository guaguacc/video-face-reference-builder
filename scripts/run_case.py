from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfrb.pipeline import run_pipeline

CASE_VIDEO = PROJECT_ROOT / "案例" / "视频人脸拟合文件" / "原视频" / "d6b785016780fcc8b3e32a51954b11ec.mp4"


def main() -> None:
    result = run_pipeline(
        video_path=CASE_VIDEO,
        output_root=PROJECT_ROOT / "outputs",
        every_n_frames=30,
        max_frames=120,
        best_frame_count=24,
    )
    print(result.run_dir)


if __name__ == "__main__":
    main()
