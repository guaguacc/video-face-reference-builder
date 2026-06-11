import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from vfrb.build_reference_boards import build_reference_board
from vfrb.extract_frames import extract_frames
from vfrb.generate_report import generate_report
from vfrb.score_frames import score_frames, select_best_frames, write_scores
from vfrb.video_info import read_video_info, write_video_info


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    video_info_path: Path
    scores_path: Path
    report_path: Path
    reference_boards: list[Path]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_pipeline(
    video_path: PathLike,
    output_root: PathLike,
    run_id: Optional[str] = None,
    every_n_frames: int = 30,
    max_frames: Optional[int] = 120,
    best_frame_count: int = 24,
) -> PipelineResult:
    video = Path(video_path)
    root = Path(output_root)
    run_name = run_id or make_run_id()
    run_dir = root / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")

    frames_dir = run_dir / "frames"
    best_frames_dir = run_dir / "best_frames"
    boards_dir = run_dir / "reference_boards"
    best_frames_dir.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)

    info = read_video_info(video)
    video_info_path = run_dir / "video_info.json"
    write_video_info(info, video_info_path)

    extract_frames(
        video,
        frames_dir,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
    )
    scores = score_frames(frames_dir)
    best = select_best_frames(scores, limit=best_frame_count)

    scores_path = run_dir / "frame_scores.json"
    write_scores(scores, scores_path)

    for row in best:
        source = Path(row["path"])
        shutil.copy2(source, best_frames_dir / source.name)

    reference_boards = []
    if best:
        best_board = boards_dir / "best_frames.jpg"
        build_reference_board(best, best_board, title="Best Frames")
        reference_boards.append(best_board)

        sharpest = sorted(best, key=lambda row: row["sharpness"], reverse=True)
        sharp_board = boards_dir / "sharpest_frames.jpg"
        build_reference_board(sharpest, sharp_board, title="Sharpest Frames")
        reference_boards.append(sharp_board)

        balanced = sorted(best, key=lambda row: row["brightness_score"], reverse=True)
        brightness_board = boards_dir / "balanced_brightness_frames.jpg"
        build_reference_board(balanced, brightness_board, title="Balanced Brightness Frames")
        reference_boards.append(brightness_board)

    params = {
        "every_n_frames": every_n_frames,
        "max_frames": max_frames,
        "best_frame_count": best_frame_count,
    }
    outputs = {
        "frames_dir": str(frames_dir),
        "scores_path": str(scores_path),
        "best_frames_dir": str(best_frames_dir),
        "reference_boards": [str(path) for path in reference_boards],
    }
    report_path = run_dir / "report.md"
    generate_report(info, params, best, outputs, report_path)

    return PipelineResult(
        run_dir=run_dir,
        video_info_path=video_info_path,
        scores_path=scores_path,
        report_path=report_path,
        reference_boards=reference_boards,
    )
