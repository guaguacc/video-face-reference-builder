import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2

from vfrb.build_reference_boards import build_reference_board
from vfrb.score_frames import score_frame
from vfrb.video_info import read_video_info


PathLike = Union[str, Path]


@dataclass(frozen=True)
class Candidate:
    timestamp: float
    path: str
    final_score: float
    visible_regions: list[str]
    stage: str = "global"
    deterministic_score: float = 0.0
    ai_reference_value: float = 0.0
    diversity_bonus: float = 0.0
    reason: str = ""


def timestamps_for_global_sweep(duration_seconds: float, step_seconds: Optional[float] = None) -> list[float]:
    if duration_seconds <= 0:
        return []
    step = step_seconds or _default_global_step(duration_seconds)
    if step <= 0:
        raise ValueError("step_seconds must be greater than 0")

    timestamps = []
    current = 0.0
    while current <= duration_seconds:
        timestamps.append(round(current, 3))
        current += step
    if timestamps[-1] < duration_seconds:
        timestamps.append(round(duration_seconds, 3))
    return timestamps


def _default_global_step(duration_seconds: float) -> float:
    if duration_seconds <= 90:
        return 0.5
    if duration_seconds <= 300:
        return 1.0
    return 2.0


def combine_scores(
    deterministic_score: float,
    ai_reference_value: float,
    diversity_bonus: float,
    deterministic_weight: float = 0.55,
    ai_weight: float = 0.35,
    diversity_weight: float = 0.10,
) -> float:
    value = (
        deterministic_score * deterministic_weight
        + ai_reference_value * ai_weight
        + diversity_bonus * diversity_weight
    )
    return round(value, 4)


def cluster_timestamps(
    timestamps: list[float],
    max_gap_seconds: float = 1.0,
    padding_seconds: float = 0.5,
) -> list[tuple[float, float]]:
    if not timestamps:
        return []

    ordered = sorted(timestamps)
    groups = [[ordered[0]]]
    for timestamp in ordered[1:]:
        if timestamp - groups[-1][-1] <= max_gap_seconds:
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])

    return [
        (
            round(max(0.0, group[0] - padding_seconds), 3),
            round(group[-1] + padding_seconds, 3),
        )
        for group in groups
    ]


def refine_window_timestamps(start: float, end: float, step_seconds: float = 0.25) -> list[float]:
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be greater than 0")

    timestamps = []
    current = start
    while current <= end + 1e-9:
        timestamps.append(round(current, 3))
        current += step_seconds
    if timestamps[-1] != round(end, 3):
        timestamps.append(round(end, 3))
    return timestamps


def select_diverse_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    if limit < 1:
        return []

    selected = []
    used_regions = set()
    for candidate in sorted(candidates, key=lambda item: item.final_score, reverse=True):
        primary_region = candidate.visible_regions[0] if candidate.visible_regions else "unknown"
        if primary_region not in used_regions:
            selected.append(candidate)
            used_regions.add(primary_region)
        if len(selected) >= limit:
            return selected

    for candidate in sorted(candidates, key=lambda item: item.final_score, reverse=True):
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def rank_windows_by_candidate_score(
    windows: list[tuple[float, float]],
    candidates: list[Candidate],
) -> list[tuple[float, float]]:
    def score_window(window: tuple[float, float]) -> tuple[float, float]:
        start, end = window
        inside = [candidate for candidate in candidates if start <= candidate.timestamp <= end]
        if not inside:
            return (0.0, 0.0)
        max_score = max(candidate.final_score for candidate in inside)
        avg_score = sum(candidate.final_score for candidate in inside) / len(inside)
        return (max_score, avg_score)

    return sorted(windows, key=score_window, reverse=True)


def run_keyframe_selection(
    video_path: PathLike,
    output_dir: PathLike,
    ai_judgement_path: Optional[PathLike] = None,
    global_step_seconds: Optional[float] = None,
    top_global_count: int = 24,
    max_refined_windows: int = 4,
    refined_step_seconds: float = 0.25,
    final_count: int = 12,
    min_ai_reference_value: Optional[float] = None,
) -> dict[str, Any]:
    video = Path(video_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    global_dir = output / "global_candidates"
    refined_dir = output / "refined_windows"
    selected_dir = output / "selected_keyframes"
    sheets_dir = output / "contact_sheets"
    for directory in [global_dir, refined_dir, selected_dir, sheets_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    info = read_video_info(video)
    ai_judgements = _load_ai_judgements(ai_judgement_path)
    global_timestamps = timestamps_for_global_sweep(info["duration_seconds"], global_step_seconds)
    global_rows = _extract_timestamped_frames(video, global_timestamps, global_dir, "global")
    global_candidates = [_row_to_candidate(row, stage="global", ai_judgements=ai_judgements) for row in global_rows]
    global_top = sorted(global_candidates, key=lambda item: item.final_score, reverse=True)[:top_global_count]

    if global_top:
        build_reference_board(
            [_candidate_to_board_row(candidate) for candidate in global_top],
            sheets_dir / "global_top_candidates.jpg",
            title="Global Top Candidates",
        )

    clustering_candidates = [
        candidate
        for candidate in global_top
        if candidate.ai_reference_value >= (40.0 if ai_judgements else 0.0)
    ]
    clustered = cluster_timestamps([candidate.timestamp for candidate in clustering_candidates])
    windows = rank_windows_by_candidate_score(clustered, global_top)[:max_refined_windows]

    refined_candidates = []
    for window_index, (start, end) in enumerate(windows, start=1):
        window_dir = refined_dir / f"window_{window_index:02d}_{start:.3f}_{end:.3f}"
        window_dir.mkdir(parents=True, exist_ok=True)
        timestamps = refine_window_timestamps(start, end, refined_step_seconds)
        rows = _extract_timestamped_frames(video, timestamps, window_dir, "refined")
        refined_candidates.extend(
            _row_to_candidate(row, stage="refined", ai_judgements=ai_judgements)
            for row in rows
        )

    min_ai_value = min_ai_reference_value
    if min_ai_value is None:
        min_ai_value = 40.0 if ai_judgements else 0.0
    all_candidates = [
        candidate
        for candidate in global_top + refined_candidates
        if candidate.ai_reference_value >= min_ai_value
    ]
    selected = select_diverse_candidates(all_candidates, limit=final_count)
    for index, candidate in enumerate(selected, start=1):
        source = Path(candidate.path)
        suffix = source.suffix.lower()
        destination = selected_dir / f"selected_{index:03d}_t_{candidate.timestamp:09.3f}{suffix}"
        shutil.copy2(source, destination)

    if selected:
        build_reference_board(
            [_candidate_to_board_row(candidate) for candidate in selected],
            sheets_dir / "selected_keyframes.jpg",
            title="Selected Keyframes",
        )

    result = {
        "video_info": info,
        "global_step_seconds": global_step_seconds or _default_global_step(info["duration_seconds"]),
        "ai_judgement_path": str(ai_judgement_path) if ai_judgement_path else "",
        "global_candidate_count": len(global_candidates),
        "top_global_count": len(global_top),
        "windows": [{"start": start, "end": end} for start, end in windows],
        "refined_candidate_count": len(refined_candidates),
        "min_ai_reference_value": min_ai_value,
        "selected": [asdict(candidate) for candidate in selected],
    }

    (output / "keyframe_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(result, output / "keyframe_selection.md")
    return result


def _extract_timestamped_frames(
    video_path: Path,
    timestamps: list[float],
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    rows = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            path = output_dir / f"{prefix}_t_{timestamp:09.3f}.jpg"
            written = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not written:
                raise ValueError(f"Could not write frame: {path}")
            row = score_frame(path)
            row["timestamp"] = timestamp
            rows.append(row)
    finally:
        capture.release()
    return rows


def _load_ai_judgements(path: Optional[PathLike]) -> dict[float, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    items = data.get("judgements", data if isinstance(data, list) else [])
    return {round(float(item["timestamp"]), 3): item for item in items}


def _row_to_candidate(
    row: dict[str, Any],
    stage: str,
    ai_judgements: Optional[dict[float, dict[str, Any]]] = None,
) -> Candidate:
    deterministic_score = float(row.get("total_score", 0.0))
    judgement = (ai_judgements or {}).get(round(float(row["timestamp"]), 3), {})
    default_ai_value = 20.0 if ai_judgements else 50.0
    ai_reference_value = float(judgement.get("face_reference_value", default_ai_value))
    diversity_bonus = 0.0
    final_score = combine_scores(deterministic_score, ai_reference_value, diversity_bonus)
    visible_regions = judgement.get("visible_regions", ["unknown"])
    reason = judgement.get("reason", "Automatic candidate; AI region labeling not yet applied.")
    return Candidate(
        timestamp=float(row["timestamp"]),
        path=str(row["path"]),
        final_score=final_score,
        visible_regions=visible_regions,
        stage=stage,
        deterministic_score=deterministic_score,
        ai_reference_value=ai_reference_value,
        diversity_bonus=diversity_bonus,
        reason=reason,
    )


def _candidate_to_board_row(candidate: Candidate) -> dict[str, Any]:
    path = Path(candidate.path)
    return {
        "path": str(path),
        "file_name": f"t={candidate.timestamp:.3f}s {path.name}",
        "total_score": candidate.final_score,
    }


def _write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Keyframe Selection Report",
        "",
        "## Global Search",
        "",
        f"- global_step_seconds: {result['global_step_seconds']}",
        f"- ai_judgement_path: {result['ai_judgement_path']}",
        f"- global_candidate_count: {result['global_candidate_count']}",
        f"- top_global_count: {result['top_global_count']}",
        "",
        "## Refined Windows",
        "",
    ]
    for window in result["windows"]:
        lines.append(f"- {window['start']:.3f}s to {window['end']:.3f}s")

    lines.extend(
        [
            "",
            "## Selected Keyframes",
            "",
        ]
    )
    for index, candidate in enumerate(result["selected"], start=1):
        lines.append(
            f"{index}. t={candidate['timestamp']:.3f}s "
            f"stage={candidate['stage']} "
            f"score={candidate['final_score']} "
            f"regions={','.join(candidate['visible_regions'])}"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The whole video is searched before any local window refinement.",
            "- Current AI visual judging is represented by an injectable score field; region labels are not yet automated.",
            "- Old manually selected timestamps can be used as a benchmark, not as hard-coded selection logic.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
