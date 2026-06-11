from pathlib import Path
from typing import Any, Union


PathLike = Union[str, Path]


def generate_report(
    video_info: dict[str, Any],
    params: dict[str, Any],
    best_frames: list[dict[str, Any]],
    outputs: dict[str, Any],
    output_path: PathLike,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Video Face Reference Report",
        "",
        "## Input Video",
        "",
        f"- file_name: {video_info.get('file_name', '')}",
        f"- resolution: {video_info.get('width', '')}x{video_info.get('height', '')}",
        f"- fps: {video_info.get('fps', '')}",
        f"- frame_count: {video_info.get('frame_count', '')}",
        f"- duration_seconds: {round(float(video_info.get('duration_seconds', 0.0)), 3)}",
        f"- codec: {video_info.get('codec', '')}",
        "",
        "## Parameters",
        "",
    ]

    for key in sorted(params):
        lines.append(f"- {key}: {params[key]}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- frames_dir: {outputs.get('frames_dir', '')}",
            f"- scores_path: {outputs.get('scores_path', '')}",
            f"- best_frames_dir: {outputs.get('best_frames_dir', '')}",
            "",
            "### Reference Boards",
            "",
        ]
    )

    for board in outputs.get("reference_boards", []):
        lines.append(f"- {board}")

    lines.extend(["", "## Best Frames", ""])
    for index, row in enumerate(best_frames, start=1):
        lines.append(
            f"{index}. {row.get('file_name', '')} "
            f"score={row.get('total_score', '')} "
            f"sharpness={row.get('sharpness', '')} "
            f"brightness={row.get('brightness', '')}"
        )

    lines.extend(
        [
            "",
            "## Risk Fields",
            "",
            "- perspective_risk: unknown",
            "- lens_distortion_risk: unknown",
            "- face_shape_confidence: not_estimated",
            "",
            "## Current Evidence Boundary",
            "",
            "- This first version only extracts and scores real video frames.",
            "- It does not infer identity, complete missing face regions, or perform 3D face fitting.",
            "- Candidate full-face generation and local correction are intentionally left for later stages.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
