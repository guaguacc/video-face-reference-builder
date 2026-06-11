from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    source_frame_index: int


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    every_n_frames: int = 30,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 95,
) -> list[ExtractedFrame]:
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")

    extracted: list[ExtractedFrame] = []
    source_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_index % every_n_frames == 0:
                output_path = destination / f"frame_{len(extracted) + 1:06d}.jpg"
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not written:
                    raise ValueError(f"Could not write frame: {output_path}")
                extracted.append(ExtractedFrame(output_path, source_index))

                if max_frames is not None and len(extracted) >= max_frames:
                    break

            source_index += 1
    finally:
        capture.release()

    return extracted
