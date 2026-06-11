import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vfrb.extract_frames import extract_frames


class ExtractFramesTests(unittest.TestCase):
    def make_video(self, path: Path, frame_count: int = 10) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_extract_frames_uses_interval_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample.mp4"
            output_dir = tmp_path / "frames"
            self.make_video(video_path, frame_count=10)

            frames = extract_frames(video_path, output_dir, every_n_frames=2, max_frames=3)

            self.assertEqual(len(frames), 3)
            self.assertEqual([frame.path.name for frame in frames], [
                "frame_000001.jpg",
                "frame_000002.jpg",
                "frame_000003.jpg",
            ])
            self.assertEqual([frame.source_frame_index for frame in frames], [0, 2, 4])
            self.assertTrue((output_dir / "frame_000001.jpg").exists())

    def test_extract_frames_rejects_existing_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample.mp4"
            output_dir = tmp_path / "frames"
            output_dir.mkdir()
            (output_dir / "old.jpg").write_text("old", encoding="utf-8")
            self.make_video(video_path)

            with self.assertRaises(FileExistsError):
                extract_frames(video_path, output_dir)

    def test_extract_frames_rejects_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                extract_frames(Path(tmp) / "missing.mp4", Path(tmp) / "frames")


if __name__ == "__main__":
    unittest.main()
