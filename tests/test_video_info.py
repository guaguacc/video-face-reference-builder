import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vfrb.video_info import read_video_info, write_video_info


class VideoInfoTests(unittest.TestCase):
    def make_video(self, path: Path, frame_count: int = 6) -> None:
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

    def test_read_video_info_returns_basic_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "sample.mp4"
            self.make_video(video_path, frame_count=6)

            info = read_video_info(video_path)

            self.assertEqual(info["file_name"], "sample.mp4")
            self.assertEqual(info["width"], 64)
            self.assertEqual(info["height"], 48)
            self.assertEqual(info["frame_count"], 6)
            self.assertAlmostEqual(info["fps"], 10.0, places=1)
            self.assertGreater(info["duration_seconds"], 0)

    def test_write_video_info_creates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "sample.mp4"
            output_path = Path(tmp) / "video_info.json"
            self.make_video(video_path, frame_count=3)

            info = read_video_info(video_path)
            write_video_info(info, output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["file_name"], "sample.mp4")

    def test_read_video_info_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_video_info("/missing/video.mp4")


if __name__ == "__main__":
    unittest.main()
