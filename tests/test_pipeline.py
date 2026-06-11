import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vfrb.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def make_video(self, path: Path, frame_count: int = 8) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_run_pipeline_creates_first_version_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample.mp4"
            output_root = tmp_path / "outputs"
            self.make_video(video_path)

            result = run_pipeline(
                video_path=video_path,
                output_root=output_root,
                run_id="test-run",
                every_n_frames=2,
                max_frames=4,
                best_frame_count=3,
            )

            run_dir = output_root / "test-run"
            self.assertEqual(result.run_dir, run_dir)
            self.assertTrue((run_dir / "video_info.json").exists())
            self.assertTrue((run_dir / "frame_scores.json").exists())
            self.assertTrue((run_dir / "reference_boards" / "best_frames.jpg").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertEqual(len(list((run_dir / "frames").glob("*.jpg"))), 4)
            self.assertEqual(len(list((run_dir / "best_frames").glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
