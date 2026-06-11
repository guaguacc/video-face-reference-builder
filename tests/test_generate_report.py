import tempfile
import unittest
from pathlib import Path

from vfrb.generate_report import generate_report


class GenerateReportTests(unittest.TestCase):
    def test_generate_report_writes_markdown_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "report.md"
            video_info = {
                "file_name": "sample.mp4",
                "width": 64,
                "height": 48,
                "fps": 10.0,
                "frame_count": 12,
                "duration_seconds": 1.2,
                "codec": "mp4v",
            }
            params = {
                "every_n_frames": 2,
                "max_frames": 3,
                "best_frame_count": 2,
            }
            best_frames = [
                {"file_name": "frame_000001.jpg", "total_score": 90.0},
                {"file_name": "frame_000002.jpg", "total_score": 80.0},
            ]
            outputs = {
                "frames_dir": "outputs/run/frames",
                "scores_path": "outputs/run/frame_scores.json",
                "reference_boards": ["outputs/run/reference_boards/best_frames.jpg"],
            }

            generate_report(video_info, params, best_frames, outputs, output_path)

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("# Video Face Reference Report", text)
            self.assertIn("sample.mp4", text)
            self.assertIn("frame_000001.jpg", text)
            self.assertIn("perspective_risk", text)
            self.assertIn("face_shape_confidence", text)


if __name__ == "__main__":
    unittest.main()
