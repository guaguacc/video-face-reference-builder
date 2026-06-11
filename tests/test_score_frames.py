import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vfrb.score_frames import score_frame, score_frames, select_best_frames


class ScoreFramesTests(unittest.TestCase):
    def write_image(self, path: Path, image: np.ndarray) -> None:
        ok = cv2.imwrite(str(path), image)
        self.assertTrue(ok)

    def test_score_frame_reports_sharpness_and_brightness(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sharp.jpg"
            image = np.zeros((80, 80, 3), dtype=np.uint8)
            image[:, 40:] = 255
            self.write_image(image_path, image)

            result = score_frame(image_path)

            self.assertEqual(result["file_name"], "sharp.jpg")
            self.assertGreater(result["sharpness"], 0)
            self.assertGreaterEqual(result["brightness"], 0)
            self.assertLessEqual(result["brightness"], 255)
            self.assertIn("total_score", result)

    def test_select_best_frames_orders_by_total_score(self):
        rows = [
            {"path": "a.jpg", "total_score": 1.0},
            {"path": "b.jpg", "total_score": 3.0},
            {"path": "c.jpg", "total_score": 2.0},
        ]

        best = select_best_frames(rows, limit=2)

        self.assertEqual([row["path"] for row in best], ["b.jpg", "c.jpg"])

    def test_score_frames_writes_scores_for_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp) / "frames"
            frame_dir.mkdir()
            self.write_image(frame_dir / "frame_000001.jpg", np.zeros((20, 20, 3), dtype=np.uint8))
            self.write_image(frame_dir / "frame_000002.jpg", np.full((20, 20, 3), 128, dtype=np.uint8))

            scores = score_frames(frame_dir)

            self.assertEqual(len(scores), 2)
            self.assertEqual(scores[0]["file_name"], "frame_000001.jpg")


if __name__ == "__main__":
    unittest.main()
