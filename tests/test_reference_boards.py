import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vfrb.build_reference_boards import build_reference_board


class ReferenceBoardTests(unittest.TestCase):
    def test_build_reference_board_creates_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images = []
            for index in range(3):
                path = tmp_path / f"frame_{index:06d}.jpg"
                Image.new("RGB", (40, 30), (index * 40, 20, 120)).save(path)
                images.append({"path": str(path), "file_name": path.name, "total_score": float(index)})

            output_path = tmp_path / "board.jpg"
            build_reference_board(images, output_path, title="Best Frames", columns=2)

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertGreater(image.width, 40)
                self.assertGreater(image.height, 30)


if __name__ == "__main__":
    unittest.main()
