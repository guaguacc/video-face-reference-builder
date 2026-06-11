import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vfrb.local_reference import build_local_reference_package


class LocalReferenceTests(unittest.TestCase):
    def test_build_local_reference_package_groups_selected_keyframes_by_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_dir = root / "selected_keyframes"
            selected_dir.mkdir()
            for index in range(1, 3):
                Image.new("RGB", (20, 20), (index * 50, 20, 20)).save(
                    selected_dir / f"selected_{index:03d}_t_0000{index}.000.jpg"
                )

            selection_path = root / "keyframe_selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "selected": [
                            {
                                "timestamp": 1.0,
                                "final_score": 90.0,
                                "visible_regions": ["mouth", "philtrum"],
                                "reason": "mouth reference",
                            },
                            {
                                "timestamp": 2.0,
                                "final_score": 80.0,
                                "visible_regions": ["eye", "cheek"],
                                "reason": "eye reference",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = build_local_reference_package(
                keyframe_selection_path=selection_path,
                selected_keyframes_dir=selected_dir,
                output_dir=root / "local_reference",
            )

            self.assertEqual(summary["selected_keyframe_count"], 2)
            self.assertEqual(summary["regions"]["mouth"]["count"], 1)
            self.assertEqual(summary["regions"]["eye"]["count"], 1)
            self.assertTrue((root / "local_reference" / "reference_boards" / "mouth.jpg").exists())


if __name__ == "__main__":
    unittest.main()
