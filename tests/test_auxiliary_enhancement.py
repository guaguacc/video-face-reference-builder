import tempfile
import unittest
from pathlib import Path

from vfrb.auxiliary_enhancement import summarize_codeformer_results


class AuxiliaryEnhancementTests(unittest.TestCase):
    def test_summarize_codeformer_results_keeps_failures_as_original_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "final_results"
            restored_dir = root / "restored_faces"
            final_dir.mkdir()
            restored_dir.mkdir()
            (final_dir / "selected_001_t_00039.000.png").write_text("final", encoding="utf-8")
            (final_dir / "selected_002_t_00048.000.png").write_text("final", encoding="utf-8")
            (restored_dir / "selected_001_t_00039.000_00.png").write_text("restored", encoding="utf-8")

            rows = summarize_codeformer_results(root)

            by_name = {row["stem"]: row for row in rows}
            self.assertEqual(by_name["selected_001_t_00039.000"]["use_for_next_stage"], "optional_enhanced_reference")
            self.assertEqual(by_name["selected_002_t_00048.000"]["use_for_next_stage"], "original_keyframe_only")


if __name__ == "__main__":
    unittest.main()
