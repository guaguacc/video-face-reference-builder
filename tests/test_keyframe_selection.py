import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vfrb.keyframe_selection import (
    Candidate,
    cluster_timestamps,
    combine_scores,
    rank_windows_by_candidate_score,
    refine_window_timestamps,
    select_diverse_candidates,
    timestamps_for_global_sweep,
    run_keyframe_selection,
)


class KeyframeSelectionTests(unittest.TestCase):
    def test_timestamps_for_global_sweep_covers_full_video(self):
        timestamps = timestamps_for_global_sweep(duration_seconds=61.2, step_seconds=1.0)

        self.assertEqual(timestamps[0], 0.0)
        self.assertIn(38.0, timestamps)
        self.assertIn(48.0, timestamps)
        self.assertGreaterEqual(timestamps[-1], 61.0)

    def test_combine_scores_uses_weights(self):
        score = combine_scores(
            deterministic_score=80.0,
            ai_reference_value=60.0,
            diversity_bonus=20.0,
        )

        self.assertEqual(score, 67.0)

    def test_cluster_timestamps_groups_nearby_candidates(self):
        windows = cluster_timestamps([38.0, 38.5, 39.0, 48.0, 48.5], max_gap_seconds=1.0, padding_seconds=0.5)

        self.assertEqual(windows, [(37.5, 39.5), (47.5, 49.0)])

    def test_refine_window_timestamps_uses_time_step(self):
        timestamps = refine_window_timestamps(38.0, 38.5, step_seconds=0.25)

        self.assertEqual(timestamps, [38.0, 38.25, 38.5])

    def test_select_diverse_candidates_keeps_region_variety(self):
        candidates = [
            Candidate(timestamp=1.0, path="a.jpg", final_score=95.0, visible_regions=["mouth"]),
            Candidate(timestamp=2.0, path="b.jpg", final_score=92.0, visible_regions=["mouth"]),
            Candidate(timestamp=3.0, path="c.jpg", final_score=80.0, visible_regions=["jawline"]),
            Candidate(timestamp=4.0, path="d.jpg", final_score=70.0, visible_regions=["nose"]),
        ]

        selected = select_diverse_candidates(candidates, limit=3)

        regions = [candidate.visible_regions[0] for candidate in selected]
        self.assertEqual(regions, ["mouth", "jawline", "nose"])

    def test_rank_windows_by_candidate_score_prefers_high_value_later_window(self):
        windows = [(0.0, 4.0), (38.0, 40.0)]
        candidates = [
            Candidate(timestamp=1.0, path="early.jpg", final_score=40.0, visible_regions=["eye"]),
            Candidate(timestamp=39.0, path="late.jpg", final_score=80.0, visible_regions=["philtrum"]),
        ]

        ranked = rank_windows_by_candidate_score(windows, candidates)

        self.assertEqual(ranked[0], (38.0, 40.0))

    def test_run_keyframe_selection_uses_ai_judgement_to_rank_later_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample.mp4"
            output_dir = tmp_path / "keyframes"
            judgement_path = tmp_path / "ai.json"

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                2.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(8):
                frame = np.full((48, 64, 3), 40 + index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            judgement_path.write_text(json.dumps({
                "judgements": [
                    {
                        "timestamp": 3.0,
                        "face_reference_value": 95,
                        "visible_regions": ["mouth"],
                        "reason": "later frame is more useful"
                    }
                ]
            }), encoding="utf-8")

            result = run_keyframe_selection(
                video_path=video_path,
                output_dir=output_dir,
                ai_judgement_path=judgement_path,
                global_step_seconds=1.0,
                top_global_count=4,
                max_refined_windows=1,
                final_count=2,
            )

            self.assertGreaterEqual(result["windows"][0]["start"], 2.0)

if __name__ == "__main__":
    unittest.main()
