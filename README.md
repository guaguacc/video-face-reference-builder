# Video Face Reference Builder

An implementation for building traceable face-reference assets from a difficult, close-range, partial-face video.

The project keeps original video frames as the evidence base. AI-generated images are review artifacts, not ground truth.

## Implemented Method

```text
video
-> global frame sampling
-> deterministic frame scoring
-> contact sheets
-> AI-assisted keyframe review
-> dense sampling around useful windows
-> selected original keyframes
-> optional CodeFormer/GFPGAN reference comparison
-> AI-curated local face crops
-> strong feature reference pack
-> imagegen full-face review result
```

OpenCV is used for video/frame IO and deterministic scoring. It is not used to stitch the final face result.

## Example Result

![Feature-driven face result](docs/cases/feature_driven_ai_assembly_20260611/assets/result_feature_driven.png)

Case assets:

- [Selected keyframes](docs/cases/feature_driven_ai_assembly_20260611/assets/selected_keyframes.jpg)
- [Primary original scaffold](docs/cases/feature_driven_ai_assembly_20260611/assets/primary_scaffold_selected_003.jpg)
- [Strong feature pack](docs/cases/feature_driven_ai_assembly_20260611/assets/strong_feature_reference_pack.jpg)
- [Review sheet](docs/cases/feature_driven_ai_assembly_20260611/assets/feature_pack_vs_result_review_clear.png)

## Documentation

- [Implementation approach](docs/implementation_approach.md)
- [Generated result prompt record](docs/result_feature_driven_flow.md)
- [Case notes](docs/cases/feature_driven_ai_assembly_20260611/README.md)
- [Project skill](skills/video-face-reference-builder/SKILL.md)

## Run

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_case.py
.venv/bin/python scripts/run_keyframe_selection.py
.venv/bin/python scripts/build_strong_feature_reference_pack_case.py
```

Local videos, generated outputs, model weights, and `.venv/` are not committed.
