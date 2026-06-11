# Feature-Driven AI Assembly Case - 2026-06-11

This case records a lightweight example of the feature-driven full-face result flow.

It is not a final identity reconstruction. The complete face result contains AI-inferred regions and should only be used for workflow review.

## Case Assets

| Asset | Path | Role |
| --- | --- | --- |
| Overview | `assets/feature_result_overview.png` | Compact GitHub preview with feature pack on the left and generated result on the right |
| Selected keyframes sheet | `assets/selected_keyframes.jpg` | First-stage selected original video frames |
| Primary scaffold | `assets/primary_scaffold_selected_003.jpg` | Most complete original face frame used for layout and face shape |
| Strong feature pack | `assets/strong_feature_reference_pack.jpg` | Hard local references for face shape, eyes, nose, philtrum, mouth, cheek/skin |
| Generated result | `assets/result_feature_driven.png` | Feature-driven full-face AI result |
| Review sheet | `assets/feature_pack_vs_result_review_clear.png` | Clear side-by-side check: feature pack vs generated result |

## Workflow Summary

1. Treat keyframe selection as complete and keep original frames as primary evidence.
2. Use `selected_003_t_00039.000.jpg` as the face-shape and layout scaffold.
3. Build a strong local feature pack from the previously AI-curated slices.
4. Use the `imagegen` skill with a hard-constrained prompt to generate the result.
5. Review whether the result actually preserves mouth, philtrum, nose, eye, cheek, and skin cues from the feature pack.

## Important Constraint

The scaffold must come from the most complete original keyframe. Do not use a previously generated full-face image as the evidence baseline.

## Review Result

The generated result uses the mouth, philtrum, and nose slices more visibly than the previous generic AI attempt. It is still not final:

- the missing eye is AI-inferred;
- hair and full outline are heavily inferred;
- the image still tends toward a clean natural selfie;
- the next stage should use localized masked edits for mouth, nose/philtrum, and eyes.

## Full Prompt Record

See:

```text
../../result_feature_driven_flow.md
```
