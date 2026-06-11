# Implementation Approach

This document describes the implemented approach in this repository. It is written as an implementation reference, not as a personal workflow record.

## Scope

The project processes a close-range, partial-face video and builds traceable reference assets:

- selected original keyframes
- optional enhancement comparisons
- local face-region crops
- a strong feature reference pack
- a full-face AI review result
- review sheets for visual auditing

The generated face result is not treated as identity truth. It is a review artifact derived from traceable visual inputs.

## Method Overview

```text
video evidence
-> global candidate frames
-> deterministic scores
-> contact sheets
-> AI-assisted visual review
-> refined temporal windows
-> selected original keyframes
-> optional enhancement references
-> local face-region crops
-> strong feature pack
-> imagegen result
-> review sheet
```

The implementation avoids hard-coded case timestamps. A case can be used as a benchmark, but the selection logic must search the whole video first.

## Step 1: Video Metadata

Code:

- `src/vfrb/video_info.py`

Technology:

- OpenCV `cv2.VideoCapture`

Outputs:

- fps
- frame count
- duration
- width and height
- codec/fourcc where available

Purpose:

Metadata is used to choose sampling strategy and to make reports reproducible.

## Step 2: Global Frame Sampling

Code:

- `src/vfrb/extract_frames.py`
- `src/vfrb/keyframe_selection.py`

Technology:

- OpenCV `cv2.VideoCapture`
- OpenCV `cv2.imwrite`

Typical output:

```text
outputs/keyframe_selection_case/global_candidates/
```

Purpose:

The pipeline first covers the whole video. It does not begin from known good timestamps.

Recommended sampling:

- short videos: every `0.5s-1.0s`
- medium videos: every `1.0s-2.0s`
- long videos: coarse sampling first, then adaptive refinement

## Step 3: Deterministic Frame Scoring

Code:

- `src/vfrb/score_frames.py`

Technology:

- OpenCV `cv2.cvtColor`
- OpenCV `cv2.Laplacian(...).var()`

Signals:

- sharpness
- brightness
- exposure penalties
- center-region information
- placeholder fields for later perspective and region-risk scoring

Purpose:

Deterministic scoring provides stable ranking and filtering. It is not the only decision signal because full-image sharpness can favor background edges, subtitles, hands, or compression artifacts.

## Step 4: Contact Sheets

Code:

- `src/vfrb/build_reference_boards.py`
- `src/vfrb/keyframe_selection.py`

Technology:

- Pillow image composition

Typical outputs:

```text
outputs/keyframe_selection_case/contact_sheets/global_candidates.jpg
outputs/keyframe_selection_case/contact_sheets/selected_keyframes.jpg
```

Purpose:

Contact sheets make global review practical and preserve the evidence behind selected frames.

## Step 5: AI-Assisted Visual Review

Input:

- global and refined contact sheets
- deterministic score tables

Expected review fields:

```json
{
  "timestamp": 38.5,
  "face_reference_value": 85,
  "visible_regions": ["mouth", "philtrum", "nose"],
  "problems": ["partial_face", "minor_motion_blur"],
  "reason": "Visible local face evidence with usable shading."
}
```

Purpose:

AI review adds semantic judgment: which face areas are visible, whether the frame contains usable identity reference, and what should be rejected.

## Step 6: Temporal Window Refinement

Code:

- `src/vfrb/keyframe_selection.py`

Purpose:

High-value global candidates are grouped into time windows. Those windows are sampled more densely, usually around `0.1s-0.25s`, to find the strongest local evidence.

The final keyframes should cover diverse evidence rather than only the highest-scoring frames.

## Step 7: Selected Original Keyframes

Typical outputs:

```text
outputs/keyframe_selection_case/selected_keyframes/
outputs/keyframe_selection_case/keyframe_selection.json
outputs/keyframe_selection_case/keyframe_selection.md
```

Rule:

Selected keyframes are original video frames. They are the primary evidence base.

## Step 8: Optional Enhancement References

Code:

- `src/vfrb/auxiliary_enhancement.py`
- `scripts/summarize_codeformer_case.py`

Technology:

- CodeFormer/GFPGAN, when available locally

Rule:

Enhancement is auxiliary only. Enhanced images do not replace original keyframes, and failed enhancement does not invalidate the original evidence.

## Step 9: AI-Curated Local Face Crops

Code:

- `src/vfrb/local_reference.py`
- `scripts/build_local_reference_case.py`
- `scripts/build_ai_curated_crops_case.py`

Regions:

- face shape
- eye
- nose
- philtrum
- mouth
- cheek/skin

Purpose:

Local crops isolate the useful parts of partial frames. Large contact sheets are too weak as image-generation references, so the implementation builds smaller, stronger region references.

## Step 10: Strong Feature Reference Pack

Code:

- `scripts/build_strong_feature_reference_pack_case.py`

Typical output:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/strong_feature_reference_pack.jpg
```

Content:

- face-shape anchors
- eye crops
- nose crops
- philtrum crops
- mouth crops
- cheek/skin crop

Purpose:

The feature pack makes the local organs visually dominant before AI generation. It is the key reference used for the generated result.

## Step 11: Feature-Driven AI Result

Technology:

- Codex `imagegen` skill

Inputs:

- the strongest original keyframe as the layout scaffold
- the strong feature reference pack as hard local evidence

Primary rule:

Use the original frame for face layout, skin tone, close-camera perspective, and hair boundary. Use the feature pack to constrain mouth, nose, philtrum, eyes, and cheek/skin cues.

Output:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/result_feature_driven.png
```

OpenCV is not used to assemble this result.

## Step 12: Review Sheet

Typical output:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/feature_pack_vs_result_review_clear.png
```

Purpose:

The review sheet compares the strong feature pack against the generated result so local feature preservation can be visually audited.

## Current Case Assets

GitHub case:

```text
docs/cases/feature_driven_ai_assembly_20260611/
```

Assets:

- `assets/selected_keyframes.jpg`
- `assets/primary_scaffold_selected_003.jpg`
- `assets/strong_feature_reference_pack.jpg`
- `assets/result_feature_driven.png`
- `assets/feature_pack_vs_result_review_clear.png`

The full prompt record is stored in:

```text
docs/result_feature_driven_flow.md
```

## Project Skill

The repository includes one project skill:

```text
skills/video-face-reference-builder/SKILL.md
```

It records the reusable instructions for keyframe selection and feature-driven result generation. The detailed generated-result prompt is stored in:

```text
skills/video-face-reference-builder/references/feature_driven_assembly.md
```

## Boundaries

- Do not use generated full-face images as evidence baselines.
- Do not mix enhanced images into keyframe-selection conclusions.
- Do not hard-code the current case timestamps or face features.
- Mark missing regions as AI-inferred.
- Treat the generated face as a review result, not a verified identity reconstruction.
