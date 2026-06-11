---
name: video-face-reference-builder
description: Use when building traceable face-reference assets from a difficult partial-face video. Covers full-video keyframe discovery, deterministic frame scoring, optional CodeFormer/GFPGAN references, AI-curated local face crops, strong feature-pack construction, and imagegen-based full-face review result generation.
---

# Video Face Reference Builder

## Purpose

Build a traceable face-reference package from a difficult video where no single frame contains a complete clean face.

This skill is not only a keyframe selector. Keyframe selection is the first stage. The complete implemented method is:

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
-> review sheet
```

Generated full-face images are review artifacts, not ground truth. The evidence baseline is always original video frames.

## Required References

For the detailed implementation approach, read:

```text
../../docs/implementation_approach.md
```

For the generated-result prompt and exact case reproduction notes, read:

```text
references/feature_driven_assembly.md
```

## Core Rules

- Search the whole video before refining any window.
- Do not hard-code case timestamps, face features, or old manually selected frames.
- Keep original selected keyframes as primary evidence.
- Treat CodeFormer/GFPGAN outputs as auxiliary references only.
- Build a small strong feature pack before image generation; large contact sheets are weak generation references.
- Use the most complete original keyframe as the scaffold, never an AI-generated full face.
- Do not use OpenCV stitching for the final generated result. OpenCV is used for video/frame IO and deterministic scoring.
- Mark missing regions as AI-inferred.

## Inputs

- source video path
- output directory
- optional sampling budget:
  - global sampling interval
  - max global candidates
  - refined window count
  - refined sampling interval
  - final keyframe count
- optional CodeFormer/GFPGAN outputs
- AI visual review notes or JSON when available

## Main Outputs

Keyframe selection:

```text
outputs/keyframe_selection_case/
  global_candidates/
  refined_windows/
  selected_keyframes/
  contact_sheets/
  keyframe_selection.json
  keyframe_selection.md
```

Local references:

```text
outputs/keyframe_selection_case/local_reference/
  ai_curated_crops/
  boards/
```

Generated review result:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/
  strong_feature_reference_pack.jpg
  result_feature_driven.png
  feature_pack_vs_result_review_clear.png
```

## Workflow

### 1. Video Metadata

Use OpenCV to read fps, dimensions, frame count, duration, and codec information.

Code:

```text
src/vfrb/video_info.py
```

### 2. Global Frame Sampling

Sample the full video by time, not by a fixed known window.

Recommended defaults:

```text
duration <= 60s: every 0.5s to 1.0s
duration 60s-5min: every 1.0s to 2.0s
duration > 5min: every 2.0s to 5.0s, then adaptively refine
```

Code:

```text
src/vfrb/extract_frames.py
src/vfrb/keyframe_selection.py
```

### 3. Deterministic Frame Scoring

Compute stable quality signals:

- sharpness via Laplacian variance
- brightness balance
- exposure clipping
- center-region detail
- non-duplicate score where available

Do not rank by full-image sharpness alone. It can favor hands, subtitles, background edges, or compression artifacts.

Code:

```text
src/vfrb/score_frames.py
```

### 4. Contact Sheets

Build visual review sheets for global candidates, refined candidates, and selected keyframes.

Code:

```text
src/vfrb/build_reference_boards.py
src/vfrb/keyframe_selection.py
```

### 5. AI-Assisted Review

Ask a vision-capable model to judge each candidate for face-reference value.

Expected structured fields:

```json
{
  "timestamp": 38.5,
  "face_reference_value": 85,
  "visible_regions": ["mouth", "philtrum", "nose"],
  "problems": ["partial_face", "minor_motion_blur"],
  "reason": "Visible local face evidence with usable shading."
}
```

AI review is one signal. Keep deterministic scores in the report.

### 6. Temporal Refinement

Cluster high-value global candidates into time windows, then resample densely inside those windows.

The final selected keyframes should cover diverse evidence:

- mouth / philtrum
- nose / nasolabial area
- eyes / brow if visible
- cheek / skin tone
- jawline / chin
- face width or side contour

### 7. Optional Enhancement References

Run CodeFormer/GFPGAN only after final original keyframes are selected.

Command:

```bash
.venv/bin/python scripts/summarize_codeformer_case.py
```

Enhanced outputs do not replace original keyframes.

### 8. AI-Curated Local Crops

Build local references for face components.

Commands:

```bash
.venv/bin/python scripts/build_local_reference_case.py
.venv/bin/python scripts/build_ai_curated_crops_case.py
```

Regions:

- face shape
- eye
- nose
- philtrum
- mouth
- cheek/skin

Crop metadata should preserve source frame, region, bbox, quality notes, and offset where available.

### 9. Strong Feature Pack

Build the compact evidence board used for image generation.

Command:

```bash
.venv/bin/python scripts/build_strong_feature_reference_pack_case.py
```

Output:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/strong_feature_reference_pack.jpg
```

### 10. Imagegen Full-Face Review Result

Use `imagegen` with:

- the strong feature reference pack as local organ evidence
- the most complete original keyframe as layout scaffold

The prompt must constrain mouth, nose, philtrum, eyes, cheeks, skin tone, close-camera perspective, and video softness.

Do not ask imagegen for a clean beauty portrait. Ask for a review result that visibly follows the feature pack.

Full prompt record:

```text
references/feature_driven_assembly.md
```

### 11. Review Sheet

Create or keep a side-by-side review sheet comparing:

- left: feature/reference components
- right: generated review result

GitHub case overview:

```text
docs/cases/feature_driven_ai_assembly_20260611/assets/feature_result_overview.png
```

## Run Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_case.py
.venv/bin/python scripts/run_keyframe_selection.py
.venv/bin/python scripts/build_strong_feature_reference_pack_case.py
```

## Case Assets

```text
docs/cases/feature_driven_ai_assembly_20260611/
  assets/feature_result_overview.png
  assets/selected_keyframes.jpg
  assets/primary_scaffold_selected_003.jpg
  assets/strong_feature_reference_pack.jpg
  assets/result_feature_driven.png
  assets/feature_pack_vs_result_review_clear.png
```

## Failure Modes

- If global search finds no useful face evidence, stop and report insufficient evidence.
- If AI and deterministic scoring disagree, keep both candidates and flag the disagreement.
- If all selected frames cluster around one short segment, preserve lower-ranked frames from other segments when they contain distinct evidence.
- If generated output ignores the feature pack, rebuild a smaller stronger feature pack or use localized masked edits instead of accepting the result.
