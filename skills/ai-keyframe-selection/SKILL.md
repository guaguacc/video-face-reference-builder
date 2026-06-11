---
name: ai-keyframe-selection
description: Use when selecting face-reference keyframes from a video, especially when replacing manual visual review with an AI-assisted process that must search the whole video first, then refine promising time windows without losing global coverage.
---

# AI Keyframe Selection

## Purpose

Select face-reference keyframes from a difficult video where no single frame contains a complete, clean face.

This skill replaces manual review with an AI-assisted workflow. It must not hard-code known good timestamps from a prior case. It must search the full video first, then refine promising time windows.

## Core Rule

Never start from a fixed time window alone.

Always run:

```text
global sweep -> candidate scoring -> temporal clustering -> local window refinement -> final diverse keyframe set
```

The final output should include both:

- globally discovered candidate frames
- refined frames from high-value windows around those candidates

## Inputs

- source video path
- output directory
- target use: face reference, mouth reference, face shape, or general person identity reference
- optional budget:
  - max global preview frames
  - max refined windows
  - frames per refined window
  - final keyframe count

## Outputs

Create:

```text
keyframe_selection/
  global_candidates/
  refined_windows/
  selected_keyframes/
  contact_sheets/
  keyframe_selection.json
  keyframe_selection.md
```

The report must include:

- video duration, fps, frame count
- global sampling strategy
- candidate timestamps
- selected refinement windows
- final selected timestamps
- why each final keyframe was selected
- what face evidence it contains
- which parts are uncertain or unusable

## Workflow

### 1. Global Sweep

Sample the entire video at a coarse but coverage-preserving interval.

Recommended default:

```text
duration <= 60s: every 0.5s to 1.0s
duration 60s-5min: every 1.0s to 2.0s
duration > 5min: every 2.0s to 5.0s, then adaptively refine
```

Do not use a single fixed frame interval such as every 30 frames unless it is converted from a time-based sampling decision.

Save global candidates with timestamped names:

```text
global_candidates/t_00038.500.jpg
```

### 2. Automated Frame Scoring

For each global candidate, compute deterministic scores:

- sharpness: Laplacian variance
- brightness balance
- exposure clipping
- motion blur proxy
- center-region detail
- skin-tone area proxy
- face/partial-face detector confidence when available
- non-duplicate score against nearby frames

Do not rank by full-image sharpness alone. Full-image sharpness often favors background edges, subtitles, hands, or high-contrast artifacts instead of useful face evidence.

### 3. AI Visual Judging

Build contact sheets from global candidates.

Ask a vision-capable model to judge each candidate for face-reference value. The prompt should ask for structured output:

```json
{
  "timestamp": 38.5,
  "face_reference_value": 0,
  "visible_regions": ["mouth", "philtrum", "nose"],
  "problems": ["motion_blur", "cropped_face"],
  "reason": "Upper lip and philtrum visible with usable shading."
}
```

Use AI judgment as one signal, not the only signal. Preserve deterministic scores in the report.

### 4. Temporal Clustering

Group high-scoring global candidates into time windows.

Example:

```text
38.0s, 38.5s, 39.0s, 39.5s -> window 37.5s-40.0s
48.0s, 48.5s, 49.0s, 49.5s, 50.0s -> window 47.5s-50.5s
```

Windows are discovered from global evidence. They are not assumed before global search.

Keep at least a few singleton candidates outside the top windows if they contain distinct face evidence, such as side face, jawline, forehead, or skin tone.

### 5. Local Window Refinement

For each selected window, resample more densely by time.

Recommended default:

```text
0.1s to 0.25s step inside the window
```

Then rescore and ask the AI judge again on local contact sheets.

Local refinement should improve frame choice within a discovered window; it must not erase the global search results.

### 6. Diversity Selection

Final keyframes should cover different evidence, not just the highest total score.

Prefer a diverse set:

- mouth / philtrum
- nose / nasolabial area
- eyes / brow if visible
- cheek / skin tone
- jawline / chin
- face width or side contour
- best overall identity reference

If a region has no usable evidence, mark it missing.

### 7. Final Report

Write `keyframe_selection.md` with:

- selected frames table
- timestamp
- source stage: global or refined
- visible regions
- score summary
- AI reason
- deterministic score summary
- limitations

The report must state when the final frames came from a globally discovered window, and when they came from non-window singleton candidates.

## Failure Modes

If AI and deterministic scoring disagree, keep both candidates and flag the disagreement.

If no useful face regions are found globally, stop and report that the video lacks usable evidence instead of forcing a fake selection.

If the best windows are all near one short segment, keep some lower-ranked global candidates from other segments for context unless they are visually unusable.

## Case Benchmark Guidance

For the existing example case, the old manually selected frames around `38s-39.5s` and `48s-50s` may be used only as benchmark references.

Do not hard-code those timestamps into the selection logic.

The algorithm should be evaluated by whether it discovers those windows, or finds better evidence elsewhere in the full video.

## Worked Benchmark: Current Face Video Case

Use this benchmark to calibrate behavior, not to constrain future videos.

### What happened manually

In the current project case, manual review first looked across the full video, then selected two high-value time regions:

```text
38.0s, 38.5s, 39.0s, 39.5s
48.0s, 48.5s, 49.0s, 49.5s, 50.0s
```

These were not model-selected timestamps. They were manually chosen after visual global review, then extracted by script and enhanced with CodeFormer.

### Why those frames were useful

The useful evidence was local, not full-face:

- `38.0s-40.0s`: cheek, mouth, lower nose, philtrum, near-face color
- `47.0s-49.0s`: nose, philtrum, mouth area, cheek, eye-side context
- `48.0s-48.5s`: strongest local reference for philtrum/nose/mouth shading

The frames are still imperfect:

- close perspective distortion
- partial face only
- motion blur
- no complete face shape
- CodeFormer detects only some frames because the face is heavily cropped

### What a good automatic run should do

A good run on this case should:

- sample the whole video first
- produce full timeline contact sheets, not only top-score sheets
- avoid selecting early hand-occlusion frames just because they have high edge sharpness
- identify windows near `38s-40s` and `47s-50s` from global evidence
- keep secondary candidates only if they add distinct evidence, such as mouth, cheek, glasses, or side contour
- run CodeFormer only after candidate selection
- record CodeFormer failures instead of hiding them

### What a bad automatic run looks like

This case exposed several failure modes:

- Ranking only by full-image sharpness selected early `0s-19s` frames where the arm occluded the face.
- Building contact sheets only from algorithm top frames hid useful later windows.
- Selecting windows by chronological order caused early weak windows to displace better later windows.
- Giving unlabeled frames a neutral AI score allowed occluded frames to survive final selection.

The skill should avoid these failures by:

- generating timeline contact sheets for global AI review
- letting AI reference-value scores affect window ranking
- ranking windows by candidate score, not chronological order
- lowering default AI score for unlabeled candidates when an AI judgement file is being used

### CodeFormer behavior in this case

On the selected local-face candidates, CodeFormer detected faces in only a small subset of frames.

This is expected for very close, partial, or cropped faces. The pipeline should treat CodeFormer output as an enhancement attempt, not proof that the frame is valid.

Reports should include:

- number of frames sent to CodeFormer
- number of detected/restored faces
- frames with no detected face
- original-to-enhanced mapping
- warning when most candidates are partial-face crops
