# Feature-Driven Full-Face Result Workflow

Use this reference when the user asks to reproduce the generated result flow from GitHub.

## Preconditions

Run from the repo root.

The keyframe selection step is already complete when these files exist:

```text
outputs/keyframe_selection_case/keyframe_selection.md
outputs/keyframe_selection_case/contact_sheets/selected_keyframes.jpg
outputs/keyframe_selection_case/selected_keyframes/
```

For the current case, the primary original scaffold is:

```text
outputs/keyframe_selection_case/selected_keyframes/selected_003_t_00039.000.jpg
```

Do not use an AI-generated full face as the scaffold.

## Pipeline

1. Optional auxiliary enhancement summary:

```bash
.venv/bin/python scripts/summarize_codeformer_case.py
```

CodeFormer/GFPGAN outputs are only auxiliary references. They do not replace original keyframes.

2. Build local reference crops and AI-curated crop boards:

```bash
.venv/bin/python scripts/build_local_reference_case.py
.venv/bin/python scripts/build_ai_curated_crops_case.py
```

Important outputs:

```text
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/mouth.jpg
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/philtrum.jpg
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/nose.jpg
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/eye.jpg
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/face_shape.jpg
outputs/keyframe_selection_case/local_reference/ai_curated_crops/boards/accepted_alignment_landmark_review.jpg
```

The AI landmark points are hints, not trusted facts. Bad landmarks must not drive final claims.

3. Build the strong feature reference pack:

```bash
.venv/bin/python scripts/build_strong_feature_reference_pack_case.py
```

Output:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/strong_feature_reference_pack.jpg
```

This feature pack is the critical input for the generated result. It enlarges selected local slices:

- face shape anchor and side
- eye anchor and soft lid
- nose anchor and nostril/side
- philtrum anchor and alternate
- mouth anchor, frontal, and side alternate
- cheek/skin

4. Use the `imagegen` skill to generate the result.

Before calling imagegen, load or view these images so they are visible in context:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/strong_feature_reference_pack.jpg
outputs/keyframe_selection_case/selected_keyframes/selected_003_t_00039.000.jpg
```

Use this prompt:

```text
Use case: compositing
Asset type: full-face reconstruction candidate using hard local feature references

Input images visible in this conversation:
Image 1: strong feature reference pack. This is the primary evidence pack and must drive the organ features. It contains labeled local crops: FACE SHAPE anchor/side, EYE anchor/soft lid, NOSE anchor/nostril side, PHILTRUM anchor/alt, MOUTH anchor/frontal/side alt, and CHEEK/SKIN.
Image 2: selected_003 original keyframe. This is the main face-layout anchor for cheek fullness, close-camera distortion, approximate organ spacing, skin color, hair boundary, and video softness.
Previous candidate 03 visible in conversation: use only as a warning example of what is too generic and too polished. Do not copy its clean selfie style.

Primary request: Generate one complete face reconstruction candidate, but make the local organs visibly derived from Image 1's feature pack. Use the feature pack as hard evidence: the mouth must resemble the MOUTH anchor/frontal crops, the nose must resemble the NOSE anchor/nostril-side crops, the philtrum must keep the short soft transition and bright highlight from the PHILTRUM crops, and the eyes must preserve the narrow soft eyelid style from the EYE crops.

Assembly rule:
- First place the approximate face shape, cheek volume, crop perspective, and hair boundary from Image 2.
- Then insert/fit the local components from Image 1: eye shape, nose, philtrum, mouth, cheek/skin tone.
- Only after placing these components, infer the missing half of the face conservatively so the output becomes a complete face.
- The missing side may be inferred by symmetry and neighboring-frame logic, but it must stay consistent with the feature pack, not with a generic beauty portrait.

Composition:
- Single full-face vertical portrait candidate.
- Complete face visible: both eyes, full nose, philtrum, full mouth, cheeks, chin, and dark hair framing.
- Keep close phone-video perspective, slight off-angle, rounded/full cheek impression, soft low-resolution texture, and mild compression blur.

Hard feature constraints:
- Mouth: modest natural lips, soft pink-red color, slightly V-shaped upper lip contour as seen in the mouth crops; no glossy enlarged lips.
- Nose: small soft nose, low-detail blurred bridge/tip, nostril placement suggested by the nose crops; no sharp sculpted nose.
- Philtrum: short, soft, with the pale vertical highlight/transition visible in the crops; do not replace with a clean model philtrum.
- Eyes: narrow, soft eyelids, dark eye opening, blurred brow/eyelid relationship; no sharp lashes, no large bright eyes.
- Cheeks/skin: rounded cheek, pink flush, smooth phone-video blur; no pore-level detail or studio skin.

Avoid:
- Do not make another clean AI selfie like candidate 03.
- Do not beautify, glamorize, add makeup, sharpen hair, sharpen eyebrows, or create a model-like symmetric face.
- Do not ignore the feature pack. The generated mouth/nose/philtrum/eye shapes should visibly echo the labeled crops.
- No text, labels, landmark dots, collage, borders, UI overlay, or watermark.

Output: one complete full-face candidate reconstruction for review only. The local components should clearly come from the provided crop evidence, while missing regions are AI-inferred conservatively.
```

5. Copy the generated image into:

```text
outputs/keyframe_selection_case/candidate_guided_composite/ai_assembly_round_01/result_feature_driven.png
```

6. Build a review sheet.

The uploaded case includes a clear PNG review sheet:

```text
docs/cases/feature_driven_ai_assembly_20260611/assets/feature_pack_vs_result_review_clear.png
```

If rebuilding locally, use a large TrueType font and PNG output. Small default bitmap fonts plus JPG compression make review text blurry.

## Uploaded Case

The lightweight GitHub case is:

```text
docs/cases/feature_driven_ai_assembly_20260611/
```

It contains:

- selected keyframes sheet
- primary scaffold
- strong feature pack
- generated result
- clear review sheet

## OpenCV Role In This Repo

OpenCV is used in the main process only for base video/frame handling and deterministic scoring:

- video metadata and frame extraction: `cv2.VideoCapture`, `cv2.imwrite`
- deterministic scoring: grayscale conversion and `cv2.Laplacian(...).var()`

OpenCV is not used to generate the result. The result is produced by the `imagegen` skill using the primary scaffold and strong feature pack as references.

Do not add OpenCV stitching steps to the generated-result reproduction flow.
