from pathlib import Path
from typing import Any, Union


PathLike = Union[str, Path]


def summarize_codeformer_results(codeformer_dir: PathLike) -> list[dict[str, Any]]:
    root = Path(codeformer_dir)
    final_dir = root / "final_results"
    restored_dir = root / "restored_faces"
    restored_stems = set()
    if restored_dir.exists():
        for restored in restored_dir.glob("*.png"):
            stem = restored.stem
            if stem.endswith("_00"):
                stem = stem[:-3]
            restored_stems.add(stem)

    rows = []
    for final_image in sorted(final_dir.glob("*.png")) if final_dir.exists() else []:
        enhanced_success = final_image.stem in restored_stems
        rows.append(
            {
                "stem": final_image.stem,
                "final_result_path": str(final_image),
                "enhanced_success": enhanced_success,
                "detected_faces": 1 if enhanced_success else 0,
                "use_for_next_stage": "optional_enhanced_reference" if enhanced_success else "original_keyframe_only",
            }
        )
    return rows
