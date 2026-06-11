from pathlib import Path
from typing import Any, Union

from PIL import Image, ImageDraw, ImageOps


PathLike = Union[str, Path]


def build_reference_board(
    rows: list[dict[str, Any]],
    output_path: PathLike,
    title: str,
    columns: int = 4,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if not rows:
        raise ValueError("Cannot build a reference board without images")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    padding = 18
    label_height = 38
    title_height = 46
    columns = max(1, columns)
    rows_count = (len(rows) + columns - 1) // columns
    cell_width = thumb_size[0] + padding
    cell_height = thumb_size[1] + label_height + padding
    board_width = columns * cell_width + padding
    board_height = title_height + rows_count * cell_height + padding

    board = Image.new("RGB", (board_width, board_height), "white")
    draw = ImageDraw.Draw(board)
    draw.text((padding, 14), title, fill=(20, 20, 20))

    for index, row in enumerate(rows):
        image_path = Path(row["path"])
        with Image.open(image_path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), thumb_size)

        col = index % columns
        row_index = index // columns
        x = padding + col * cell_width
        y = title_height + row_index * cell_height
        board.paste(thumb, (x, y))
        label = f"{row.get('file_name', image_path.name)}  score={row.get('total_score', '')}"
        draw.text((x, y + thumb_size[1] + 8), label[:34], fill=(40, 40, 40))

    board.save(output, format="JPEG", quality=92)
