"""Input and output validation."""
from __future__ import annotations

from pathlib import Path

from config import settings
from .box_detection import ALL_IDS, Box
from .image_processing import ALLOWED_EXT, ImageError


def validate_upload(name: str, size_bytes: int, max_mb: int = 25) -> None:
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise ImageError(f"'{name}': unsupported type. Use JPG, JPEG or PNG.")
    if size_bytes > max_mb * 1024 * 1024:
        raise ImageError(f"'{name}' is larger than {max_mb} MB.")


def validate_mark(value) -> int | None:
    """None stays None (blank). Anything else must be an integer 0..10."""
    if value is None or value == "":
        return None
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{value}' is not a whole number") from exc
    if not settings.MIN_MARK <= v <= settings.MAX_MARK:
        raise ValueError(f"mark {v} is outside {settings.MIN_MARK}-{settings.MAX_MARK}")
    return v


def validate_boxes(boxes: dict[str, Box], image_shape: tuple[int, ...]) -> list[str]:
    """Returns a list of problems (empty list = fine)."""
    problems: list[str] = []
    h, w = image_shape[:2]
    missing = [i for i in ALL_IDS if i not in boxes]
    if missing:
        problems.append(f"missing boxes: {missing}")
    for b in boxes.values():
        if b.w < 20 or b.h < 15:
            problems.append(f"{b.id} is too small")
        if b.x < 0 or b.y < 0 or b.x + b.w > w or b.y + b.h > h:
            problems.append(f"{b.id} is outside the image")
    if not missing:
        for q in range(2, 8):
            a, bb = boxes[f"{q}A"], boxes[f"{q}B"]
            if not a.x < bb.x:
                problems.append(f"{q}A must be left of {q}B")
        if not boxes["1a"].x < boxes["1b"].x < boxes["1c"].x < boxes["1d"].x < boxes["1e"].x:
            problems.append("Part A boxes are not in left-to-right order")
        if not boxes["2A"].y < boxes["4A"].y < boxes["6A"].y:
            problems.append("Part B rows are not in top-to-bottom order")
    return problems
