"""High-quality crops: lossless PNG, original pixels, border inset only (no resizing of the original)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import settings
from .box_detection import Box
from .image_processing import enhance_colour, enlarge, save_png


def cut(image: np.ndarray, box: Box, inset_ratio: float = settings.CROP_INSET_RATIO) -> np.ndarray:
    ins = max(2, int(round(inset_ratio * min(box.w, box.h))))
    h, w = image.shape[:2]
    x0, y0 = max(0, box.x + ins), max(0, box.y + ins)
    x1, y1 = min(w, box.x + box.w - ins), min(h, box.y + box.h - ins)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"Box {box.id} is too small after trimming; adjust its boundaries.")
    return image[y0:y1, x0:x1].copy()


def extract_all(image: np.ndarray, boxes: dict[str, Box]) -> dict[str, np.ndarray]:
    return {bid: cut(image, b) for bid, b in boxes.items()}


def save_crops(sheet_id: str, crops: dict[str, np.ndarray], out_root: Path = settings.CROP_DIR,
               factors=settings.ENLARGE_FACTORS, method: str = settings.ENLARGE_METHOD) -> dict[str, dict[str, str]]:
    """Write original / enlarged / enhanced PNGs; returns relative file names per box."""
    paths: dict[str, dict[str, str]] = {}
    for bid, crop in crops.items():
        entry = {"original": str(save_png(out_root / "original" / f"{sheet_id}_{bid}.png", crop))}
        for f in factors:
            entry[f"enlarged_{f}x"] = str(save_png(out_root / "enlarged" / f"{sheet_id}_{bid}_{f}x.png",
                                                   enlarge(crop, f, method)))
        entry["enhanced"] = str(save_png(out_root / "enhanced" / f"{sheet_id}_{bid}.png",
                                         enhance_colour(enlarge(crop, 2, method))))
        paths[bid] = entry
    return paths
