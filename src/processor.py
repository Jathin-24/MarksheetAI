"""End-to-end processing of one photo (used by the web app, the CLI and the evaluation script)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import settings
from .box_detection import Box
from .crop_extraction import extract_all
from .image_processing import load_image, save_png
from .matrix_generator import build_matrix
from .pipeline_layout import Aligned, align_and_detect
from .recognition_engine import BoxResult, recognise_box
from .validation import validate_boxes

log = logging.getLogger(__name__)


@dataclass
class SheetResult:
    original: np.ndarray
    aligned: Aligned
    crops: dict[str, np.ndarray]
    results: dict[str, BoxResult]
    problems: list[str] = field(default_factory=list)

    def values(self) -> dict[str, int | None]:
        return {k: r.value for k, r in self.results.items()}

    def matrix(self) -> dict:
        return build_matrix(self.values())


def recognise_all(crops: dict[str, np.ndarray], engines: list) -> dict[str, BoxResult]:
    return {bid: recognise_box(bid, c, engines) for bid, c in crops.items()}


def process_image(original: np.ndarray, engines: list, tpl: dict | None = None,
                  force_rotation: int | None = None) -> SheetResult:
    tpl = tpl or settings.load_template()
    aligned = align_and_detect(original, tpl, force_rotation)
    problems = validate_boxes(aligned.boxes, aligned.image.shape)
    crops = extract_all(aligned.image, aligned.boxes)
    return SheetResult(original, aligned, crops, recognise_all(crops, engines), problems)


def process_path(path: str | Path, engines: list) -> SheetResult:
    return process_image(load_image(path), engines)
