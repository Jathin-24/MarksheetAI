"""Orchestrates levelling -> orientation -> 17 boxes, with progressively more tolerant retries."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .box_detection import Box, boxes_from_layout
from .image_alignment import MIN_WORKING_SIDE, estimate_skew_deg, rotate_expand
from .table_detection import LayoutNotFound, TableLayout, detect_layout

import cv2

log = logging.getLogger(__name__)
EXTRA_ANGLES = (0.0, 1.0, -1.0, 2.0, -2.0, 3.5, -3.5, 5.0, -5.0)


@dataclass
class Aligned:
    image: np.ndarray            # levelled + oriented sheet (what crops are cut from)
    boxes: dict[str, Box]
    skew_deg: float
    rotation_k: int


def align_and_detect(original: np.ndarray, tpl: dict, force_k: int | None = None) -> Aligned:
    """Raises LayoutNotFound if no orientation/skew produces the template grid."""
    img = original
    short = min(img.shape[:2])
    if short < MIN_WORKING_SIDE:
        f = MIN_WORKING_SIDE / short
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        log.warning("low-resolution photo upscaled x%.2f", f)
    base = estimate_skew_deg(img)
    last: Exception | None = None
    for extra in EXTRA_ANGLES:
        angle = base + extra
        try:
            layout: TableLayout = detect_layout(rotate_expand(img, angle), tpl, force_k)
            return Aligned(layout.image, boxes_from_layout(layout, tpl), angle, layout.rotation_k)
        except LayoutNotFound as exc:
            last = exc
    raise LayoutNotFound(str(last))
