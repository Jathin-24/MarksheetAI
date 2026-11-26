"""Isolate the teacher's red ink from the printed blue grid and paper, and prepare recogniser inputs."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from config import settings


@dataclass
class Ink:
    mask: np.ndarray                         # uint8 0/255, same size as the crop
    ratio: float                             # ink pixels / crop pixels
    is_blank: bool
    parts: list[tuple[int, int, int, int]] = field(default_factory=list)  # digit bounding boxes, left->right


def redness(crop_bgr: np.ndarray) -> np.ndarray:
    c = crop_bgr.astype(np.float32)
    red = c[..., 2] - np.maximum(c[..., 1], c[..., 0])
    return red - np.median(red)               # removes warm/cool paper tint


def extract_ink(crop_bgr: np.ndarray, seed: float = 20.0, grow: float = 9.0) -> Ink:
    """Hysteresis threshold on 'redness' (thin strokes are faint; paper noise is not red).

    Thresholds adapt to how strong the ink is in this crop, with absolute floors so empty boxes stay empty."""
    red = redness(crop_bgr)
    peak = float(np.percentile(red, 99.8))
    seed, grow = max(seed, 0.40 * peak), max(grow, 0.18 * peak)
    weak = (red > grow).astype(np.uint8)
    strong = (red > seed).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong > 0])] = True
    keep[0] = False
    mask = keep[labels].astype(np.uint8) * 255
    area = mask.size
    # drop specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(8, int(settings.BLANK_MIN_COMPONENT_RATIO * area))
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    # strokes that merely poke in from the top/bottom edge (tails of the digit in the row above/below)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    H = clean.shape[0]
    for i in range(1, n):
        y, h = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        if (y <= 1 or y + h >= H - 1) and h < 0.45 * H:
            clean[labels == i] = 0
    ratio = float((clean > 0).sum()) / area
    blank = ratio < settings.BLANK_INK_RATIO
    return Ink(clean, ratio, blank, [] if blank else digit_parts(clean))


def digit_parts(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Left-to-right glyph boxes; strokes that overlap horizontally belong to the same digit."""
    ys, xs = np.nonzero(mask)
    height = max(1, ys.max() - ys.min() + 1)
    k = max(3, int(0.10 * height)) | 1       # a gap smaller than ~10 % of the digit height is the same digit
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = [(int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3])) for s in stats[1:]]
    if not boxes:
        return []
    tallest = max(b[3] - b[1] for b in boxes)
    boxes = [b for b in boxes if (b[3] - b[1]) >= 0.25 * tallest or (b[2] - b[0]) >= 0.25 * tallest]
    boxes.sort(key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - max(m[0], b[0])
            if overlap > 0.3 * min(m[2] - m[0], b[2] - b[0]):
                m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                continue
        merged.append(list(b))
    return [tuple(m) for m in merged]  # type: ignore[misc]


def glyph_image(mask: np.ndarray, part: tuple[int, int, int, int], size: int = 28, inner: int = 20,
                shift: tuple[int, int] = (0, 0), thicken: int = 0) -> np.ndarray:
    """MNIST-style glyph: aspect preserved, scaled to `inner` px, centred by centre of mass, float 0..1.

    Stroke width is matched to MNIST (~2-3 px at 28 px) from the real pen width, instead of blindly bolding.
    """
    x0, y0, x1, y1 = part
    g = mask[y0:y1, x0:x1]
    dt = cv2.distanceTransform(g, cv2.DIST_L2, 3)
    pen = 2.0 * float(dt[g > 0].mean()) if (g > 0).any() else 1.0
    h, w = g.shape
    s = inner / max(h, w)
    g = cv2.resize(g.astype(np.float32) / 255.0, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_AREA)
    g = (g > 0.12).astype(np.float32)
    if pen * s < 2.0 or thicken:
        g = cv2.dilate(g, np.ones((3, 3), np.uint8), iterations=1 + thicken)
    canvas = np.zeros((size, size), np.float32)
    ys, xs = np.nonzero(g > 0)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (g.shape[0] / 2, g.shape[1] / 2)
    oy, ox = int(round(size / 2 - cy)) + shift[0], int(round(size / 2 - cx)) + shift[1]
    oy, ox = max(0, min(size - g.shape[0], oy)), max(0, min(size - g.shape[1], ox))
    canvas[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    return np.clip(canvas / max(canvas.max(), 1e-6), 0, 1)


def line_image(mask: np.ndarray, pad_ratio: float = 0.35, height: int = 96) -> Image.Image:
    """Black ink on white, tight crop with margin - the style of input TrOCR was trained on."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    g = 255 - mask[y0:y1, x0:x1]
    g = cv2.dilate(255 - g, np.ones((3, 3), np.uint8))
    g = 255 - g
    pad = int(pad_ratio * max(g.shape))
    g = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    s = height / g.shape[0]
    g = cv2.resize(g, (max(8, int(g.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(g).convert("RGB")
