"""Deskew + orientation. The photo is resampled once (rotation) and never resized."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .image_processing import blue_mask

log = logging.getLogger(__name__)
MIN_WORKING_SIDE = 1000


@dataclass
class Leveled:
    image: np.ndarray
    skew_deg: float


def estimate_skew_deg(bgr: np.ndarray) -> float:
    """Length-weighted median angle of the long printed table lines, folded into (-45, 45]."""
    mask = blue_mask(bgr)
    h, w = mask.shape
    s = 1200.0 / max(h, w)
    if s < 1:
        mask = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = mask.shape
    k = max(15, min(h, w) // 25)
    long_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    long_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    lines_img = cv2.bitwise_or(long_h, long_v)
    lines = cv2.HoughLinesP(lines_img, 1, np.pi / 720, threshold=60,
                            minLineLength=int(0.12 * min(h, w)), maxLineGap=8)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(((a + 45.0) % 90.0) - 45.0)
        weights.append(np.hypot(x2 - x1, y2 - y1))
    angles, weights = np.array(angles), np.array(weights)
    order = np.argsort(angles)
    cum = np.cumsum(weights[order])
    return float(angles[order][np.searchsorted(cum, cum[-1] / 2.0)])


def rotate_expand(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate counter-clockwise by angle_deg, growing the canvas so nothing is cropped."""
    if abs(angle_deg) < 0.05:
        return bgr.copy()
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(bgr, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def level_image(bgr: np.ndarray) -> Leveled:
    """Make table lines axis-aligned (0/90/180/270 orientation is decided later, losslessly)."""
    short = min(bgr.shape[:2])
    if short < MIN_WORKING_SIDE:   # tiny photos: grid lines are 1 px wide; upscale once so they can be found
        f = MIN_WORKING_SIDE / short
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        log.warning("low-resolution photo upscaled x%.2f - retake at higher resolution if results are poor", f)
    angle = estimate_skew_deg(bgr)
    log.info("estimated skew %.2f deg", angle)
    return Leveled(rotate_expand(bgr, angle), angle)


def rot90_k(bgr: np.ndarray, k: int) -> np.ndarray:
    """Lossless counter-clockwise quarter turns."""
    return np.ascontiguousarray(np.rot90(bgr, k % 4))
