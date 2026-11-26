"""Find printed table cells (blue grid) and the two marks tables, in any of 4 orientations."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .image_alignment import rot90_k
from .image_processing import blue_mask, line_mask_red_channel

log = logging.getLogger(__name__)


class LayoutNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class Cell:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class TableLayout:
    rotation_k: int                 # quarter turns (counter-clockwise) applied to the levelled image
    image: np.ndarray               # the oriented, levelled image
    part_a_header: list[Cell]
    part_a_row: list[Cell]          # 7 cells: Q.No, a..e, Total
    part_b_rows: list[list[Cell]]   # 3 rows x 9 cells


def grid_mask(bgr: np.ndarray, pre_dilate: int = 1, open_div: int = 45, gap_div: int = 8,
              mode: str = "hsv") -> np.ndarray:
    """Printed grid lines only: keep long straight runs, then bridge gaps along each direction."""
    mask = blue_mask(bgr) if mode == "hsv" else line_mask_red_channel(bgr)
    if pre_dilate:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=pre_dilate)  # tolerate tilt
    h, w = mask.shape
    short = min(h, w)
    k_open = max(18, short // open_div)          # drops letters, keeps line fragments
    k_gap = max(40, short // gap_div)            # bridges breaks caused by ink / glare
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_open)))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_gap, 3)))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_gap)))
    grid = cv2.bitwise_or(horiz, vert)
    return cv2.dilate(grid, np.ones((3, 3), np.uint8))


def detect_cells(bgr: np.ndarray, **kw) -> list[Cell]:
    grid = grid_mask(bgr, **kw)
    h, w = grid.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(grid), connectivity=4)
    short = min(h, w)
    cells: list[Cell] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw < 0.035 * short or ch < 0.02 * short:
            continue
        if area > 0.12 * h * w or area / float(cw * ch) < 0.72:
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue
        cells.append(Cell(int(x), int(y), int(cw), int(ch)))
    return cells


def cluster_rows(cells: list[Cell]) -> list[list[Cell]]:
    rows: list[list[Cell]] = []
    for c in sorted(cells, key=lambda c: c.cy):
        if rows:
            ref = rows[-1]
            if abs(c.cy - float(np.mean([r.cy for r in ref]))) < 0.45 * float(np.mean([r.h for r in ref])):
                ref.append(c)
                continue
        rows.append([c])
    return [sorted(r, key=lambda c: c.x) for r in rows]


def _extent(row: list[Cell]) -> int:
    return row[-1].x + row[-1].w - row[0].x


GRID_CONFIGS = [dict(pre_dilate=0, open_div=45), dict(pre_dilate=1, open_div=30),
                dict(pre_dilate=0, open_div=30), dict(pre_dilate=0, open_div=70, gap_div=10),
                dict(pre_dilate=0, open_div=45, mode="red"), dict(pre_dilate=1, open_div=30, mode="red"),
                dict(pre_dilate=0, open_div=30, mode="red")]


def find_stacks(bgr: np.ndarray, tpl: dict, **grid_kw):
    """Return (part_a_rows[2], part_b_rows[4]) found by structure alone, or None.

    Cells may be merged where a printed line is faint, so rows are judged by their span and
    by how they stack (2 rows for Part A, 4 rows for Part B, A above B), not by exact cell counts.
    """
    rows = cluster_rows(detect_cells(bgr, **grid_kw))
    cand = [r for r in rows if len(r) >= 3]
    if len(cand) < 6:
        return None
    wmax = max(_extent(r) for r in cand)
    cand = sorted((r for r in cand if _extent(r) >= 0.75 * wmax), key=lambda r: r[0].cy)
    stacks: list[list[list[Cell]]] = []
    for r in cand:
        if stacks:
            prev = stacks[-1][-1]
            pitch = r[0].cy - prev[0].cy
            h = float(np.mean([c.h for c in prev + r]))
            if pitch < 1.6 * h:
                stacks[-1].append(r)
                continue
        stacks.append([r])
    need_b = tpl["part_b"]["data_rows"] + 1
    for a, b in zip(stacks, stacks[1:]):
        if len(a) == 2 and len(b) == need_b:
            return a, b
    return None


def expand_row(row: list[Cell], fractions: list[float], left: int | None = None,
               right: int | None = None, snap: float = 0.02) -> list[Cell]:
    """Rebuild the full set of columns for a row.

    Canonical column edges (fractions of the row width, from template calibration) fill in any
    printed line that the photo lost; real detected edges are preferred whenever they are close.
    """
    left = row[0].x if left is None else left
    right = row[-1].x + row[-1].w if right is None else right
    width = float(right - left)
    top = int(np.median([c.y for c in row]))
    bottom = int(np.median([c.y + c.h for c in row]))
    detected = sorted({c.x for c in row} | {c.x + c.w for c in row})
    edges = []
    for f in fractions:
        e = left + f * width
        near = [d for d in detected if abs(d - e) < snap * width]
        edges.append(int(min(near, key=lambda d: abs(d - e))) if near else int(round(e)))
    return [Cell(edges[i], top, edges[i + 1] - edges[i], bottom - top) for i in range(len(edges) - 1)]


def find_layout_in(bgr: np.ndarray, tpl: dict):
    """Return (part_a_header, part_a_row, part_b_data_rows) if this orientation matches, else None."""
    found = None
    for cfg in GRID_CONFIGS:           # photos differ in sharpness/tilt: try several line filters
        found = find_stacks(bgr, tpl, **cfg)
        if found:
            break
    if not found:
        return None
    a_rows, b_rows = found
    fa, fb = tpl["part_a"]["column_fractions"], tpl["part_b"]["column_fractions"]
    def span(rows):
        return min(r[0].x for r in rows), max(r[-1].x + r[-1].w for r in rows)
    la, ra = span(a_rows)
    lb, rb = span(b_rows)
    a = [expand_row(r, fa, la, ra) for r in a_rows]
    b = [expand_row(r, fb, lb, rb) for r in b_rows]
    return a[0], a[1], b[1:]


def detect_layout(levelled: np.ndarray, tpl: dict, force_k: int | None = None) -> TableLayout:
    """Try the four quarter-turn orientations; keep the one whose grid matches the template."""
    for k in ([force_k % 4] if force_k is not None else [0, 1, 2, 3]):
        img = rot90_k(levelled, k)
        found = find_layout_in(img, tpl)
        if found:
            log.info("layout found at rotation k=%d", k)
            return TableLayout(k, img, *found)
    raise LayoutNotFound(
        "Could not find the Part A / Part B tables. Retake the photo with the whole marks table "
        "visible and in focus, or pick the rotation manually.")
