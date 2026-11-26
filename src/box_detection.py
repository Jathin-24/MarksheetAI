"""Map detected grid cells to the 17 named answer boxes (independent of any OCR)."""
from __future__ import annotations

from dataclasses import dataclass

from .table_detection import TableLayout

PART_A_IDS = ["1a", "1b", "1c", "1d", "1e"]
PART_B_IDS = [[f"{q}A", f"{q}B"] for q in range(2, 8)]
ALL_IDS = PART_A_IDS + [i for pair in PART_B_IDS for i in pair]


@dataclass
class Box:
    id: str
    x: int
    y: int
    w: int
    h: int
    adjusted: bool = False


def boxes_from_layout(layout: TableLayout, tpl: dict) -> dict[str, Box]:
    boxes: dict[str, Box] = {}
    for col, bid in zip(tpl["part_a"]["box_columns"], tpl["part_a"]["ids"]):
        c = layout.part_a_row[col]
        boxes[bid] = Box(bid, c.x, c.y, c.w, c.h)
    lc, rc = tpl["part_b"]["left_columns"], tpl["part_b"]["right_columns"]
    for (q_left, q_right), row in zip(tpl["part_b"]["row_questions"], layout.part_b_rows):
        for q, cols in ((q_left, lc), (q_right, rc)):
            for col, sub in zip(cols, "AB"):
                c = row[col]
                bid = f"{q}{sub}"
                boxes[bid] = Box(bid, c.x, c.y, c.w, c.h)
    return {bid: boxes[bid] for bid in ALL_IDS}
