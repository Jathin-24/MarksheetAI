"""Logical Part A / Part B matrices. No totals, averages or grades are ever computed here."""
from __future__ import annotations

from typing import Mapping

from .box_detection import PART_A_IDS, PART_B_IDS


def build_matrix(values: Mapping[str, int | None], student_id: str | None = None) -> dict:
    """values: box id -> mark or None. Returns the structure required by the spec."""
    return {
        "student_id": student_id,
        "Part_A": [values.get(i) for i in PART_A_IDS],
        "Part_B": [[values.get(a), values.get(b)] for a, b in PART_B_IDS],
    }


def matrix_to_values(matrix: dict) -> dict[str, int | None]:
    out = dict(zip(PART_A_IDS, matrix["Part_A"]))
    for (a, b), row in zip(PART_B_IDS, matrix["Part_B"]):
        out[a], out[b] = row
    return out
