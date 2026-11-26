"""Tables + Excel/CSV reports for one student and for the whole class."""
from __future__ import annotations

import io

import pandas as pd

from config import settings
from .evaluation import PART_B_MAX, TOTAL_MAX, Evaluation, fmt


def tables(ev: Evaluation) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    each = settings.PART_A_MAX / 5
    a = pd.DataFrame([{"Question": n, "Marks": fmt(v), "Out of": f"{each:g}"} for n, v in ev.part_a]
                     + [{"Question": "Part A total", "Marks": fmt(ev.part_a_total), "Out of": f"{settings.PART_A_MAX}"}])
    rows = []
    for p in ev.pairs:
        for s in (p.left, p.right):
            rows.append({"Question": f"Q{s.q}", "a": fmt(s.a), "b": fmt(s.b),
                         "a+b": fmt(s.total) if s.attempted else "–", "Counted": "✔" if s.counted else ""})
    rows.append({"Question": "Part B (3 best of pairs)", "a": "", "b": "", "a+b": f"{ev.part_b_raw}", "Counted": "of 30"})
    b = pd.DataFrame(rows)
    s = pd.DataFrame([
        {"": "Part A", "Marks": fmt(ev.part_a_total), "Out of": f"{settings.PART_A_MAX}"},
        {"": "Part B (30 ÷ 2)", "Marks": fmt(ev.part_b), "Out of": f"{PART_B_MAX:g}"},
        {"": "TOTAL", "Marks": fmt(ev.total), "Out of": f"{TOTAL_MAX:g}"}])
    return a, b, s


def student_excel(ev: Evaluation, student: str, matrix_meta: list[dict] | None = None) -> bytes:
    """Numbers are written as real numbers so the teacher can use them in formulas."""
    each = settings.PART_A_MAX / 5
    a = pd.DataFrame([{"Question": n, "Marks": v, "Out of": each} for n, v in ev.part_a]
                     + [{"Question": "Part A total", "Marks": ev.part_a_total, "Out of": settings.PART_A_MAX}])
    rows = []
    for p in ev.pairs:
        for q in (p.left, p.right):
            rows.append({"Question": f"Q{q.q}", "a": q.a, "b": q.b, "a+b": q.total if q.attempted else None,
                         "Counted": "yes" if q.counted else ""})
    b = pd.DataFrame(rows)
    s = pd.DataFrame([{"": "Part A", "Marks": ev.part_a_total, "Out of": settings.PART_A_MAX},
                      {"": "Part B (30 / 2)", "Marks": ev.part_b, "Out of": PART_B_MAX},
                      {"": "TOTAL", "Marks": ev.total, "Out of": TOTAL_MAX}])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        s.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        xw.sheets["Summary"]["A1"] = f"Student: {student or '-'}"
        a.to_excel(xw, sheet_name="Part A", index=False)
        b.to_excel(xw, sheet_name="Part B", index=False)
        if ev.notes:
            pd.DataFrame({"Notes": ev.notes}).to_excel(xw, sheet_name="Notes", index=False)
        if matrix_meta:
            pd.DataFrame(matrix_meta).to_excel(xw, sheet_name="Extraction metadata", index=False)
        for ws in xw.sheets.values():
            ws.column_dimensions["A"].width = 26
    return buf.getvalue()


CLASS_COLS = ["Student", "1a", "1b", "1c", "1d", "1e", "Part A /10", "Q counted 1", "Q counted 2", "Q counted 3",
              "Part B raw /30", "Part B /15", "Total /25"]


def class_row(student: str, values: dict, ev: Evaluation) -> dict:
    counted = [f"Q{p.counted_q}" if p.counted_q else "-" for p in ev.pairs]
    return {"Student": student, **{n: v for n, v in ev.part_a}, "Part A /10": ev.part_a_total,
            "Q counted 1": counted[0], "Q counted 2": counted[1], "Q counted 3": counted[2],
            "Part B raw /30": ev.part_b_raw, "Part B /15": ev.part_b, "Total /25": ev.total}


def class_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CLASS_COLS)


def class_excel(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        class_frame(rows).to_excel(xw, sheet_name="Class", index=False)
        for col in xw.sheets["Class"].column_dimensions.values():
            col.width = 14
    return buf.getvalue()
