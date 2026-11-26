"""JSON / CSV / Excel / ZIP exports. Metadata is kept apart from the marks matrix."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .box_detection import PART_A_IDS, PART_B_IDS


def matrix_json(matrix: dict) -> str:
    return json.dumps(matrix, indent=2)


def matrix_csv(matrix: dict) -> str:
    """Long, spreadsheet-friendly layout: one line per box, in logical order."""
    rows = [("Part A", i, v) for i, v in zip(PART_A_IDS, matrix["Part_A"])]
    for (a, b), (va, vb) in zip(PART_B_IDS, matrix["Part_B"]):
        rows += [("Part B", a, va), ("Part B", b, vb)]
    return pd.DataFrame(rows, columns=["part", "box", "mark"]).to_csv(index=False)


def matrix_excel(matrix: dict, metadata: list[dict] | None = None) -> bytes:
    """Sheet 'Marks' has Part A as a row and Part B as a 6x2 grid; blanks stay empty cells."""
    buf = io.BytesIO()
    a = pd.DataFrame([matrix["Part_A"]], columns=PART_A_IDS, index=["Part A"])
    b = pd.DataFrame(matrix["Part_B"], columns=["A", "B"], index=[f"Q{q}" for q in range(2, 8)])
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        a.to_excel(xw, sheet_name="Marks", startrow=1)
        b.to_excel(xw, sheet_name="Marks", startrow=5)
        ws = xw.sheets["Marks"]
        ws["A1"], ws["A5"] = "Part A (1a-1e)", "Part B (Q2-Q7, sub-parts A and B)"
        ws["A14"] = f"student_id: {matrix.get('student_id') or ''}"
        for col in "ABCDEFG":
            ws.column_dimensions[col].width = 9
        if metadata:
            pd.DataFrame(metadata).to_excel(xw, sheet_name="Extraction metadata", index=False)
    return buf.getvalue()


def crops_zip(files: list[Path], extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=Path(f).name)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def metadata_records(results, boxes, crop_paths: dict[str, dict[str, str]], verified: dict[str, str]) -> list[dict]:
    recs = []
    for bid, r in results.items():
        b = boxes[bid]
        recs.append({
            "box_id": bid,
            "original_crop": Path(crop_paths.get(bid, {}).get("original", "")).name,
            "predicted_mark": r.value,
            "alternative_predictions": "; ".join(f"{v} ({p:.2f})" for v, p in r.alternatives),
            "recognition_engines": ", ".join(f"{k}={v}" for k, v in r.engines.items()),
            "preprocessing": r.preprocessing,
            "confidence": None if r.confidence is None else round(r.confidence, 3),
            "status": r.status,
            "verification": verified.get(bid, "not reviewed"),
            "flags": "; ".join(r.reasons),
            "x": b.x, "y": b.y, "w": b.w, "h": b.h,
        })
    return recs
