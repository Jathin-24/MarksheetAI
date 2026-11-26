import io
import openpyxl
from src.box_detection import ALL_IDS
from src.evaluation import evaluate
from src.report import class_excel, class_row, student_excel

N = None
V = dict(zip(ALL_IDS, [2, 2, 2, 2, 2, N, N, 4, N, N, N, 4, 4, 4, 4, N, N]))


def test_student_excel_has_numeric_totals():
    ws = openpyxl.load_workbook(io.BytesIO(student_excel(evaluate(V), "R1")))["Summary"]
    vals = {r[0]: r[1] for r in ws.iter_rows(min_row=3, values_only=True)}
    assert vals["TOTAL"] == 20 and isinstance(vals["TOTAL"], (int, float))


def test_class_excel_row():
    e = evaluate(V)
    ws = openpyxl.load_workbook(io.BytesIO(class_excel([class_row("R1", V, e)])))["Class"]
    row = [c.value for c in ws[2]]
    assert row[0] == "R1" and row[-1] == 20 and row[6] == 10
