from src.matrix_generator import build_matrix, matrix_to_values
from src.box_detection import ALL_IDS


def test_shape_and_order():
    vals = {i: n for n, i in enumerate(ALL_IDS)}
    m = build_matrix(vals)
    assert m["Part_A"] == [0, 1, 2, 3, 4]
    assert m["Part_B"] == [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    assert matrix_to_values(m) == vals


def test_blank_stays_none_and_no_totals():
    m = build_matrix({})
    assert m["Part_A"] == [None] * 5 and m["Part_B"] == [[None, None]] * 6
    assert set(m) == {"student_id", "Part_A", "Part_B"}
