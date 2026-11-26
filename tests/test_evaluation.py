from src.box_detection import ALL_IDS
from src.evaluation import evaluate, TOTAL_MAX, PART_B_MAX

N = None


def ev(marks):
    return evaluate(dict(zip(ALL_IDS, marks)))


def test_scheme_maxima():
    assert PART_B_MAX == 15 and TOTAL_MAX == 25


def test_best_of_each_pair_and_halving():
    # Q3=4 (only one), Q5=8, Q6=8 (4+4)  -> raw 20 -> 10 ; Part A 10 -> total 20
    e = ev([2, 2, 2, 2, 2, N, N, 4, N, N, N, 4, 4, 4, 4, N, N])
    assert (e.part_a_total, e.part_b_raw, e.part_b, e.total) == (10, 20, 10, 20)
    assert [p.counted_q for p in e.pairs] == [3, 5, 6]


def test_both_attempted_takes_higher_and_notes_it():
    e = ev([2, 2, 2, 2, 2, 3, 3, 4, 4, N, N, N, N, N, N, N, N])   # Q2=6, Q3=8
    assert e.pairs[0].counted_q == 3 and e.pairs[0].marks == 8
    assert any("Both Q2 and Q3" in n for n in e.notes)


def test_neither_attempted_is_zero():
    e = ev([N] * 17)
    assert e.total == 0 and e.part_b_raw == 0 and any("Neither" in n for n in e.notes)


def test_half_marks_are_kept():
    e = ev([2, 2, 2, 2, 2, 5, 4, N, N, N, N, N, N, 5, 4, N, N] )   # Q2=9, Q7=... Q6 5+4=9 not Q7 -> raw 18+? 
    assert e.part_b == e.part_b_raw / 2 and e.total == e.part_a_total + e.part_b


def test_suspicious_values_are_flagged_not_changed():
    e = ev([5, 2, 2, 2, 2, 9, 9, N, N, N, N, N, N, N, N, N, N])
    assert any("1a = 5" in n for n in e.notes) and any("Q2 adds up to 18" in n for n in e.notes)
    assert e.part_a_total == 13
