"""Marks calculation (kept separate from extraction).

Scheme
------
* Part A  = 1a+1b+1c+1d+1e                          (out of 10)
* Part B  = for each internal-choice pair (2|3), (4|5), (6|7) count ONLY ONE question - the one with
            more marks (a+b). 3 pairs x 10 = 30, then halved                (out of 15)
* Total   = Part A + Part B/2                                              (out of 25)
Blank boxes count as 0 here (and are listed in `notes`). Nothing is rounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from config import settings

PART_B_MAX = settings.PART_B_QUESTION_MAX * len(settings.PART_B_PAIRS) * settings.PART_B_SCALE
TOTAL_MAX = settings.PART_A_MAX + PART_B_MAX


@dataclass
class QuestionScore:
    q: int
    a: int | None
    b: int | None
    total: int
    attempted: bool
    counted: bool = False


@dataclass
class PairScore:
    left: QuestionScore
    right: QuestionScore
    counted_q: int | None
    marks: int


@dataclass
class Evaluation:
    part_a: list[tuple[str, int | None]]
    part_a_total: int
    pairs: list[PairScore]
    part_b_raw: int                  # out of 30
    part_b: float                    # out of 15
    total: float                     # out of 25
    notes: list[str] = field(default_factory=list)


def _q(values: Mapping[str, int | None], q: int) -> QuestionScore:
    a, b = values.get(f"{q}A"), values.get(f"{q}B")
    return QuestionScore(q, a, b, (a or 0) + (b or 0), a is not None or b is not None)


def evaluate(values: Mapping[str, int | None]) -> Evaluation:
    notes: list[str] = []
    each_max = settings.PART_A_MAX / 5
    part_a = [(f"1{c}", values.get(f"1{c}")) for c in "abcde"]
    for name, v in part_a:
        if v is None:
            notes.append(f"{name} is blank - counted as 0")
        elif v > each_max:
            notes.append(f"{name} = {v} is more than {each_max:g} (check the reading)")
    a_total = sum(v or 0 for _, v in part_a)

    pairs: list[PairScore] = []
    for ql, qr in settings.PART_B_PAIRS:
        L, R = _q(values, ql), _q(values, qr)
        for s in (L, R):
            if s.total > settings.PART_B_QUESTION_MAX:
                notes.append(f"Q{s.q} adds up to {s.total} (more than {settings.PART_B_QUESTION_MAX}) - check the reading")
        if L.attempted and R.attempted:
            notes.append(f"Both Q{ql} and Q{qr} are filled - only the higher one counts")
        if not (L.attempted or R.attempted):
            notes.append(f"Neither Q{ql} nor Q{qr} is filled - counted as 0")
            pairs.append(PairScore(L, R, None, 0))
            continue
        best = R if R.total > L.total else L
        best.counted = True
        pairs.append(PairScore(L, R, best.q, best.total))

    raw = sum(p.marks for p in pairs)
    b = raw * settings.PART_B_SCALE
    return Evaluation(part_a, a_total, pairs, raw, b, a_total + b, notes)


def fmt(x: float | int | None) -> str:
    if x is None:
        return "\u2013"
    return f"{x:g}"
