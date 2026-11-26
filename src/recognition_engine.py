"""Ensemble of engines + honest status flags. Never guesses: blank stays None, doubt is flagged."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import settings
from .handwriting_ocr import MARKS, EngineResult
from .ink import extract_ink

# status values
BLANK, AUTO, REVIEW, MANUAL = "blank", "auto", "review", "manual"


@dataclass
class BoxResult:
    box_id: str
    value: int | None
    status: str
    confidence: float | None = None
    alternatives: list[tuple[int, float]] = field(default_factory=list)
    engines: dict[str, int | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    ink_ratio: float = 0.0
    preprocessing: str = "red-ink hysteresis mask"


def recognise_box(box_id: str, crop_bgr: np.ndarray, engines: list) -> BoxResult:
    ink = extract_ink(crop_bgr)
    if ink.is_blank:
        return BoxResult(box_id, None, BLANK, None, ink_ratio=ink.ratio, reasons=["no handwriting found"])
    results: list[EngineResult] = []
    for eng in engines:
        try:
            results.append(eng.predict(ink))
        except Exception as exc:  # noqa: BLE001 - one broken engine must not kill the sheet
            results.append(EngineResult(getattr(eng, "name", "engine"), {}, f"failed: {exc}"))
    usable = [r for r in results if r.probs and max(r.probs.values()) > 0]
    reasons: list[str] = []
    if not usable:
        return BoxResult(box_id, None, REVIEW, None, ink_ratio=ink.ratio,
                         reasons=["no engine could read this box"],
                         engines={r.engine: None for r in results})
    mean = np.mean([[r.probs.get(m, 0.0) for m in MARKS] for r in usable], axis=0)
    order = np.argsort(mean)[::-1]
    top, conf = int(MARKS[order[0]]), float(mean[order[0]])
    alts = [(int(MARKS[i]), float(mean[i])) for i in order[1:4] if mean[i] > 0.01]
    tops = {r.engine: r.top for r in usable}
    if len(set(tops.values())) > 1:
        reasons.append("engines disagree: " + ", ".join(f"{k}={v}" for k, v in tops.items()))
    need = settings.AUTO_ACCEPT_CONFIDENCE if len(usable) > 1 else settings.SINGLE_ENGINE_CONFIDENCE
    if conf < need:
        reasons.append(f"low confidence ({conf:.2f})")
    if ink.ratio < settings.FAINT_INK_RATIO:
        reasons.append("faint ink")
    if len(ink.parts) > 2:
        reasons.append(f"{len(ink.parts)} separate strokes found (overwriting/cross-out?)")
    status = AUTO if not reasons else REVIEW
    return BoxResult(box_id, top, status, conf, alts, tops, reasons, ink.ratio)
