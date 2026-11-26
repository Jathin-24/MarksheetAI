"""Recognition engines. Each one turns the isolated red-ink mask into a probability over marks 0..10.

Engines (all local, no API key):
  * MnistMlpEngine     - always available (pre-trained model ships in models/); scikit-learn MLP on MNIST.
  * TorchCnnEngine     - small CNN trained on MNIST (+ thin-stroke augmentation) by scripts/setup_models.py.
  * TrOCREngine        - microsoft/trocr-small-handwritten; instead of free decoding it SCORES each of the
                         11 permitted answers ("0".."10") so it can never invent an out-of-range mark.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings
from .ink import Ink, glyph_image, line_image

log = logging.getLogger(__name__)
MARKS = list(range(settings.MIN_MARK, settings.MAX_MARK + 1))


@dataclass
class EngineResult:
    engine: str
    probs: dict[int, float]          # genuine model probabilities over permitted marks (may sum to < 1)
    note: str = ""

    @property
    def top(self) -> int | None:
        return max(self.probs, key=self.probs.get) if self.probs and max(self.probs.values()) > 0 else None


def _combine_digits(per_glyph: list[np.ndarray]) -> dict[int, float]:
    """per_glyph: list of length-10 digit distributions, left->right. Returns P(mark) for mark in 0..10."""
    probs = {m: 0.0 for m in MARKS}
    if len(per_glyph) == 1:
        for d in range(10):
            probs[d] = float(per_glyph[0][d])
    elif len(per_glyph) == 2:
        probs[10] = float(per_glyph[0][1] * per_glyph[1][0])      # only "10" is a valid two-digit mark
    return probs


class MnistMlpEngine:
    """scikit-learn MLP trained on MNIST with thin-stroke/rotation/shear augmentation (ships pre-trained)."""
    name = "mnist-mlp"

    def __init__(self) -> None:
        import joblib
        from .mnist_train import MODEL_PATH, train
        self.clf = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()

    TEMPERATURE = 3.0   # MLP logits are over-confident; soften them so doubt can show up

    def _softmax_logits(self, x: np.ndarray) -> np.ndarray:
        h = x
        for w, b in zip(self.clf.coefs_[:-1], self.clf.intercepts_[:-1]):
            h = np.maximum(h @ w + b, 0)
        z = (h @ self.clf.coefs_[-1] + self.clf.intercepts_[-1]) / self.TEMPERATURE
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, ink: Ink) -> EngineResult:
        if not ink.parts:
            return EngineResult(self.name, {m: 0.0 for m in MARKS})
        # test-time augmentation: small shifts / stroke-width changes; disagreement lowers confidence
        variants = [dict(), dict(thicken=1), dict(shift=(1, 0)), dict(shift=(-1, 1)), dict(inner=18), dict(inner=22)]
        per_glyph = []
        for part in ink.parts:
            x = np.stack([glyph_image(ink.mask, part, **v).ravel() for v in variants]).astype(np.float32)
            p = self._softmax_logits(x).mean(0)
            r = np.zeros(10)
            r[self.clf.classes_.astype(int)] = p
            per_glyph.append(r)
        return EngineResult(self.name, _combine_digits(per_glyph))


class TorchCnnEngine:
    name = "mnist-cnn"
    _path = settings.MODEL_DIR / "digit_cnn.pt"

    def __init__(self) -> None:
        import torch
        if not self._path.exists():
            raise FileNotFoundError("digit_cnn.pt missing - run: python scripts/setup_models.py")
        from .cnn_model import DigitCNN
        self.torch = torch
        self.net = DigitCNN()
        self.net.load_state_dict(torch.load(self._path, map_location="cpu"))
        self.net.eval()

    def predict(self, ink: Ink) -> EngineResult:
        torch = self.torch
        if not ink.parts:
            return EngineResult(self.name, {m: 0.0 for m in MARKS})
        batch = torch.tensor(np.stack([glyph_image(ink.mask, p) for p in ink.parts]))[:, None]
        with torch.no_grad():
            p = torch.softmax(self.net(batch), dim=1).numpy()
        return EngineResult(self.name, _combine_digits(list(p)))


class TrOCREngine:
    name = "trocr"

    def __init__(self, model_name: str = settings.TROCR_MODEL) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
        tok = self.proc.tokenizer
        self.cands = {m: tok(str(m), add_special_tokens=True).input_ids for m in MARKS}
        self.start = self.model.config.decoder_start_token_id

    def predict(self, ink: Ink) -> EngineResult:
        torch = self.torch
        pix = self.proc(images=line_image(ink.mask), return_tensors="pt").pixel_values
        with torch.no_grad():
            enc = self.model.encoder(pixel_values=pix)
            scores = {}
            for m, ids in self.cands.items():
                # ids = [<s>, tokens..., </s>]; teacher-force and sum the log-probs of tokens after <s>
                dec_in = torch.tensor([[self.start] + ids[1:-1]])
                logits = self.model.decoder(input_ids=dec_in, encoder_hidden_states=enc.last_hidden_state).logits
                logp = torch.log_softmax(logits[0], -1)
                target = ids[1:]
                scores[m] = float(sum(logp[i, t] for i, t in enumerate(target)))
        arr = np.array([scores[m] for m in MARKS])
        p = np.exp(arr - arr.max())
        p /= p.sum()          # relative likelihood among the 11 permitted marks (not an absolute confidence)
        return EngineResult(self.name, {m: float(v) for m, v in zip(MARKS, p)},
                            "likelihood normalised over permitted marks 0-10")


def available_engines(use_cnn: bool = True, use_trocr: bool = True) -> tuple[list, dict[str, str]]:
    """Load whatever works on this machine; report why the others were skipped."""
    engines, skipped = [], {}
    for flag, cls in ((True, MnistMlpEngine), (use_cnn, TorchCnnEngine), (use_trocr, TrOCREngine)):
        if not flag:
            continue
        try:
            engines.append(cls())
        except Exception as exc:  # noqa: BLE001
            skipped[cls.name if hasattr(cls, "name") else cls.__name__] = f"{type(exc).__name__}: {exc}"
            log.warning("engine %s unavailable: %s", cls.__name__, exc)
    return engines, skipped
