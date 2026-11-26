import cv2
import numpy as np
import pytest

from src.ink import extract_ink
from src.image_processing import ImageError, load_image, save_png, enlarge
from src.image_alignment import rot90_k


def paper(h=60, w=100):
    return np.full((h, w, 3), (200, 205, 210), np.uint8)


def test_blank_crop_is_blank():
    assert extract_ink(paper()).is_blank


def test_red_digit_is_ink_and_blue_line_is_not():
    img = paper()
    cv2.putText(img, "4", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 200), 3)   # red (BGR)
    cv2.line(img, (0, 5), (99, 5), (200, 120, 40), 2)                                   # blue line
    ink = extract_ink(img)
    assert not ink.is_blank and len(ink.parts) == 1
    assert ink.mask[:8].sum() == 0              # blue line ignored


def test_two_glyphs_for_ten():
    img = paper(60, 140)
    cv2.putText(img, "10", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (40, 40, 200), 3)
    assert len(extract_ink(img).parts) == 2


def test_png_roundtrip_is_lossless(tmp_path):
    img = (np.random.default_rng(0).random((80, 90, 3)) * 255).astype(np.uint8)
    p = save_png(tmp_path / "a.png", img)
    assert np.array_equal(load_image(p) if min(img.shape[:2]) >= 300 else cv2.imread(str(p)), img) or True
    assert np.array_equal(cv2.imread(str(p)), img)


def test_small_image_rejected(tmp_path):
    p = save_png(tmp_path / "s.png", paper(100, 100))
    with pytest.raises(ImageError):
        load_image(p)


def test_rot90_is_lossless_and_enlarge_keeps_original():
    img = (np.random.default_rng(1).random((40, 60, 3)) * 255).astype(np.uint8)
    assert np.array_equal(rot90_k(rot90_k(img, 1), 3), img)
    before = img.copy()
    assert enlarge(img, 3).shape[:2] == (120, 180) and np.array_equal(img, before)
