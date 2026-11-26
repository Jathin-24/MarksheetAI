"""Runs the real pipeline on the reference photos (needs the bundled MNIST-MLP model only)."""
import json
import pytest

from config import settings
from src.box_detection import ALL_IDS
from src.handwriting_ocr import MnistMlpEngine
from src.image_processing import load_image
from src.processor import process_image

GT = json.loads(settings.GROUND_TRUTH.read_text(encoding="utf-8"))["sheets"]


@pytest.fixture(scope="module")
def engines():
    return [MnistMlpEngine()]


@pytest.mark.parametrize("entry", GT, ids=[e["image"].split("/")[-1] for e in GT])
def test_sheet(entry, engines):
    res = process_image(load_image(settings.ROOT / entry["image"]), engines)
    assert not res.problems and len(res.crops) == 17
    for bid, truth in zip(ALL_IDS, entry["marks"]):
        r = res.results[bid]
        if truth is None:                         # blank must never become a number
            assert r.value is None, bid
        elif truth != "?":
            # either right, or honestly flagged - a wrong mark must never be auto-accepted
            assert r.value == truth or r.status == "review", (bid, r.value, truth)
