"""Central configuration. All paths are relative to the project root (cross-platform)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ALIGNED_DIR = DATA_DIR / "aligned"
CROP_DIR = DATA_DIR / "crops"
EXTRACTED_DIR = DATA_DIR / "extracted"
LABELLED_DIR = DATA_DIR / "labelled"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
REFERENCE_DIR = ROOT / "assets" / "reference_images"
TEMPLATE_CONFIG = ROOT / "config" / "template_config.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth.json"

# Allowed marks (the sheet is out of 10 per sub-question)
MIN_MARK = 0
MAX_MARK = 10

# Recognition
TROCR_MODEL = "microsoft/trocr-small-handwritten"   # or microsoft/trocr-base-handwritten
AUTO_ACCEPT_CONFIDENCE = 0.85      # ensemble confidence needed to skip the "review" flag
BLANK_INK_RATIO = 0.012            # below this fraction of ink pixels the box is treated as blank (measured: blanks <=0.009, real marks >=0.0196)
FAINT_INK_RATIO = 0.02             # marks with less ink than this are always flagged for review
BLANK_MIN_COMPONENT_RATIO = 0.0012 # a single stroke blob must be at least this big

# Crops
CROP_INSET_RATIO = 0.03            # trim this fraction off each side to drop the printed border
ENLARGE_FACTORS = (2, 3, 4)
ENLARGE_METHOD = "lanczos"         # lanczos | bicubic | nearest


def load_template() -> dict:
    with open(TEMPLATE_CONFIG, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs() -> None:
    for p in (UPLOAD_DIR, ALIGNED_DIR, CROP_DIR / "original", CROP_DIR / "enlarged",
              CROP_DIR / "enhanced", EXTRACTED_DIR, LABELLED_DIR, OUTPUT_DIR, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)

# ---- Evaluation scheme (edit here if the college changes the rules) ----
PART_A_MAX = 10                    # 1a..1e added together
PART_B_PAIRS = ((2, 3), (4, 5), (6, 7))   # internal choice: only ONE question of each pair counts (the better one)
PART_B_QUESTION_MAX = 10           # one question = its a + b
PART_B_SCALE = 0.5                 # Part B (out of 30) is halved to 15
MAX_WORKING_SIDE = 3200            # phone photos larger than this are shrunk for processing only (original bytes untouched)
