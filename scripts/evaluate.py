"""Measure accuracy against data/ground_truth.json (box detection, mark accuracy, full-sheet accuracy)."""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from config import settings
from src.box_detection import ALL_IDS
from src.handwriting_ocr import available_engines
from src.image_processing import load_image
from src.processor import process_image
from src.table_detection import LayoutNotFound

gt = json.loads(settings.GROUND_TRUTH.read_text(encoding="utf-8"))
engines, skipped = available_engines()
print("engines:", [e.name for e in engines], "| skipped:", list(skipped))
det_ok = n_sheets = marks_ok = marks_tot = auto_wrong = auto_n = sheets_perfect = blank_ok = blank_tot = 0
for entry in gt["sheets"]:
    path = settings.ROOT / entry["image"]
    n_sheets += 1
    try:
        res = process_image(load_image(path), engines)
    except LayoutNotFound:
        print(f"{path.name}: BOX DETECTION FAILED"); continue
    det_ok += 1
    truth = dict(zip(ALL_IDS, entry["marks"]))
    perfect = True
    for bid in ALL_IDS:
        t = truth[bid]
        if t == "?":                      # ambiguous in the photo itself: excluded from scoring
            continue
        r = res.results[bid]
        if t is None:
            blank_tot += 1; blank_ok += (r.value is None or r.status == "review")
            perfect &= r.value is None
            continue
        marks_tot += 1; marks_ok += (r.value == t); perfect &= (r.value == t)
        if r.status == "auto":
            auto_n += 1; auto_wrong += (r.value != t)
    sheets_perfect += perfect
    print(f"{path.name}: {'all correct' if perfect else 'has errors/flags'}")
print(f"\nbox detection (17 boxes found + validated): {det_ok}/{n_sheets} sheets")
print(f"blank boxes handled correctly:               {blank_ok}/{blank_tot}")
print(f"handwritten mark accuracy (raw):             {marks_ok}/{marks_tot} = {100*marks_ok/max(marks_tot,1):.1f}%")
print(f"auto-accepted marks that were WRONG:         {auto_wrong}/{auto_n}")
print(f"sheets with every box correct (raw):         {sheets_perfect}/{n_sheets}")
print("NOTE: tiny test set (sheets listed in data/ground_truth.json). Do not extrapolate; add more labelled sheets.")
