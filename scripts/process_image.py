"""Command-line use:  python scripts/process_image.py photo.jpg [--out outputs/]"""
import argparse, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from config import settings
from src.crop_extraction import save_crops
from src.exporter import matrix_csv, matrix_excel, matrix_json, metadata_records
from src.handwriting_ocr import available_engines
from src.image_processing import load_image, save_png
from src.processor import process_image

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--out", default=str(settings.OUTPUT_DIR))
ap.add_argument("--no-trocr", action="store_true")
ap.add_argument("--no-cnn", action="store_true")
args = ap.parse_args()

settings.ensure_dirs()
engines, skipped = available_engines(use_cnn=not args.no_cnn, use_trocr=not args.no_trocr)
print("engines:", [e.name for e in engines], "| skipped:", skipped or "none")
res = process_image(load_image(args.image), engines)
sid = Path(args.image).stem
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
paths = save_crops(sid, res.crops)
save_png(out / f"{sid}_aligned.png", res.aligned.image)
m = res.matrix()
(out / f"{sid}_marks.json").write_text(matrix_json(m), encoding="utf-8")
(out / f"{sid}_marks.csv").write_text(matrix_csv(m), encoding="utf-8")
meta = metadata_records(res.results, res.aligned.boxes, paths, {})
(out / f"{sid}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(out / f"{sid}_marks.xlsx").write_bytes(matrix_excel(m, meta))
print(matrix_json(m))
review = [i for i, r in res.results.items() if r.status == "review"]
print("NEEDS REVIEW:", review or "none")
