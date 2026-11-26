---
title: Marksheet Evaluator
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# Marksheet AI

Photo of a handwritten marksheet cover → matrix of the **17 marks** (Part A `1a–1e`, Part B `2A–7B`).
It never computes totals, percentages or grades. Everything runs locally; no API key; no uploads.

```json
{ "student_id": null,
  "Part_A": [2, 2, 2, 2, 2],
  "Part_B": [[null,null],[4,null],[null,null],[4,4],[4,4],[null,null]] }
```
`null` = blank box (never 0). A mark the models are unsure about is **flagged orange for you to check**, never silently guessed.


## NEW: Evaluation + mobile app (`app.py`)

Open the site on a phone → **Scan & evaluate** → take a photo → check the (few) orange boxes → get the marks.

| | Rule |
|---|---|
| Part A | 1a+1b+1c+1d+1e → **/10** |
| Part B | internal choice **(Q2 or Q3), (Q4 or Q5), (Q6 or Q7)** – only the better question of each pair counts (3 × 10 = 30) → **halved → /15** |
| Total | Part A + Part B → **/25** (nothing rounded: 22.5 stays 22.5) |

Edit the rules in `config/settings.py` (`PART_A_MAX`, `PART_B_PAIRS`, `PART_B_SCALE`). Calculation lives in `src/evaluation.py` (unit-tested).
Screens: marks table for Part A, marks table for Part B (counted question ticked ✔), then **Part A / Part B / TOTAL** at the bottom. Exports: per-student Excel + JSON, and a **Class list** (Excel/CSV) for all scanned students. The old 9-step developer workflow is under *Advanced workflow* in the menu.
Safety: totals show as *provisional* and "Add to class list" stays disabled until every unclear (orange) box is ticked ✔ by the teacher.

## Deploy for free
**Hugging Face Spaces (recommended):** New Space → SDK **Docker** → push this folder (the `README.md` header and `Dockerfile` are already set for port 7860). In *Settings → Variables and secrets* add secret `APP_PASSWORD` so only your teachers can open it.
**Render:** New → Blueprint → connect the repo (`render.yaml` is included, free plan); set `APP_PASSWORD` in the dashboard.
Free-tier realities (check current limits on each site): the app sleeps when idle and takes ~30-60 s to wake; disk is temporary, so **download the class Excel before closing**; Render's free 512 MB RAM is enough for the default models but **not** for TrOCR/PyTorch – use Hugging Face if you want those. Use the phone's *Take photo* option in the upload button (full-resolution camera); photos are processed in memory and not stored.

## How it works (my architecture)

1. **Level & orient** – skew from the long printed table lines; the 4 quarter-turns are tried losslessly until the sheet's grid matches the template (Part A = 2 rows above Part B = 4 rows). The original file is never changed.
2. **Find the 17 boxes without any OCR** – printed blue grid → cells → rows. If a printed line is faint in the photo, missing columns are rebuilt from calibrated column fractions (`config/template_config.json`), snapping to real edges when they exist. Several line-filter settings and extra skew angles are tried before giving up.
3. **Crop** at original pixels (PNG, only a 3 % border trim), plus 2×/3×/4× and a colour-preserving enhanced version.
4. **Isolate the red ink** (blue grid and paper are ignored) → blank decision by ink amount → split into digits.
5. **Recognise** with an ensemble; confidence-based flags (below).
6. **Verify** in the web app, then export.

### Recognition models
| Engine | Needs | Status |
|---|---|---|
| `mnist-mlp` – scikit-learn MLP trained on MNIST with thin-stroke/rotation/shear augmentation, test-time augmentation | nothing (model is in `models/`) | **measured, see below** |
| `mnist-cnn` – small PyTorch CNN | `pip install -r requirements-optional.txt` + `python scripts/setup_models.py` | code written; **not run in my build environment (no PyTorch there)** |
| `trocr` – `microsoft/trocr-small-handwritten`, scoring only the 11 allowed answers `0…10` (cannot output anything else) | same as above (~250 MB download once) | code written; **not run in my build environment (no internet to Hugging Face)** |

When several engines are active their probabilities are averaged; they must agree and be ≥ 0.85 confident to be shown green. With one engine the bar is 0.95. Faint ink, 3+ separate strokes (cross-outs), or any disagreement → orange.
I did not pick TrOCR as the main model: it is a line-of-text model, and isolated digits in red ink are exactly where such models are weakest. The ensemble keeps it as a *second opinion*; measure it on your own sheets with `scripts/evaluate.py` before trusting it more.

## Measured results (honest)
Run `python scripts/evaluate.py`. On the 9 supplied photos (only 6 distinct sheets' worth of handwriting; two photos are the same sheet), with the bundled `mnist-mlp` engine only:

* Box detection: **9/9** photos (all 17 boxes found, correct order, any rotation; includes 90° rotated, a 1000×… low-resolution photo and strong perspective).
* Blank boxes handled correctly: **106/106**.
* Handwritten marks read correctly: **43/46 (93.5 %)**; all 3 errors were flagged orange. **0 of 12** auto-accepted (green) marks were wrong.
* Limits of this number: tiny test set; ground truth in `data/ground_truth.json` was read by eye by me (please check it); one struck-out box (`3B` on sheet 4) is excluded as ambiguous. **Do not read this as "93 % on all marksheets".** Add your own labelled sheets to `ground_truth.json` and re-run.

Not tested here: the Streamlit app in a real browser (its 9 screens were run headlessly with Streamlit's test harness with no errors), PyTorch CNN, TrOCR.

## Install (Windows)
1. Install **Python 3.10–3.12** from python.org (tick *Add python.exe to PATH*).
2. Unzip the folder, e.g. `C:\MarksheetAI`.
3. Open *Command Prompt* in that folder (type `cmd` in the folder's address bar):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The browser opens at http://localhost:8501. That's all you need.

**Optional accuracy boost (second opinion models):**
```
pip install -r requirements-optional.txt
python scripts/setup_models.py
```
Restart the app; the sidebar lists the active models.

## VS Code
Open the folder → `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv` → open a terminal (`Ctrl+``) → `streamlit run app.py`.

## Using it
Upload → *Process* → look at steps 3–4 (boxes) → **7 Verify** (orange boxes first; tick *blank* or type the mark) → **8 Matrix** → **9 Export** (JSON, CSV, Excel, crops ZIP). *New marksheet* in the sidebar resets for the next one. Without the browser: `python scripts/process_image.py photo.jpg`.

**Improve it on your sheets:** on Export tick *Save my corrections as training data*, then run `python scripts/retrain_from_labels.py` – the bundled model is fine-tuned on your students' real handwriting.

## Tests
`python -m pytest -q` (36 tests: matrix order, validation, ink/blank logic, lossless PNG, all reference photos end-to-end).

## Troubleshooting
* *"Could not find the Part A / Part B tables"* → whole marks tables must be in the frame and in focus; try *Rotate* on the Upload screen or force a rotation on the Alignment screen; adjust a single box on the Boxes screen.
* Orange on many boxes → light too dim/blurry, or only one engine active; retake, or install the optional models.
* `streamlit` not found → activate the venv (`.venv\Scripts\activate`).
* TrOCR/CNN "not available" in the sidebar → run the optional install; first TrOCR run needs internet once.
* Slow first start → models load once and are cached.

## Layout
`app.py` UI · `src/` (image_processing, image_alignment, table_detection, box_detection, pipeline_layout, crop_extraction, ink, handwriting_ocr, recognition_engine, matrix_generator, validation, exporter, processor) · `config/` · `assets/reference_images/` (byte-for-byte copies of your photos) · `models/` · `scripts/` · `tests/` · `outputs/sample/`.
