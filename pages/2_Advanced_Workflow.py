"""Marksheet AI - local Streamlit app.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from config import settings
from src.box_detection import ALL_IDS, PART_A_IDS, PART_B_IDS
from src.crop_extraction import extract_all, save_crops
from src.exporter import crops_zip, matrix_csv, matrix_excel, matrix_json, metadata_records
from src.handwriting_ocr import available_engines
from src.image_processing import ImageError, bytes_to_image, enlarge, png_bytes, save_png
from src.matrix_generator import build_matrix
from src.ink import extract_ink
from src.processor import SheetResult, recognise_all
from src.pipeline_layout import align_and_detect
from src.table_detection import LayoutNotFound
from src.validation import validate_boxes, validate_mark, validate_upload

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
settings.ensure_dirs()
st.set_page_config(page_title="Advanced workflow", page_icon="ðŸ“", layout="wide")
from src.access import require_access_code  # noqa: E402

require_access_code()

STEPS = ["1 Home", "2 Upload", "3 Alignment", "4 Boxes", "5 Crops", "6 Recognition", "7 Verify", "8 Matrix", "9 Export"]
BADGE = {"auto": "ðŸŸ¢ confident", "review": "ðŸŸ  please check", "blank": "âšª blank", "manual": "âœï¸ edited"}


@st.cache_resource(show_spinner="Loading recognition models (first time only)...")
def get_engines(use_trocr: bool, use_cnn: bool):
    return available_engines(use_cnn=use_cnn, use_trocr=use_trocr)


def reset() -> None:
    n = st.session_state.get("sheet_no", 0) + 1
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["sheet_no"] = n


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sheet() -> SheetResult | None:
    return st.session_state.get("sheet")


def need_sheet() -> SheetResult | None:
    s = sheet()
    if s is None:
        st.info("Upload a marksheet photo first (step 2).")
    return s


def run_pipeline(force_k: int | None = None) -> None:
    engines, _ = get_engines(st.session_state.get("use_trocr", True), st.session_state.get("use_cnn", True))
    tpl = settings.load_template()
    bar = st.progress(0.0, text="Aligning photo and finding the 17 boxes...")
    original = st.session_state["working"]
    aligned = align_and_detect(original, tpl, force_k)
    crops = extract_all(aligned.image, aligned.boxes)
    results = {}
    for i, (bid, crop) in enumerate(crops.items()):
        from src.recognition_engine import recognise_box
        results[bid] = recognise_box(bid, crop, engines)
        bar.progress((i + 1) / len(crops), text=f"Reading handwriting: box {bid}")
    bar.empty()
    st.session_state["sheet"] = SheetResult(original, aligned, crops, results, validate_boxes(aligned.boxes, aligned.image.shape))
    st.session_state["final"] = {}      # manual overrides: box id -> int | None


def final_values(s: SheetResult) -> dict[str, int | None]:
    fin = st.session_state.get("final", {})
    return {b: (fin[b] if b in fin else s.results[b].value) for b in ALL_IDS}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("ðŸ“ Marksheet AI")
    step = st.radio("Workflow", STEPS, key="step")
    st.divider()
    st.checkbox("Use TrOCR (if installed)", value=True, key="use_trocr")
    st.checkbox("Use CNN (if installed)", value=True, key="use_cnn")
    engines, skipped = get_engines(st.session_state.get("use_trocr", True), st.session_state.get("use_cnn", True))
    st.caption("Active models: " + ", ".join(e.name for e in engines))
    if skipped:
        st.caption("Not available: " + ", ".join(skipped) + " - see README to enable.")
    if st.button("ðŸ”„ New marksheet", use_container_width=True):
        reset()
        st.rerun()

# ------------------------------------------------------------------ 1 Home
if step == STEPS[0]:
    st.header("Marksheet AI")
    st.write("Turn a photo of a handwritten marksheet cover into a clean matrix of the 17 sub-question marks "
             "(Part A: 1a-1e, Part B: 2A-7B). **It never adds up totals** - that stays with you.")
    st.markdown("""
**Photo tips:** whole marks tables in the frame Â· good light, no shadow across the tables Â· camera roughly above the
page Â· keep the phone steady Â· any rotation is fine (the app turns the page the right way up).

**Formats:** JPG, JPEG, PNG. Everything stays on this computer.
""")
    st.success("Go to **2 Upload** in the sidebar to start.")

# ------------------------------------------------------------------ 2 Upload
elif step == STEPS[1]:
    st.header("Upload marksheet")
    up = st.file_uploader("Marksheet photo", type=["jpg", "jpeg", "png"], key=f"up{st.session_state.get('sheet_no', 0)}")
    if up is not None:
        try:
            validate_uploaded = validate_upload(up.name, up.size)
            data = up.getvalue()
            img = bytes_to_image(data, up.name)
        except ImageError as exc:
            st.error(str(exc))
        else:
            if st.session_state.get("upload_name") != up.name:
                st.session_state.update(upload_name=up.name, original=img, working=img.copy(), sheet=None)
                (settings.UPLOAD_DIR / Path(up.name).name).write_bytes(data)   # untouched copy of the original
            work = st.session_state["working"]
            st.image(rgb(work), caption=f"{up.name} - {work.shape[1]}x{work.shape[0]} px", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            if c1.button("â†º Rotate left"):
                st.session_state["working"] = np.ascontiguousarray(np.rot90(work, 1)); st.session_state["sheet"] = None; st.rerun()
            if c2.button("â†» Rotate right"):
                st.session_state["working"] = np.ascontiguousarray(np.rot90(work, 3)); st.session_state["sheet"] = None; st.rerun()
            if c3.button("â–¶ Process this marksheet", type="primary"):
                try:
                    run_pipeline()
                    st.success("Done. Check the next steps, then verify marks in step 7.")
                except LayoutNotFound as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Processing failed: {exc}")
            st.caption("Rotation is optional - the app tries all four orientations by itself.")

# ------------------------------------------------------------------ 3 Alignment
elif step == STEPS[2]:
    st.header("Image alignment")
    s = need_sheet()
    if s:
        a = s.aligned
        c1, c2 = st.columns(2)
        c1.image(rgb(s.original), caption="Original (never modified)", use_container_width=True)
        vis = a.image.copy()
        xs = [b.x for b in a.boxes.values()] ; ys = [b.y for b in a.boxes.values()]
        cv2.rectangle(vis, (min(xs) - 6, min(ys) - 6), (max(b.x + b.w for b in a.boxes.values()) + 6, max(b.y + b.h for b in a.boxes.values()) + 6), (0, 160, 0), 4)
        c2.image(rgb(vis), caption=f"Aligned (skew {a.skew_deg:+.1f}Â°, turned {a.rotation_k * 90}Â°) - green = marks region", use_container_width=True)
        st.write("If the sheet is upside-down or the boxes look wrong, force an orientation:")
        k = st.selectbox("Rotation (counter-clockwise)", [None, 0, 1, 2, 3], format_func=lambda v: "automatic" if v is None else f"{v * 90}Â°")
        if st.button("Re-run alignment"):
            try:
                run_pipeline(k); st.rerun()
            except LayoutNotFound as exc:
                st.error(str(exc))

# ------------------------------------------------------------------ 4 Boxes
elif step == STEPS[3]:
    st.header("Box detection")
    s = need_sheet()
    if s:
        a = s.aligned
        probs = validate_boxes(a.boxes, a.image.shape)
        (st.success("All 17 boxes found and in the expected order.") if not probs else st.error("; ".join(probs)))
        vis = a.image.copy()
        for b in a.boxes.values():
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 160, 0), 3)
            cv2.putText(vis, b.id, (b.x + 5, b.y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        st.image(rgb(vis), use_container_width=True)
        with st.expander("Adjust a box (pixels)"):
            bid = st.selectbox("Box", ALL_IDS)
            b = a.boxes[bid]
            cols = st.columns(4)
            dx = cols[0].number_input("Move x", -80, 80, 0); dy = cols[1].number_input("Move y", -80, 80, 0)
            dw = cols[2].number_input("Width +/-", -80, 80, 0); dh = cols[3].number_input("Height +/-", -80, 80, 0)
            if st.button("Apply and re-read this box"):
                a.boxes[bid] = replace(b, x=b.x + dx, y=b.y + dy, w=b.w + dw, h=b.h + dh, adjusted=True)
                from src.crop_extraction import cut
                from src.recognition_engine import recognise_box
                s.crops[bid] = cut(a.image, a.boxes[bid])
                s.results[bid] = recognise_box(bid, s.crops[bid], engines)
                st.session_state.get("final", {}).pop(bid, None)
                st.rerun()

# ------------------------------------------------------------------ 5 Crops
elif step == STEPS[4]:
    st.header("Extracted crops")
    s = need_sheet()
    if s:
        method = st.selectbox("Enlargement method", ["lanczos", "bicubic", "nearest"])
        for bid in ALL_IDS:
            crop = s.crops[bid]
            with st.expander(f"Box {bid}", expanded=bid == "1a"):
                cols = st.columns(5)
                cols[0].image(rgb(crop), caption=f"original {crop.shape[1]}x{crop.shape[0]}")
                for col, f in zip(cols[1:4], (2, 3, 4)):
                    col.image(rgb(enlarge(crop, f, method)), caption=f"{f}x")
                from src.image_processing import enhance_colour
                cols[4].image(rgb(enhance_colour(enlarge(crop, 3, method))), caption="enhanced (colour kept)")
                ink = extract_ink(crop)
                st.image(255 - ink.mask, caption="what the recogniser sees (ink only)", width=160)

# ------------------------------------------------------------------ 6 Recognition
elif step == STEPS[5]:
    st.header("Handwriting recognition")
    s = need_sheet()
    if s:
        st.write("Models: " + ", ".join(e.name for e in engines))
        rows = []
        for bid in ALL_IDS:
            r = s.results[bid]
            rows.append({"Box": bid, "Mark": r.value, "Status": BADGE[r.status],
                         "Confidence": None if r.confidence is None else round(r.confidence, 2),
                         "Alternatives": ", ".join(f"{v} ({p:.2f})" for v, p in r.alternatives),
                         "Per model": ", ".join(f"{k}={v}" for k, v in r.engines.items()),
                         "Why flagged": "; ".join(r.reasons)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Confidence is the models' own probability, softened and averaged - it is not a guarantee. "
                   "Blank boxes are decided by ink amount, not by a model.")
        if st.button("Re-run recognition"):
            s.results = recognise_all(s.crops, engines); st.session_state["final"] = {}; st.rerun()

# ------------------------------------------------------------------ 7 Verify
elif step == STEPS[6]:
    st.header("Verify marks")
    s = need_sheet()
    if s:
        n_flag = sum(1 for r in s.results.values() if r.status == "review")
        (st.warning(f"{n_flag} box(es) need a look (orange).") if n_flag else st.success("No box is flagged - still glance over the numbers."))
        fin = st.session_state.setdefault("final", {})
        sn = st.session_state.get("sheet_no", 0)
        order = [("Part A", PART_A_IDS)] + [(f"Question {int(p[0][0])}", p) for p in PART_B_IDS]
        for title, ids in order:
            st.subheader(title)
            cols = st.columns(len(ids))
            for col, bid in zip(cols, ids):
                r = s.results[bid]
                with col:
                    st.caption(f"**{bid}** Â· {BADGE[fin_status] if (fin_status := ('manual' if bid in fin else r.status)) else ''}")
                    st.image(rgb(enlarge(s.crops[bid], 3)), use_container_width=True)
                    if r.reasons and bid not in fin:
                        st.warning("; ".join(r.reasons))
                    cur = fin[bid] if bid in fin else r.value
                    is_blank = st.checkbox("blank", value=cur is None, key=f"bl{sn}{bid}")
                    val = st.number_input("mark", 0, 10, value=int(cur) if cur is not None else 0, step=1,
                                          key=f"v{sn}{bid}", disabled=is_blank, label_visibility="collapsed")
                    new = None if is_blank else int(val)
                    if new != (r.value):
                        fin[bid] = new
                    else:
                        fin.pop(bid, None)

# ------------------------------------------------------------------ 8 Matrix
elif step == STEPS[7]:
    st.header("Final matrix")
    s = need_sheet()
    if s:
        m = build_matrix(final_values(s), st.session_state.get("student_id") or None)
        st.text_input("Student ID (optional)", key="student_id")
        st.subheader("Part A")
        st.dataframe(pd.DataFrame([m["Part_A"]], columns=PART_A_IDS), hide_index=True)
        st.subheader("Part B")
        st.dataframe(pd.DataFrame(m["Part_B"], columns=["A", "B"], index=[f"Q{q}" for q in range(2, 8)]))
        st.caption("Empty = blank box (not zero). No totals are calculated.")
        st.code(matrix_json(m), language="json")
        st.download_button("Download JSON", matrix_json(m), "marks.json", "application/json")

# ------------------------------------------------------------------ 9 Export
elif step == STEPS[8]:
    st.header("Export")
    s = need_sheet()
    if s:
        m = build_matrix(final_values(s), st.session_state.get("student_id") or None)
        sid = Path(st.session_state.get("upload_name", "marksheet")).stem
        paths = save_crops(sid, s.crops)
        fin = st.session_state.get("final", {})
        meta = metadata_records(s.results, s.aligned.boxes, paths, {b: "edited by teacher" for b in fin})
        c = st.columns(3)
        c[0].download_button("JSON", matrix_json(m), f"{sid}_marks.json", "application/json")
        c[1].download_button("CSV", matrix_csv(m), f"{sid}_marks.csv", "text/csv")
        c[2].download_button("Excel", matrix_excel(m, meta), f"{sid}_marks.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        files = [Path(p) for e in paths.values() for p in e.values()]
        extra = {f"{sid}_original.png": png_bytes(s.original), f"{sid}_aligned.png": png_bytes(s.aligned.image),
                 f"{sid}_metadata.json": json.dumps(meta, indent=2).encode()}
        st.download_button("ZIP of all crops + original + aligned + metadata", crops_zip(files, extra), f"{sid}_crops.zip", "application/zip")
        st.caption("Marks and extraction metadata are separate files by design.")
        if st.checkbox("Save my corrections as training data (data/labelled/)"):
            from src.ink import glyph_image
            saved = 0
            for bid in fin:
                v = fin[bid]
                ink = extract_ink(s.crops[bid])
                if v is not None and 0 <= v <= 9 and len(ink.parts) == 1:
                    g = (glyph_image(ink.mask, ink.parts[0]) * 255).astype(np.uint8)
                    save_png(settings.LABELLED_DIR / f"{v}_{sid}_{bid}_{int(time.time())}.png", g); saved += 1
            st.success(f"Saved {saved} corrected digit(s). Retrain with: python scripts/retrain_from_labels.py")
