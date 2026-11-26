"""Marksheet Evaluator - mobile-first. Photo in -> marks matrix -> Part A /10, Part B /15, Total /25.

Run locally:  streamlit run app.py        (the 9-step developer view is under 'Advanced workflow' in the menu)
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import streamlit as st

from config import settings
from src.box_detection import ALL_IDS, PART_A_IDS, PART_B_IDS
from src.evaluation import PART_B_MAX, TOTAL_MAX, evaluate, fmt
from src.handwriting_ocr import available_engines
from src.image_processing import ImageError, bytes_to_image, enlarge
from src.matrix_generator import build_matrix
from src.exporter import matrix_json, metadata_records
from src.processor import SheetResult, process_image
from src.report import class_excel, class_frame, class_row, student_excel, tables
from src.table_detection import LayoutNotFound
from src.validation import validate_upload

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

st.set_page_config(page_title="Marksheet Evaluator", page_icon="ðŸ“", layout="centered")
st.markdown("""<style>.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:760px}
div[data-testid="stMetricValue"]{font-size:1.6rem}</style>""", unsafe_allow_html=True)


from src.access import require_access_code  # noqa: E402

require_access_code()


@st.cache_resource(show_spinner="Loading models (first start only)...")
def engines():
    return available_engines(use_cnn=True, use_trocr=True)


def sn() -> int:
    return st.session_state.setdefault("sheet_no", 0)


def new_sheet() -> None:
    n = sn() + 1
    keep = {"class_rows": st.session_state.get("class_rows", []), "authed": st.session_state.get("authed")}
    st.session_state.clear()
    st.session_state.update(keep, sheet_no=n)


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cap_size(img: np.ndarray) -> np.ndarray:
    """Huge phone photos are shrunk for the working copy only (keeps free-tier RAM safe)."""
    m = max(img.shape[:2])
    if m <= settings.MAX_WORKING_SIDE:
        return img
    f = settings.MAX_WORKING_SIDE / m
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ header
st.title("ðŸ“ Marksheet Evaluator")
eng, skipped = engines()
tab_scan, tab_class, tab_help = st.tabs(["ðŸ“· Scan & evaluate", "ðŸ‘¥ Class list", "â„¹ï¸ Help"])

# ================================================================== SCAN
with tab_scan:
    up = st.file_uploader("Take a photo of the marks page (or pick one from the gallery)",
                          type=["jpg", "jpeg", "png"], key=f"up{sn()}")
    if up is None:
        st.info("Tip: whole Part A and Part B tables in the frame, good light, no shadow. Any rotation is fine.")
    else:
        sig = (up.name, up.size)
        if st.session_state.get("sig") != sig:
            st.session_state.pop("sheet", None)
            try:
                validate_upload(up.name, up.size, max_mb=25)
                with st.spinner("Reading the marksheet..."):
                    img = cap_size(bytes_to_image(up.getvalue(), up.name))
                    st.session_state["sheet"] = process_image(img, eng)
                st.session_state["sig"] = sig
            except (ImageError, LayoutNotFound) as exc:
                st.session_state["sig"] = sig
                st.error(str(exc))
            except Exception:  # noqa: BLE001
                log.exception("processing failed")
                st.session_state["sig"] = sig
                st.error("Something went wrong while reading this photo. Please retake it and try again.")

    sheet: SheetResult | None = st.session_state.get("sheet")
    if sheet is not None:
        n = sn()
        flagged = [b for b in ALL_IDS if sheet.results[b].status == "review"]
        values: dict[str, int | None] = {}

        def box_editor(bid: str, need_check: bool) -> None:
            r = sheet.results[bid]
            c1, c2 = st.columns([1, 1])
            c1.image(rgb(enlarge(sheet.crops[bid], 2)), caption=f"Box {bid}")
            with c2:
                if need_check and r.reasons:
                    st.caption("âš ï¸ " + "; ".join(r.reasons))
                blank = st.checkbox("blank", value=r.value is None, key=f"bl{n}{bid}")
                val = st.number_input("mark", 0, 10, value=int(r.value) if r.value is not None else 0,
                                      step=1, key=f"v{n}{bid}", disabled=blank, label_visibility="collapsed")
                if need_check:
                    st.checkbox("âœ” I checked this", key=f"ok{n}{bid}")
            values[bid] = None if blank else int(val)

        st.subheader("1 Â· Check the marks")
        if flagged:
            unchecked = [b for b in flagged if not st.session_state.get(f"ok{n}{b}")]
            st.warning(f"{len(flagged)} box(es) are unclear. Compare with the photo, fix if needed and tick âœ”.")
            for bid in flagged:
                with st.container(border=True):
                    box_editor(bid, True)
        else:
            unchecked = []
            st.success("All boxes were read confidently. Still, glance over the list below.")
        with st.expander("All other boxes (tap to review)"):
            for bid in [b for b in ALL_IDS if b not in flagged]:
                with st.container(border=True):
                    box_editor(bid, False)
        for bid in ALL_IDS:                       # widgets inside a closed expander still report values
            values.setdefault(bid, sheet.results[bid].value)
        with st.expander("See what was detected on the page"):
            vis = sheet.aligned.image.copy()
            for b in sheet.aligned.boxes.values():
                cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 160, 0), 3)
                cv2.putText(vis, b.id, (b.x + 4, b.y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            st.image(rgb(vis))

        # ---------------------------------------------------------- result
        ev = evaluate(values)
        ta, tb, ts = tables(ev)
        st.subheader("2 Â· Result")
        if unchecked:
            st.error(f"Totals are provisional - {len(unchecked)} unclear box(es) not yet ticked âœ”: {', '.join(unchecked)}")
        st.markdown("**Part A** (marks per question)")
        st.table(ta.set_index("Question"))
        st.markdown("**Part B** (internal choice - the better question of each pair counts)")
        st.table(tb.set_index("Question"))
        for note in ev.notes:
            st.caption("â„¹ï¸ " + note)
        st.markdown("**Final marks**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Part A", f"{fmt(ev.part_a_total)} / {settings.PART_A_MAX}")
        m2.metric("Part B", f"{fmt(ev.part_b)} / {PART_B_MAX:g}", help=f"{ev.part_b_raw} out of 30, halved")
        m3.metric("TOTAL", f"{fmt(ev.total)} / {TOTAL_MAX:g}")

        # ---------------------------------------------------------- save / export
        st.subheader("3 Â· Save")
        student = st.text_input("Roll no. / name", key=f"stu{n}")
        matrix = build_matrix(values, student or None)
        meta = metadata_records(sheet.results, sheet.aligned.boxes, {}, {})
        c1, c2 = st.columns(2)
        add = c1.button("âž• Add to class list", type="primary", disabled=bool(unchecked) or not student)
        if add:
            st.session_state.setdefault("class_rows", []).append(class_row(student, values, ev))
            st.success(f"Added {student}. Scan the next sheet below.")
        c2.download_button("â¬‡ï¸ This student (Excel)", student_excel(ev, student, meta),
                           f"{student or 'student'}_evaluation.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("â¬‡ï¸ Marks matrix (JSON)", matrix_json(matrix), f"{student or 'student'}_marks.json", "application/json")
        if unchecked:
            st.caption("Tick âœ” on every unclear box to enable 'Add to class list'.")
        st.button("ðŸ“· Next marksheet", on_click=new_sheet)

# ================================================================== CLASS
with tab_class:
    rows = st.session_state.get("class_rows", [])
    if not rows:
        st.info("Evaluated students appear here. Download before you close the page - nothing is stored on the server.")
    else:
        df = class_frame(rows)
        st.dataframe(df, hide_index=True)
        st.caption(f"{len(rows)} student(s)")
        c1, c2 = st.columns(2)
        c1.download_button("â¬‡ï¸ Class (Excel)", class_excel(rows), "class_marks.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c2.download_button("â¬‡ï¸ Class (CSV)", df.to_csv(index=False), "class_marks.csv", "text/csv")
        if st.button("ðŸ—‘ï¸ Clear class list"):
            st.session_state["class_rows"] = []
            st.rerun()

# ================================================================== HELP
with tab_help:
    st.markdown(f"""
**How the marks are calculated**

* **Part A** = 1a + 1b + 1c + 1d + 1e â†’ out of **{settings.PART_A_MAX}**
* **Part B** has internal choice: **(Q2 or Q3), (Q4 or Q5), (Q6 or Q7)**. For every pair only the question with
  more marks (a + b) is counted â†’ 3 Ã— 10 = 30, then **halved** â†’ out of **{PART_B_MAX:g}**
* **Total** = Part A + Part B â†’ out of **{TOTAL_MAX:g}**. Nothing is rounded (e.g. 22.5 stays 22.5).
* A blank box counts as 0 in the sum (a note is shown). If both questions of a pair are filled, the higher one counts.

**Privacy:** photos are processed in memory and not saved. Download your class sheet before closing the page.

**Models active:** {", ".join(e.name for e in eng)}{(" Â· unavailable: " + ", ".join(skipped)) if skipped else ""}
""")
