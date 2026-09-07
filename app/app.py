"""
ReceiptIQ -- interactive demo.

Run with:
    streamlit run app/app.py

Lets you upload a receipt photo (or pick one of the bundled samples),
watch the preprocessing pipeline straighten/clean it up, see the OCR text
and structured fields it extracts, and compare that against skipping
preprocessing entirely. Also keeps a running "expense tracker" for
whatever you process during the session.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_pipeline, simple_grayscale
from src.ocr import run_ocr
from src.extraction import extract_fields

st.set_page_config(page_title="ReceiptIQ", page_icon="🧾", layout="wide")

SAMPLE_RECEIPT_IDS = [
    "R0005_6T4U", "R0100_YB6L", "R0077_NGWY", "R0136_6WTL", "R0015_RPN6",
    "R0156_VIAC", "R0071_TOAM", "R0116_8SYL", "R0014_Z1GV", "R0083_2IXI",
    "R0029_5X4W", "R0113_OBT3",
]


@st.cache_data
def load_manifest():
    return {m["receipt_id"]: m for m in json.loads((ROOT / "data" / "manifest.json").read_text())}


@st.cache_data
def load_metrics():
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def process_image(img: Image.Image):
    pre = preprocess_pipeline(img)
    ocr_pipeline = run_ocr(pre.final)
    fields_pipeline = extract_fields(ocr_pipeline.text)

    ocr_raw = run_ocr(img)
    fields_raw = extract_fields(ocr_raw.text)

    return {
        "preprocessed_image": pre.final,
        "deskewed_image": pre.deskewed,
        "angle_corrected": pre.angle_corrected_deg,
        "pipeline_text": ocr_pipeline.text,
        "pipeline_conf": ocr_pipeline.mean_confidence,
        "pipeline_fields": fields_pipeline,
        "raw_text": ocr_raw.text,
        "raw_conf": ocr_raw.mean_confidence,
        "raw_fields": fields_raw,
    }


def fields_to_table(fields) -> pd.DataFrame:
    d = asdict(fields)
    items = d.pop("items")
    rows = [{"field": k, "value": v} for k, v in d.items()]
    df = pd.DataFrame(rows)
    items_df = pd.DataFrame(items) if items else pd.DataFrame(columns=["name", "price"])
    return df, items_df


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: store, total, category


def main():
    init_state()
    manifest = load_manifest()
    metrics = load_metrics()

    st.title("🧾 ReceiptIQ")
    st.caption(
        "Computer-vision preprocessing + OCR + rule-based extraction pipeline for receipts. "
        "Everything below runs the real pipeline from `src/` -- nothing is faked for the demo."
    )

    if metrics:
        s = metrics["summary"]
        cols = st.columns(4)
        cols[0].metric("Character error rate", f"{s['pipeline']['mean_cer']*100:.1f}%",
                        f"-{(s['raw']['mean_cer'] - s['pipeline']['mean_cer'])*100:.1f} pts vs. raw photo")
        cols[1].metric("Field accuracy", f"{s['pipeline']['overall_field_accuracy']*100:.1f}%",
                        f"+{(s['pipeline']['overall_field_accuracy'] - s['raw']['overall_field_accuracy'])*100:.1f} pts")
        cols[2].metric("Item recall", f"{s['pipeline']['mean_item_recall']*100:.1f}%",
                        f"+{(s['pipeline']['mean_item_recall'] - s['raw']['mean_item_recall'])*100:.1f} pts")
        cols[3].metric("OCR confidence", f"{s['pipeline']['mean_confidence']:.0f}",
                        f"+{s['pipeline']['mean_confidence'] - s['raw']['mean_confidence']:.0f}")
        st.caption(f"Measured over {metrics['n_receipts']} held-out synthetic receipts -- see `results/` and the evaluation notebook for the full breakdown.")

    st.divider()

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("1. Pick a receipt")
        source = st.radio("Source", ["Sample gallery", "Upload your own"], horizontal=True, label_visibility="collapsed")

        img = None
        gt = None
        if source == "Upload your own":
            uploaded = st.file_uploader("Upload a receipt photo", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded)
        else:
            available = [rid for rid in SAMPLE_RECEIPT_IDS if rid in manifest]
            labels = [f"{manifest[rid]['category'].title()} · {manifest[rid]['difficulty']} · {rid}" for rid in available]
            choice = st.selectbox("Sample receipts (real synthetic test-set examples)", labels)
            rid = available[labels.index(choice)]
            entry = manifest[rid]
            img = Image.open(ROOT / entry["photo_path"])
            gt_path = ROOT / entry["gt_path"]
            if gt_path.exists():
                gt = json.loads(gt_path.read_text())

        if img is not None:
            st.image(img, caption="Input photo", use_container_width=True)

    with right:
        if img is None:
            st.info("Pick a sample or upload a photo to run the pipeline.")
            return

        st.subheader("2. Pipeline output")
        with st.spinner("Running deskew -> denoise -> CLAHE -> OCR -> extraction..."):
            result = process_image(img)

        tab_clean, tab_compare, tab_raw_text = st.tabs(["Cleaned image", "Raw vs. pipeline", "Full OCR text"])

        with tab_clean:
            st.image(result["preprocessed_image"], caption=f"After preprocessing (rotation corrected {result['angle_corrected']:+.1f}°)", use_container_width=True)

        with tab_compare:
            c1, c2 = st.columns(2)
            c1.markdown(f"**No preprocessing**  \nOCR confidence: `{result['raw_conf']:.0f}`")
            c2.markdown(f"**Full pipeline**  \nOCR confidence: `{result['pipeline_conf']:.0f}`")
            raw_df, raw_items = fields_to_table(result["raw_fields"])
            pipe_df, pipe_items = fields_to_table(result["pipeline_fields"])
            c1.dataframe(raw_df, hide_index=True, use_container_width=True)
            c2.dataframe(pipe_df, hide_index=True, use_container_width=True)

        with tab_raw_text:
            c1, c2 = st.columns(2)
            c1.text_area("Raw OCR text", result["raw_text"], height=220)
            c2.text_area("Pipeline OCR text", result["pipeline_text"], height=220)

        st.subheader("3. Extracted line items")
        st.dataframe(pipe_items, hide_index=True, use_container_width=True)

        if gt is not None:
            with st.expander("Ground truth (this is a synthetic sample, so we know the right answer)"):
                st.json({k: v for k, v in gt.items() if k != "reference_text"})

        fields = result["pipeline_fields"]
        if st.button("➕ Add to expense tracker", type="primary"):
            st.session_state.history.append({
                "merchant": fields.merchant or "Unknown",
                "date": fields.date or "Unknown",
                "total": fields.total or 0.0,
            })
            st.toast("Added to expense tracker")

    st.divider()
    st.subheader("📊 Session expense tracker")
    if not st.session_state.history:
        st.caption("Process a few receipts and add them above to see running totals here.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        c1, c2 = st.columns([1, 1.5])
        c1.metric("Total spend this session", f"${hist_df['total'].sum():.2f}")
        c1.dataframe(hist_df, hide_index=True, use_container_width=True)
        by_merchant = hist_df.groupby("merchant")["total"].sum().sort_values(ascending=False)
        c2.bar_chart(by_merchant)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()


if __name__ == "__main__":
    main()
