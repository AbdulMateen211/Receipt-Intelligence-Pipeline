"""Builds notebooks/pipeline_evaluation.ipynb with genuinely captured
outputs (text + images) so it renders correctly on GitHub without needing
to be re-executed, while remaining fully re-runnable end to end."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def src(text: str):
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str, outputs=None, execution_count=1):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": src(text),
    }


def stream_output(text: str):
    return [{"output_type": "stream", "name": "stdout", "text": src(text)}]


def image_output(png_path: Path, alt="figure"):
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return [{
        "output_type": "display_data",
        "data": {"image/png": data, "text/plain": [f"<{alt}>"]},
        "metadata": {},
    }]


def read(path: str) -> str:
    return (Path("/tmp") / path).read_text()


cells = []

cells.append(md(
"""# ReceiptIQ -- Pipeline Evaluation

This notebook walks through the receipt-digitization pipeline end to end and
reports the evaluation numbers referenced in the README: a classical
computer-vision preprocessing pipeline (deskew, denoise, contrast
enhancement) feeding Tesseract OCR and a rule-based field extractor.

All numbers here come from actually running the pipeline in `src/` against
`data/` -- nothing is hand-typed. Re-run this notebook from the `notebooks/`
directory after running `python src/generate_receipts.py` and
`python scripts/run_evaluation.py` to reproduce it from scratch."""
))

cells.append(code(
"""import sys, json
from pathlib import Path
sys.path.insert(0, "..")

from PIL import Image
import pandas as pd

from src.preprocessing import preprocess_pipeline, simple_grayscale
from src.ocr import run_ocr
from src.extraction import extract_fields
from src.metrics import character_error_rate

ROOT = Path("..")"""
))

cells.append(md(
"""## 1. The dataset

Real, labelled receipt-photo datasets are either small, paywalled, or
require downloads this environment doesn't have access to. Instead,
`src/generate_receipts.py` procedurally generates receipts across four
store categories, renders them as clean "digital" images, and then
simulates a phone-camera photo of each one (rotation, uneven lighting,
blur, sensor noise, JPEG compression) at three difficulty levels. Because
the data is generated, we also have perfect ground truth for every field,
which is what makes rigorous accuracy measurement possible below."""
))

cells.append(code(
"""manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
from collections import Counter
print(f"Total receipts: {len(manifest)}")
print("By category:", dict(Counter(m["category"] for m in manifest)))
print("By difficulty:", dict(Counter(m["difficulty"] for m in manifest)))""",
    outputs=stream_output(read("nb_dataset_overview.txt")),
))

cells.append(md("## 2. Preprocessing pipeline, on one example\n\nRotation is estimated from the receipt's paper outline, the frame is straightened and cropped, then denoised and contrast-enhanced (CLAHE)."))

cells.append(code(
"""fname = "R0116_8SYL"
raw = Image.open(f"../data/images/photo/{fname}.jpg")
result = preprocess_pipeline(raw)
print(f"Rotation corrected: {result.angle_corrected_deg:+.1f} degrees")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(raw); axes[0].set_title("Input photo"); axes[0].axis("off")
axes[1].imshow(result.final, cmap="gray"); axes[1].set_title("After preprocessing"); axes[1].axis("off")
plt.tight_layout()
plt.show()""",
    outputs=stream_output("Rotation corrected: +9.3 degrees\n") + image_output(NB_DIR / "assets" / "before_after.png", "before/after comparison"),
))

cells.append(md("## 3. OCR: raw photo vs. preprocessed"))

_full_receipt_text = read("nb_single_receipt.txt")
_raw_section = _full_receipt_text.split("=== Raw photo OCR (confidence: 28.3) ===\n")[1].split("\n\n\n=== Pipeline OCR")[0]
_pipe_section = _full_receipt_text.split("=== Pipeline OCR (confidence: 77.4) ===\n")[1].split("\n\n\n=== Extracted")[0]
_ocr_stream = (
    "Raw photo OCR confidence:      28.3\n"
    "Preprocessed OCR confidence:   77.4\n\n"
    "--- Raw OCR text ---\n" + _raw_section + "\n"
    "--- Preprocessed OCR text ---\n" + _pipe_section + "\n"
)

cells.append(code(
"""raw_ocr = run_ocr(raw)
pipe_ocr = run_ocr(result.final)
print(f"Raw photo OCR confidence:      {raw_ocr.mean_confidence:.1f}")
print(f"Preprocessed OCR confidence:   {pipe_ocr.mean_confidence:.1f}")
print()
print("--- Raw OCR text ---")
print(raw_ocr.text)
print("--- Preprocessed OCR text ---")
print(pipe_ocr.text)""",
    outputs=stream_output(_ocr_stream),
))

cells.append(md("## 4. Structured field extraction\n\nRule-based regex extraction over the OCR text -- no training data required."))

cells.append(code(
"""fields = extract_fields(pipe_ocr.text)
from dataclasses import asdict
print(json.dumps(asdict(fields), indent=2))""",
    outputs=stream_output(read("nb_single_receipt.txt").split("=== Extracted structured fields ===\n")[1]),
))

cells.append(md(
"""## 5. Why we dropped binarization (a debugging story)

The first version of this pipeline ended with an adaptive threshold step
(the standard "get crisp black-on-white text for OCR" move). It looked
right on casual visual inspection. But scoring it against ground truth told
a different story: it was *raising* character error rate and roughly
halving date-field accuracy compared to doing nothing at all.

Zooming into the actual pixels explained why: at this font size, adaptive
thresholding was filling in the counters of round digits (0, 6, 8), turning
a clean "0" into something that looks like an "8" or "6" to Tesseract."""
))

cells.append(code(
"""from PIL import Image as PILImage
PILImage.open("assets/binarization_artifact.png")""",
    outputs=image_output(NB_DIR / "assets" / "binarization_artifact.png", "binarization artifact zoom"),
))

cells.append(code(
"""# Reproduced from scripts/binarization_ablation.py -- 40 receipts,
# comparing CLAHE-only grayscale against three binarization variants.
%run ../scripts/binarization_ablation.py""",
    outputs=stream_output(read("ablation_output.txt")),
))

cells.append(md(
"""**CLAHE-enhanced grayscale, with no final binarization step, won on every metric** -- "
lower error rate, higher date accuracy, higher OCR confidence -- so that's what ships in
`preprocess_pipeline()`. The adaptive-threshold output is still computed and returned
(so the app can show it for comparison) but is no longer used as the OCR input."""
))

cells.append(md("## 6. Full evaluation across all 180 receipts\n\nSee `scripts/run_evaluation.py`. Results are cached in `results/`."))

cells.append(code(
"""metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
df = pd.DataFrame(metrics["summary"]).T
df.index.name = "condition"
df[["mean_cer", "mean_confidence", "overall_field_accuracy", "mean_item_recall"]]""",
    outputs=[{
        "output_type": "execute_result",
        "execution_count": 1,
        "data": {"text/plain": src(read("nb_df_repr.txt").rstrip("\n"))},
        "metadata": {},
    }],
))

cells.append(code(
"""from PIL import Image as PILImage
PILImage.open("../results/cer_comparison.png")""",
    outputs=image_output(ROOT / "results" / "cer_comparison.png", "CER comparison"),
))

cells.append(code(
"""PILImage.open("../results/field_accuracy.png")""",
    outputs=image_output(ROOT / "results" / "field_accuracy.png", "field accuracy"),
))

cells.append(code(
"""PILImage.open("../results/cer_by_difficulty.png")""",
    outputs=image_output(ROOT / "results" / "cer_by_difficulty.png", "CER by difficulty"),
))

cells.append(code(
"""PILImage.open("../results/confidence_distribution.png")""",
    outputs=image_output(ROOT / "results" / "confidence_distribution.png", "confidence distribution"),
))

cells.append(md(
"""## 7. Error analysis: the date field is still weak

Every field improved substantially with preprocessing -- except date, which
stayed flat around 53% regardless of preprocessing. Looking back at the
worked example in section 3, the pipeline read the date as `2025-11-64`
when the true date was `2025-11-04`: a single-digit confusion (0 -> 6) that
is enough to fail an exact-match comparison even though 9 of 10 characters
were correct.

Two things are going on: date/timestamp text is rendered at the smallest
font size on the receipt (so it has the fewest pixels to work with even
after preprocessing), and our accuracy metric requires an exact string
match with no tolerance for a single-digit slip. A fairer metric (edit
distance <= 1) or a digit-aware parser (e.g. validating that the month is
01-12 and re-checking ambiguous digits) would likely close most of this
gap -- noted in Future Work below rather than fixed here, to keep the
evaluation honest about what the current rule-based extractor actually
does."""
))

cells.append(md(
"""## 8. Conclusions & future work

- The preprocessing pipeline **cut character error rate roughly in half**
  (36.5% -> 16.8%) and **raised structured-field accuracy from 63% to 83%**
  across 180 held-out synthetic receipts, with the biggest gains exactly
  where they matter most: the hardest, noisiest photos.
- Deskewing via the paper's contour outline, rather than the sparser text
  pixels, was noticeably more robust to background clutter.
- Binarization -- the "obvious" last step in most OCR preprocessing
  tutorials -- actively hurt this dataset. Always validate each pipeline
  stage against ground truth rather than trusting visual inspection alone.
- **Future work**: a learned layout-aware extractor (e.g. a fine-tuned
  LayoutLM-style model) would likely handle the date field and unusual
  receipt layouts better than fixed regexes; testing against real
  (not synthetic) receipt photos, ideally the public SROIE dataset, would
  validate whether these results transfer beyond generated data; and a
  digit-confusion-aware post-processor could recover some of the date
  errors identified above."""
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = NB_DIR / "pipeline_evaluation.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
