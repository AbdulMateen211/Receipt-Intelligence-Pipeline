"""Thin wrapper around pytesseract with sane defaults for receipt-style text."""
from __future__ import annotations

from dataclasses import dataclass

import pytesseract
from PIL import Image

# PSM 6: "Assume a single uniform block of text" -- a good fit for a receipt,
# which is one column of left-aligned lines rather than a multi-column page.
TESSERACT_CONFIG = "--psm 6"


@dataclass
class OcrResult:
    text: str
    mean_confidence: float


def run_ocr(img: Image.Image) -> OcrResult:
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

    data = pytesseract.image_to_data(img, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
    confidences = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OcrResult(text=text, mean_confidence=mean_conf)
