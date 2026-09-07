"""Reproduces the mini-benchmark referenced in the evaluation notebook and
in preprocessing.py's docstring: compares CLAHE-only grayscale against three
binarization variants on a 40-receipt sample, scored against ground truth."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import pytesseract
from PIL import Image

from src.preprocessing import _pil_to_cv, _estimate_rotation_correction, _rotate, _auto_crop_to_document
from src.metrics import character_error_rate


def make_variant(clahe_img, kind):
    if kind == "clahe_only":
        return Image.fromarray(clahe_img)
    if kind == "otsu_global":
        _, out = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(out)
    if kind == "adaptive_block25":
        out = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=25, C=15)
        return Image.fromarray(out)
    if kind == "adaptive_block51":
        out = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=51, C=15)
        return Image.fromarray(out)
    raise ValueError(kind)


def main(n=40, seed_offset=777):
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    sample = manifest[:n]

    kinds = ["clahe_only", "otsu_global", "adaptive_block25", "adaptive_block51"]
    results = {k: {"cer": [], "date_ok": 0, "conf": []} for k in kinds}

    for entry in sample:
        gt = json.loads((ROOT / entry["gt_path"]).read_text())
        img = Image.open(ROOT / entry["photo_path"])
        bgr = _pil_to_cv(img)
        corner_fill = tuple(int(c) for c in bgr[2, 2])
        angle = _estimate_rotation_correction(bgr)
        straightened = _rotate(bgr, angle, corner_fill)
        cropped = _auto_crop_to_document(straightened)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe_img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)

        for kind in kinds:
            pil_img = make_variant(clahe_img, kind)
            data = pytesseract.image_to_data(pil_img, config="--psm 6", output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(pil_img, config="--psm 6")
            confs = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            cer = character_error_rate(gt["reference_text"], text)
            results[kind]["cer"].append(cer)
            results[kind]["conf"].append(mean_conf)
            m = re.search(r"\d{4}-\d{2}-\d{2}", text)
            if m and m.group(0) == gt["date"]:
                results[kind]["date_ok"] += 1

    print(f"{'variant':<18} {'mean CER':>10} {'date acc':>10} {'mean conf':>10}")
    for k in kinds:
        r = results[k]
        print(f"{k:<18} {sum(r['cer'])/n:>9.1%} {r['date_ok']/n:>9.1%} {sum(r['conf'])/n:>10.1f}")


if __name__ == "__main__":
    main()
