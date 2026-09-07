"""
Classical computer-vision preprocessing pipeline.

Goal: take a noisy, rotated, unevenly-lit "phone photo" of a receipt and
recover something close to the clean, high-contrast image that OCR engines
(Tesseract) work best on.

Every step uses standard OpenCV primitives -- no deep learning, no external
downloads -- so it runs anywhere.

Pipeline order matters: we estimate rotation from the receipt's *paper
outline* (one big, clean contour) while the full noisy background is still
present, rotate the whole frame to straighten it, and only then crop tightly
-- straightening first means the crop afterwards has almost no leftover
background left in the corners.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


@dataclass
class PreprocessResult:
    final: Image.Image
    grayscale: Image.Image
    deskewed: Image.Image
    binarized: Image.Image
    angle_corrected_deg: float


def _largest_paper_contour(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(biggest) / (bgr.shape[0] * bgr.shape[1])
    if area_ratio < 0.1:
        return None
    return biggest


def _estimate_rotation_correction(bgr: np.ndarray) -> float:
    """Fit a rotated rectangle to the receipt's paper outline and return the
    angle (degrees) that straightens it back to axis-aligned, normalised to
    the range (-45, 45]."""
    contour = _largest_paper_contour(bgr)
    if contour is None:
        return 0.0
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    angle = rect[-1]

    correction = angle if rw < rh else angle + 90
    if correction > 45:
        correction -= 90
    elif correction <= -45:
        correction += 90
    return float(correction)


def _rotate(bgr: np.ndarray, angle: float, fill) -> np.ndarray:
    if abs(angle) < 0.1:
        return bgr
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(bgr, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=fill)


def _auto_crop_to_document(bgr: np.ndarray) -> np.ndarray:
    """Crop tightly to the receipt's bounding box (call *after* straightening)."""
    contour = _largest_paper_contour(bgr)
    if contour is None:
        return bgr
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    return bgr[y0:y1, x0:x1]


def preprocess_pipeline(img: Image.Image) -> PreprocessResult:
    """Full pipeline: deskew -> crop -> denoise -> contrast-enhance.

    Note on a design decision that came out of the evaluation notebook: an
    earlier version of this pipeline finished with an adaptive threshold
    (binarization) step, on the common assumption that crisp black-on-white
    text helps OCR. A controlled comparison against ground truth (see
    notebooks/pipeline_evaluation.ipynb, "Why we dropped binarization")
    showed the opposite for this data: adaptive thresholding filled in the
    counters of small round digits (0/6/8) and *raised* character error rate
    from ~17% to ~22-24% and roughly halved date-field accuracy. CLAHE-
    enhanced grayscale, with no final binarization, was the best-performing
    variant we tested, so that is what ships here.
    """
    bgr = _pil_to_cv(img)
    corner_fill = tuple(int(c) for c in bgr[2, 2])  # sample background colour

    angle = _estimate_rotation_correction(bgr)
    straightened = _rotate(bgr, angle, fill=corner_fill)

    cropped = _auto_crop_to_document(straightened)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Denoise while preserving text edges.
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE (adaptive histogram equalization) to fix uneven lighting/shadow.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Adaptive threshold is computed only so the app/notebook can *show* the
    # rejected alternative for comparison -- it is NOT used as the final OCR
    # input (see docstring above).
    binarized = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=15,
    )

    final = enhanced

    return PreprocessResult(
        final=_cv_to_pil(final),
        grayscale=_cv_to_pil(gray),
        deskewed=_cv_to_pil(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)),
        binarized=_cv_to_pil(binarized),
        angle_corrected_deg=angle,
    )


def simple_grayscale(img: Image.Image) -> Image.Image:
    """The 'do nothing fancy' baseline used for the ablation study."""
    return img.convert("L")
