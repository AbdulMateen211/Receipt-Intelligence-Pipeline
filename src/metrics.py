"""Evaluation metrics: OCR text quality (CER) and structured field accuracy.

Implemented from scratch (plain Python) so the project has zero extra
dependencies just to compute an edit distance.
"""
from __future__ import annotations

from typing import Optional


def levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) edit distance with a rolling 1D DP row."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance(ref, hyp) / len(ref). Lower is better. Whitespace
    is collapsed first so layout/line-break differences don't dominate the
    score -- we care about *character content*, not exact spacing."""
    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)


def money_matches(pred: Optional[float], truth: Optional[float], tol: float = 0.01) -> bool:
    if pred is None or truth is None:
        return False
    return abs(pred - truth) <= tol


def text_matches(pred: Optional[str], truth: Optional[str]) -> bool:
    if pred is None or truth is None:
        return False
    norm = lambda s: " ".join(s.upper().split())
    return norm(pred) == norm(truth)
