"""
Rule-based information extraction from raw OCR text.

Receipts are semi-structured: free text, but with strong conventions (a
store name up top, a right-aligned amount at the end of "total"-style
lines, one line per purchased item). We lean on those conventions with
regex + heuristics rather than a trained model -- transparent, fast, and
does not need labelled training data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

MONEY_RE = re.compile(r"\d{1,4}[.,]\d{2}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

SKIP_KEYWORDS = (
    "TOTAL", "TAX", "SUBTOTAL", "VISA", "MASTERCARD", "CASH", "DEBIT", "AMEX",
    "RECEIPT", "THANK", "FOLLOW", "RETURNS", "PLEASE", "HAVE A", "ONLINE",
)


@dataclass
class ExtractedItem:
    name: str
    price: float


@dataclass
class ExtractedFields:
    merchant: Optional[str] = None
    date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    items: List[ExtractedItem] = field(default_factory=list)


def _last_money_on_line(line: str) -> Optional[float]:
    """Receipts right-align the amount that actually matters at the end of
    the line (a tax line like 'TAX (8.00%)   3.44' has TWO number-like
    tokens -- the rate and the amount -- so we deliberately take the LAST
    match, not the first)."""
    matches = MONEY_RE.findall(line)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "."))
    except ValueError:
        return None


def _find_labelled_amount(lines: List[str], label_pattern: str) -> Optional[float]:
    regex = re.compile(label_pattern, re.IGNORECASE)
    for line in lines:
        if regex.search(line):
            amount = _last_money_on_line(line)
            if amount is not None:
                return amount
    return None


def _guess_merchant(lines: List[str]) -> Optional[str]:
    for line in lines[:4]:
        letters = sum(c.isalpha() for c in line)
        if letters < 4:
            continue
        if letters / max(len(line), 1) < 0.55:
            continue
        if any(k in line.upper() for k in ("RECEIPT", "THANK")):
            continue
        return line.strip()
    return None


def _guess_items(lines: List[str]) -> List[ExtractedItem]:
    items = []
    for line in lines:
        upper = line.upper()
        if any(k in upper for k in SKIP_KEYWORDS):
            continue
        matches = list(MONEY_RE.finditer(line))
        if not matches:
            continue
        last = matches[-1]
        try:
            price = float(last.group(0).replace(",", "."))
        except ValueError:
            continue
        name_part = line[: last.start()]
        name_part = name_part.strip(" .xX*0123456789")
        if len(name_part) < 2:
            continue
        items.append(ExtractedItem(name=name_part, price=price))
    return items


def extract_fields(raw_text: str) -> ExtractedFields:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    date_match = DATE_RE.search(raw_text)
    date = date_match.group(0) if date_match else None

    return ExtractedFields(
        merchant=_guess_merchant(lines),
        date=date,
        subtotal=_find_labelled_amount(lines, r"\bSUBTOTAL\b"),
        tax=_find_labelled_amount(lines, r"\bTAX\b"),
        total=_find_labelled_amount(lines, r"\bTOTAL\b"),
        items=_guess_items(lines),
    )
