"""
Synthetic receipt generator.

Since real, labelled receipt-photo datasets are hard to license/download in
an offline environment, this module *procedurally generates* realistic
grocery / cafe / pharmacy / electronics receipts, renders them as clean
"digital" images, then simulates a phone-camera photo of that receipt
(rotation, blur, noise, uneven lighting, JPEG artefacts).

Because we generate the data ourselves, we also know perfect ground truth
for every field -- which lets the evaluation pipeline (src/evaluate.py)
measure exactly how accurate the OCR + extraction pipeline is, and how much
the preprocessing step (src/preprocessing.py) helps.
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

CATALOG = {
    "grocery": {
        "stores": ["GreenLeaf Market", "Family Fresh Grocers", "Sunrise Superstore", "CornerMart"],
        "items": [
            ("Whole Milk 1L", 2.49), ("Brown Bread", 3.10), ("Free Range Eggs 12ct", 4.25),
            ("Bananas 1kg", 1.35), ("Cheddar Cheese 200g", 3.75), ("Basmati Rice 5kg", 8.99),
            ("Olive Oil 500ml", 6.49), ("Tomatoes 1kg", 2.10), ("Chicken Breast 1kg", 7.80),
            ("Orange Juice 1L", 2.95), ("Pasta 500g", 1.60), ("Green Tea 20bags", 3.20),
        ],
    },
    "cafe": {
        "stores": ["Bean & Brew Cafe", "The Daily Grind", "Cafe Luna", "Roast House"],
        "items": [
            ("Cappuccino", 3.50), ("Espresso", 2.50), ("Blueberry Muffin", 2.75),
            ("Avocado Toast", 6.90), ("Iced Latte", 4.10), ("Croissant", 2.60),
            ("Bagel & Cream Cheese", 3.95), ("Cold Brew", 3.80), ("Chai Latte", 3.95),
        ],
    },
    "pharmacy": {
        "stores": ["HealthFirst Pharmacy", "CarePlus Drugstore", "Wellness Pharmacy"],
        "items": [
            ("Paracetamol 500mg", 3.99), ("Vitamin C 60ct", 6.50), ("Hand Sanitizer 250ml", 2.75),
            ("Bandages Box", 2.20), ("Cough Syrup 100ml", 5.40), ("Sunscreen SPF50", 9.99),
            ("Face Masks 10pk", 4.60), ("Allergy Relief 30ct", 7.25),
        ],
    },
    "electronics": {
        "stores": ["ByteHub Electronics", "CircuitCity Express", "TechNest Store"],
        "items": [
            ("USB-C Cable 1m", 6.99), ("Wireless Mouse", 14.50), ("HDMI Cable 2m", 8.99),
            ("AA Batteries 8pk", 5.25), ("Phone Case", 11.90), ("Screen Protector", 6.40),
            ("Power Bank 10000mAh", 22.00), ("Earphones", 9.75),
        ],
    },
}

PAYMENTS = ["VISA **** 4471", "MASTERCARD **** 8823", "CASH", "DEBIT **** 1190", "AMEX **** 3305"]
FOOTERS = [
    "THANK YOU FOR SHOPPING WITH US!",
    "PLEASE COME AGAIN",
    "HAVE A GREAT DAY",
    "RETURNS ACCEPTED WITHIN 14 DAYS",
    "FOLLOW US @STORE ONLINE",
]


@dataclass
class LineItem:
    name: str
    qty: int
    unit_price: float
    line_total: float


@dataclass
class ReceiptData:
    receipt_id: str
    category: str
    store: str
    date: str
    time: str
    items: List[LineItem]
    subtotal: float
    tax_rate: float
    tax: float
    total: float
    payment: str


def _rand_id(rng: random.Random, n=8) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=n))


def generate_receipt_data(rng: random.Random, receipt_id: str) -> ReceiptData:
    category = rng.choice(list(CATALOG.keys()))
    store = rng.choice(CATALOG[category]["stores"])
    n_items = rng.randint(2, 7)
    pool = CATALOG[category]["items"]
    chosen = rng.sample(pool, k=min(n_items, len(pool)))

    items: List[LineItem] = []
    subtotal = 0.0
    for name, price in chosen:
        qty = rng.choice([1, 1, 1, 2, 3])
        unit_price = round(price * rng.uniform(0.95, 1.08), 2)
        line_total = round(unit_price * qty, 2)
        subtotal += line_total
        items.append(LineItem(name=name, qty=qty, unit_price=unit_price, line_total=line_total))

    subtotal = round(subtotal, 2)
    tax_rate = rng.choice([0.05, 0.07, 0.08, 0.0825, 0.1])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    days_back = rng.randint(0, 700)
    dt = datetime.now() - timedelta(days=days_back, hours=rng.randint(0, 23), minutes=rng.randint(0, 59))

    return ReceiptData(
        receipt_id=receipt_id,
        category=category,
        store=store,
        date=dt.strftime("%Y-%m-%d"),
        time=dt.strftime("%H:%M"),
        items=items,
        subtotal=subtotal,
        tax_rate=tax_rate,
        tax=tax,
        total=total,
        payment=rng.choice(PAYMENTS),
    )


def receipt_to_reference_text(data: ReceiptData) -> str:
    """Canonical plain-text rendering of a receipt's content, used as the
    ground-truth reference string for character-error-rate (CER) scoring.
    Whitespace/column alignment doesn't need to match the rendered image
    exactly -- CER collapses runs of whitespace before comparing."""
    lines = [data.store.upper(), f"Receipt #{data.receipt_id}", f"{data.date} {data.time}"]
    for it in data.items:
        qty_part = f"x{it.qty}" if it.qty > 1 else ""
        lines.append(f"{it.name} {qty_part} {it.line_total:.2f}")
    lines.append(f"SUBTOTAL {data.subtotal:.2f}")
    lines.append(f"TAX ({data.tax_rate*100:.2f}%) {data.tax:.2f}")
    lines.append(f"TOTAL {data.total:.2f}")
    lines.append(data.payment)
    return "\n".join(lines)


def render_receipt_image(data: ReceiptData, rng: random.Random, width: int = 400) -> Image.Image:
    """Render a clean, digital-looking receipt as a white PIL image."""
    font_small = ImageFont.truetype(FONT_PATH, 15)
    font_bold = ImageFont.truetype(FONT_BOLD_PATH, 18)
    font_tiny = ImageFont.truetype(FONT_PATH, 15)

    line_h = 22
    n_lines = 11 + len(data.items) + rng.randint(0, 2)
    height = 40 + n_lines * line_h + 60
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    y = 18
    margin = 18

    def center_text(text, font, yy, fill=0):
        w = draw.textlength(text, font=font)
        draw.text(((width - w) / 2, yy), text, font=font, fill=fill)

    center_text(data.store.upper(), font_bold, y)
    y += 26
    center_text(f"Receipt #{data.receipt_id}", font_tiny, y, fill=60)
    y += 20
    center_text(f"{data.date}  {data.time}", font_tiny, y, fill=60)
    y += 22

    draw.text((margin, y), "-" * 40, font=font_small, fill=0)
    y += line_h

    for it in data.items:
        left = f"{it.name[:22]:<22}"
        qty_part = f"x{it.qty}" if it.qty > 1 else "  "
        right = f"{qty_part} {it.line_total:6.2f}"
        draw.text((margin, y), left, font=font_small, fill=0)
        rw = draw.textlength(right, font=font_small)
        draw.text((width - margin - rw, y), right, font=font_small, fill=0)
        y += line_h

    draw.text((margin, y), "-" * 40, font=font_small, fill=0)
    y += line_h

    for label, value in [
        ("SUBTOTAL", data.subtotal),
        (f"TAX ({data.tax_rate*100:.2f}%)", data.tax),
    ]:
        rtext = f"{value:.2f}"
        draw.text((margin, y), label, font=font_small, fill=0)
        rw = draw.textlength(rtext, font=font_small)
        draw.text((width - margin - rw, y), rtext, font=font_small, fill=0)
        y += line_h

    draw.text((margin, y), "TOTAL", font=font_bold, fill=0)
    rtext = f"{data.total:.2f}"
    rw = draw.textlength(rtext, font=font_bold)
    draw.text((width - margin - rw, y), rtext, font=font_bold, fill=0)
    y += line_h + 6

    draw.text((margin, y), data.payment, font=font_tiny, fill=40)
    y += 20

    center_text(rng.choice(FOOTERS), font_tiny, y, fill=60)

    return img.convert("RGB")


def _random_noise(img: np.ndarray, rng: random.Random, sigma: float) -> np.ndarray:
    noise = rng.gauss  # not vectorized; use numpy instead
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noisy = img.astype(np.float32) + np_rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_to_photo(img: Image.Image, rng: random.Random, difficulty: str = "medium") -> Image.Image:
    """Simulate a phone-camera photo of a printed receipt lying on a table."""
    import cv2

    difficulty_params = {
        "easy": dict(rot=3, blur=0.4, noise=4, jpeg=90, shadow=0.05),
        "medium": dict(rot=7, blur=0.9, noise=9, jpeg=70, shadow=0.15),
        "hard": dict(rot=12, blur=1.6, noise=16, jpeg=45, shadow=0.28),
    }
    p = difficulty_params[difficulty]

    arr = np.array(img)

    # 1. Place on a larger neutral "table" background so rotation doesn't clip content.
    h, w = arr.shape[:2]
    pad = int(max(h, w) * 0.18)
    canvas_color = int(rng.uniform(150, 210))
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), canvas_color, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = arr

    # 2. Rotate slightly, as if the photo wasn't taken perfectly straight-on.
    angle = rng.uniform(-p["rot"], p["rot"])
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    canvas = cv2.warpAffine(
        canvas, rot_mat, (canvas.shape[1], canvas.shape[0]),
        borderValue=(canvas_color, canvas_color, canvas_color),
    )

    # 3. Uneven lighting: a soft gradient "shadow" across the page.
    yy, xx = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
    dx, dy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    grad = (xx * dx + yy * dy)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    shadow = 1.0 - p["shadow"] * grad
    canvas = np.clip(canvas.astype(np.float32) * shadow[..., None], 0, 255).astype(np.uint8)

    # 4. Blur (camera focus / motion) and sensor noise.
    k = max(1, int(round(p["blur"] * 2)) | 1)  # odd kernel size
    canvas = cv2.GaussianBlur(canvas, (k, k), sigmaX=p["blur"])
    canvas = _random_noise(canvas, rng, sigma=p["noise"])

    # 5. Random brightness/contrast jitter.
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-15, 15)      # brightness
    canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    out = Image.fromarray(canvas)

    # 6. Re-encode through JPEG to add compression artefacts.
    from io import BytesIO
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=p["jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_dataset(n: int, out_dir: Path, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    clean_dir = out_dir / "images" / "clean"
    photo_dir = out_dir / "images" / "photo"
    gt_dir = out_dir / "ground_truth"
    for d in (clean_dir, photo_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    difficulties = ["easy"] * (n // 3) + ["medium"] * (n // 3)
    difficulties += ["hard"] * (n - len(difficulties))
    rng.shuffle(difficulties)

    for i in range(n):
        rid = f"R{i:04d}_{_rand_id(rng, 4)}"
        data = generate_receipt_data(rng, rid)
        clean_img = render_receipt_image(data, rng)
        difficulty = difficulties[i]
        photo_img = degrade_to_photo(clean_img, rng, difficulty=difficulty)

        clean_path = clean_dir / f"{rid}.png"
        photo_path = photo_dir / f"{rid}.jpg"
        gt_path = gt_dir / f"{rid}.json"

        clean_img.save(clean_path)
        photo_img.save(photo_path, quality=95)
        gt_dict = asdict(data)
        gt_dict["difficulty"] = difficulty
        gt_dict["reference_text"] = receipt_to_reference_text(data)
        gt_path.write_text(json.dumps(gt_dict, indent=2))

        manifest.append({
            "receipt_id": rid,
            "category": data.category,
            "difficulty": difficulty,
            "clean_path": str(clean_path.relative_to(out_dir.parent)),
            "photo_path": str(photo_path.relative_to(out_dir.parent)),
            "gt_path": str(gt_path.relative_to(out_dir.parent)),
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic receipt dataset")
    parser.add_argument("--n", type=int, default=180, help="number of receipts to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = parser.parse_args()

    manifest = build_dataset(args.n, Path(args.out), seed=args.seed)
    print(f"Generated {len(manifest)} synthetic receipts -> {args.out}")
