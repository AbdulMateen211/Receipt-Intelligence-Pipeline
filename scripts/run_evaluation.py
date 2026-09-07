"""
Runs the full receipt-digitization pipeline over the synthetic evaluation
set under THREE conditions:

  1. raw        -- OCR straight on the noisy phone-photo, no preprocessing
  2. grayscale  -- a naive "just convert to grayscale" baseline
  3. pipeline   -- our full deskew + denoise + CLAHE + adaptive-threshold pipeline

...and scores each on:
  - CER (character error rate) of the raw OCR text against ground truth
  - field-level accuracy (merchant / date / subtotal / tax / total)
  - item line-item recall (how many purchased items were correctly read)

Results are written to results/metrics.json and results/*.png so the
README and notebook can report real, reproducible numbers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_pipeline, simple_grayscale
from src.ocr import run_ocr
from src.extraction import extract_fields
from src.metrics import character_error_rate, money_matches, text_matches

sns.set_theme(style="whitegrid", palette="deep")

CONDITIONS = ["raw", "grayscale", "pipeline"]


def score_one(photo_path: Path, gt: dict) -> dict:
    img = Image.open(photo_path)
    row = {"receipt_id": gt["receipt_id"], "difficulty": gt["difficulty"], "category": gt["category"]}

    for condition in CONDITIONS:
        t0 = time.time()
        if condition == "raw":
            proc_img = img
        elif condition == "grayscale":
            proc_img = simple_grayscale(img)
        else:
            proc_img = preprocess_pipeline(img).final
        elapsed = time.time() - t0

        ocr_result = run_ocr(proc_img)
        fields = extract_fields(ocr_result.text)

        cer = character_error_rate(gt["reference_text"], ocr_result.text)
        merchant_ok = text_matches(fields.merchant, gt["store"])
        date_ok = fields.date == gt["date"]
        subtotal_ok = money_matches(fields.subtotal, gt["subtotal"])
        tax_ok = money_matches(fields.tax, gt["tax"])
        total_ok = money_matches(fields.total, gt["total"])

        gt_prices = sorted(round(it["line_total"], 2) for it in gt["items"])
        pred_prices = sorted(round(it.price, 2) for it in fields.items)
        matched = 0
        remaining = pred_prices.copy()
        for p in gt_prices:
            for r in remaining:
                if abs(p - r) <= 0.01:
                    remaining.remove(r)
                    matched += 1
                    break
        item_recall = matched / len(gt_prices) if gt_prices else 1.0

        prefix = condition
        row[f"{prefix}_cer"] = cer
        row[f"{prefix}_conf"] = ocr_result.mean_confidence
        row[f"{prefix}_merchant_ok"] = merchant_ok
        row[f"{prefix}_date_ok"] = date_ok
        row[f"{prefix}_subtotal_ok"] = subtotal_ok
        row[f"{prefix}_tax_ok"] = tax_ok
        row[f"{prefix}_total_ok"] = total_ok
        row[f"{prefix}_item_recall"] = item_recall
        row[f"{prefix}_time_s"] = elapsed

    return row


def main(n_limit: int | None = None, start: int = 0, end: int | None = None, chunk_tag: str | None = None):
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    if n_limit:
        manifest = manifest[:n_limit]
    manifest = manifest[start:end]

    rows = []
    for i, entry in enumerate(manifest):
        gt = json.loads((ROOT / entry["gt_path"]).read_text())
        photo_path = ROOT / entry["photo_path"]
        row = score_one(photo_path, gt)
        rows.append(row)
        if (i + 1) % 20 == 0 or (i + 1) == len(manifest):
            print(f"  scored {i + 1}/{len(manifest)}", flush=True)

    df = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if chunk_tag:
        df.to_csv(results_dir / f"chunk_{chunk_tag}.csv", index=False)
        print(f"Wrote chunk to results/chunk_{chunk_tag}.csv")
        return df, None

    df.to_csv(results_dir / "per_receipt_results.csv", index=False)
    summary = summarize_and_chart(df, results_dir)
    return df, summary


def summarize_and_chart(df: pd.DataFrame, results_dir: Path):
    field_cols = ["merchant_ok", "date_ok", "subtotal_ok", "tax_ok", "total_ok"]
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {
            "mean_cer": float(df[f"{condition}_cer"].mean()),
            "mean_confidence": float(df[f"{condition}_conf"].mean()),
            "mean_item_recall": float(df[f"{condition}_item_recall"].mean()),
            "mean_time_s": float(df[f"{condition}_time_s"].mean()),
            **{
                f"{field}_accuracy": float(df[f"{condition}_{field}"].mean())
                for field in field_cols
            },
            "overall_field_accuracy": float(
                df[[f"{condition}_{f}" for f in field_cols]].mean(axis=1).mean()
            ),
        }

    by_difficulty = {}
    for condition in CONDITIONS:
        by_difficulty[condition] = (
            df.groupby("difficulty")[f"{condition}_cer"].mean().to_dict()
        )

    (results_dir / "metrics.json").write_text(json.dumps(
        {"n_receipts": len(df), "summary": summary, "cer_by_difficulty": by_difficulty},
        indent=2,
    ))

    print("\n=== SUMMARY (mean over {} receipts) ===".format(len(df)))
    print(f"{'condition':<12} {'CER':>8} {'conf':>8} {'field_acc':>10} {'item_recall':>12}")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"{condition:<12} {s['mean_cer']:>8.3f} {s['mean_confidence']:>8.1f} "
              f"{s['overall_field_accuracy']:>10.1%} {s['mean_item_recall']:>12.1%}")

    make_charts(df, summary, results_dir)
    return summary


def make_charts(df: pd.DataFrame, summary: dict, results_dir: Path):
    labels = {"raw": "Raw photo", "grayscale": "Grayscale only", "pipeline": "Full pipeline"}
    colors = {"raw": "#d9534f", "grayscale": "#f0ad4e", "pipeline": "#2e7d32"}

    # 1. CER comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    conds = CONDITIONS
    cers = [summary[c]["mean_cer"] * 100 for c in conds]
    bars = ax.bar([labels[c] for c in conds], cers, color=[colors[c] for c in conds])
    ax.set_ylabel("Character Error Rate (%)")
    ax.set_title("OCR error rate: raw photo vs. our preprocessing pipeline")
    for b, v in zip(bars, cers):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "cer_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Field-level accuracy grouped bar chart
    field_cols = ["merchant_ok", "date_ok", "subtotal_ok", "tax_ok", "total_ok"]
    field_labels = ["Merchant", "Date", "Subtotal", "Tax", "Total"]
    x = np.arange(len(field_cols))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, cond in enumerate(conds):
        vals = [summary[cond][f"{f}_accuracy"] * 100 for f in field_cols]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[cond], color=colors[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(field_labels)
    ax.set_ylabel("Field accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Structured-field extraction accuracy by field")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "field_accuracy.png", dpi=150)
    plt.close(fig)

    # 3. CER by difficulty level, line chart
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    diffs = ["easy", "medium", "hard"]
    for cond in conds:
        vals = df.groupby("difficulty")[f"{cond}_cer"].mean().reindex(diffs) * 100
        ax.plot(diffs, vals, marker="o", label=labels[cond], color=colors[cond], linewidth=2)
    ax.set_ylabel("Character Error Rate (%)")
    ax.set_xlabel("Photo difficulty (rotation / blur / noise level)")
    ax.set_title("Pipeline robustness across difficulty levels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "cer_by_difficulty.png", dpi=150)
    plt.close(fig)

    # 4. OCR confidence distribution
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cond in conds:
        sns.kdeplot(df[f"{cond}_conf"], label=labels[cond], color=colors[cond], fill=True, alpha=0.15, ax=ax)
    ax.set_xlabel("Tesseract mean confidence")
    ax.set_title("OCR confidence distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "confidence_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="limit number of receipts (debug)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--chunk-tag", type=str, default=None, help="write a partial chunk CSV instead of final summary")
    parser.add_argument("--combine", action="store_true", help="combine results/chunk_*.csv into the final summary")
    args = parser.parse_args()

    if args.combine:
        chunk_files = sorted((ROOT / "results").glob("chunk_*.csv"))
        df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
        df.to_csv(ROOT / "results" / "per_receipt_results.csv", index=False)
        summarize_and_chart(df, ROOT / "results")
    else:
        main(n_limit=args.n, start=args.start, end=args.end, chunk_tag=args.chunk_tag)
