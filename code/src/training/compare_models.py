"""Compare two trained models (baseline vs ResNet18): per-class F1 bar chart + summary.

Reads the metrics JSONs written by evaluate.py and produces a side-by-side bar chart so the
report can show the improvement at a glance.

Usage:
    python -m src.training.compare_models --split test
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def load_metrics(name: str, split: str, rep_dir):
    p = rep_dir / f"metrics_{name}_{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing: {p} -- run evaluate.py for this model/split first")
    return json.loads(p.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config()
    rep_dir = resolve_path(cfg["paths"]["reports_dir"])
    fig_dir = resolve_path(cfg["paths"]["figures_dir"])

    m_b = load_metrics("baseline", args.split, rep_dir)
    m_r = load_metrics("resnet18", args.split, rep_dir)
    classes = cfg["classes"]

    f1_b = [m_b["per_class"][c]["f1-score"] for c in classes]
    f1_r = [m_r["per_class"][c]["f1-score"] for c in classes]

    # Per-class F1 grouped bar chart with delta annotation
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    b1 = ax.bar(x - w/2, f1_b, w, label=f"baseline CNN (macro-F1 {m_b['macro_f1']:.2f})",
                color="#1F4E79", edgecolor="white")
    b2 = ax.bar(x + w/2, f1_r, w, label=f"ResNet18 (macro-F1 {m_r['macro_f1']:.2f})",
                color="#2E7D32", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#1F2937")
    # delta annotation above each pair, colour-coded: green=positive, red=regression
    for xi, fb, fr in zip(x, f1_b, f1_r):
        delta = fr - fb
        col = "#2E7D32" if delta >= 0 else "#B22222"
        sign = "+" if delta >= 0 else ""
        ax.text(xi, max(fb, fr) + 0.09, f"{sign}{delta:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=col)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10); ax.set_ylabel("F1 score")
    ax.set_title(f"Per-class F1 - baseline vs ResNet18 ({args.split} split)", fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    out = fig_dir / f"compare_models_{args.split}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    logger.info("Saved -> %s", out)

    # Headline summary table to stdout (handy for the report)
    print(f"\n=== Summary ({args.split} split) ===")
    print(f"  baseline  : acc={m_b['accuracy']:.3f}  macroF1={m_b['macro_f1']:.3f}  ECE={m_b.get('ece',float('nan')):.3f}")
    print(f"  resnet18  : acc={m_r['accuracy']:.3f}  macroF1={m_r['macro_f1']:.3f}  ECE={m_r.get('ece',float('nan')):.3f}")
    print("  per-class F1 (baseline -> resnet18):")
    for c, a, b in zip(classes, f1_b, f1_r):
        print(f"    {c:24s} {a:.2f} -> {b:.2f}   ({b-a:+.2f})")


if __name__ == "__main__":
    main()
