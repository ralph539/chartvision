"""Evaluate a trained checkpoint on a split: metrics + confusion matrix + metrics.json.

Usage:
    python -m src.training.evaluate --checkpoint checkpoints/baseline_best.pt --split val
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data.dataset import make_loaders
from src.models.baseline_cnn import build_baseline
from src.models.resnet18 import build_resnet18
from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt: dict, num_classes: int):
    name = ckpt["model"]
    model = build_baseline(num_classes) if name == "baseline" else build_resnet18(num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, name


@torch.no_grad()
def predict(model, loader, device):
    ys, ps, confs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(1)
        ys.append(y.numpy()); ps.append(pred.cpu().numpy()); confs.append(conf.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(confs)


def reliability_curve(y_true, y_pred, confs, n_bins: int = 10):
    """Compute the reliability (calibration) curve and the Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    correct = (y_true == y_pred).astype(float)
    centres, accs, freqs = [], [], []
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        c = float(correct[mask].mean())
        m = float(confs[mask].mean())
        centres.append(m); accs.append(c); freqs.append(mask.sum() / n)
        ece += (mask.sum() / n) * abs(c - m)
    return np.array(centres), np.array(accs), np.array(freqs), float(ece)


def plot_reliability(centres, accs, freqs, ece: float, name: str, split: str, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.7, label="perfect calibration")
    ax.bar(centres, accs, width=0.08, color="#1F4E79", alpha=0.8, edgecolor="white",
           label="empirical accuracy")
    ax.scatter(centres, accs, s=20 + 800 * freqs, color="#F7C24A", edgecolor="black",
               zorder=3, label="bin (size $\\propto$ # samples)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability - {name} ({split})  ·  ECE = {ece:.3f}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out_path, dpi=150, facecolor="white"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    classes = ckpt["classes"]
    image_size = ckpt["image_size"]

    tr, va, te, _ = make_loaders(cfg, image_size, batch_size=32, num_workers=2)
    loader = {"val": va, "test": te}[args.split]

    model, name = load_model(ckpt, len(classes))
    model.to(device)
    y_true, y_pred, confs = predict(model, loader, device)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    logger.info("[%s | %s] accuracy=%.3f  macro-F1=%.3f", name, args.split, acc, macro_f1)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{name} confusion matrix ({args.split})  ·  acc={acc:.2f}, macroF1={macro_f1:.2f}")
    plt.tight_layout()
    fig_dir = resolve_path(cfg["paths"]["figures_dir"])
    fig.savefig(fig_dir / f"confusion_{name}_{args.split}.png", dpi=150, facecolor="white")
    plt.close(fig)

    # reliability / ECE
    centres, accs, freqs, ece = reliability_curve(y_true, y_pred, confs, n_bins=10)
    plot_reliability(centres, accs, freqs, ece, name, args.split,
                     fig_dir / f"reliability_{name}_{args.split}.png")
    logger.info("ECE = %.3f", ece)

    # dump metrics
    metrics = {"model": name, "split": args.split, "accuracy": acc, "macro_f1": macro_f1,
               "ece": ece, "per_class": report}
    out = resolve_path(cfg["paths"]["reports_dir"]) / f"metrics_{name}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics -> %s + confusion matrix + reliability curve -> figures/", out)


if __name__ == "__main__":
    main()
