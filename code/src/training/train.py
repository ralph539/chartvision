"""Train a model (baseline CNN or ResNet18) with early stopping; save checkpoint + curves.

Usage:
    python -m src.training.train --model baseline
    python -m src.training.train --model resnet18
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data.dataset import compute_class_weights, make_loaders
from src.models.baseline_cnn import build_baseline
from src.models.resnet18 import build_resnet18
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "baseline":
        return build_baseline(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, mode="finetune")
    if name == "resnet18_linear":
        return build_resnet18(num_classes, mode="linear")
    raise ValueError(f"unknown model {name}")


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over loader. If optimizer is given, train; else evaluate. Returns (loss, acc)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_curves(history: list[dict], out_path: Path, model_name: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "-o", label="train", color="#1F4E79", ms=3)
    ax1.plot(epochs, [h["val_loss"] for h in history], "-o", label="val", color="#B22222", ms=3)
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["train_acc"] for h in history], "-o", label="train", color="#1F4E79", ms=3)
    ax2.plot(epochs, [h["val_acc"] for h in history], "-o", label="val", color="#2E7D32", ms=3)
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"{model_name} training curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline",
                        choices=["baseline", "resnet18", "resnet18_linear"])
    parser.add_argument("--augment", default="light", choices=["light", "strong"])
    parser.add_argument("--class-weights", action="store_true",
                        help="weight cross-entropy by inverse class frequency")
    parser.add_argument("--tag", default="", help="optional suffix for the checkpoint filename")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tcfg = cfg["training"][args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training %s on %s  (augment=%s  class_weights=%s)",
                args.model, device, args.augment, args.class_weights)

    train_loader, val_loader, _, classes = make_loaders(
        cfg, tcfg["image_size"], tcfg["batch_size"], tcfg["num_workers"],
        augment=args.augment,
    )
    logger.info("Train batches=%d  Val batches=%d  classes=%s",
                len(train_loader), len(val_loader), classes)

    model = build_model(args.model, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: %d", n_params)

    if args.class_weights:
        w = compute_class_weights(cfg, classes).to(device)
        logger.info("Class weights: %s", w.cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    ckpt_dir = resolve_path(cfg["paths"]["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = resolve_path(cfg["paths"]["figures_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = resolve_path(cfg["paths"]["reports_dir"])
    tag = f"_{args.tag}" if args.tag else ""
    best_path = ckpt_dir / f"{args.model}{tag}_best.pt"

    history, best_val, patience = [], float("inf"), 0
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        logger.info("epoch %02d  train_loss=%.3f acc=%.3f  val_loss=%.3f acc=%.3f  (%.1fs)",
                    epoch, tr_loss, tr_acc, va_loss, va_acc, time.time() - t0)

        if va_loss < best_val - 1e-4:
            best_val, patience = va_loss, 0
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "classes": classes, "image_size": tcfg["image_size"]}, best_path)
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # save history CSV + curves
    with open(rep_dir / f"history_{args.model}{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writeheader(); w.writerows(history)
    plot_curves(history, fig_dir / f"curves_{args.model}{tag}.png", f"{args.model}{tag}")
    logger.info("Best val loss=%.3f  checkpoint=%s", best_val, best_path)


if __name__ == "__main__":
    main()
