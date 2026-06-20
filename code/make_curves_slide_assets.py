"""Generate slide-styled training curve PNGs for slide 8 of the final deck.

Reads history_*.csv from reports/ and writes two polished PNGs to
slides_kit/custom_assets/:
  - curves_baseline_dark.png
  - curves_resnet18_dark.png

Style matches the rest of the deck (dark navy bg, deck palette,
bold sans-serif titles, soft grid).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUT = ROOT.parent / "slides_kit" / "custom_assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY_TOP = "#0A0E1A"
NAVY_BOT = "#1B1233"
PANEL = "#111729"
WHITE = "#FFFFFF"
GREY = "#CBD5E1"
GRID = "#28324A"
TRAIN = "#7CC3FF"
VAL = "#F7C24A"
GREEN = "#22E36A"
RED = "#FF3B5C"


def _gradient_bg(ax_or_fig, w=2000, h=1125):
    grad = np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1)
    top = np.array([0x0A, 0x0E, 0x1A]) / 255
    bot = np.array([0x1B, 0x12, 0x33]) / 255
    img = top[None, None, :] * (1 - grad[..., None]) + bot[None, None, :] * grad[..., None]
    ax_or_fig.imshow(img, extent=(0, 1, 0, 1), transform=ax_or_fig.transAxes,
                     aspect="auto", zorder=-100)


def _style_axes(ax, *, ylabel: str):
    ax.set_facecolor((1, 1, 1, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=GREY, labelsize=15, length=4, width=1.1)
    ax.set_xlabel("epoch", color=GREY, fontsize=16, labelpad=8)
    ax.set_ylabel(ylabel, color=GREY, fontsize=16, labelpad=10)
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax):
    leg = ax.legend(loc="best", frameon=True, fontsize=14,
                    labelcolor=WHITE)
    leg.get_frame().set_facecolor("#0E1426")
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_alpha(0.85)


def _callout(fig, text: str, color: str, *, y: float = 0.04):
    fig.text(0.5, y, text, color=color, fontsize=16, ha="center", va="bottom",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.6",
                       facecolor="#1A1024", edgecolor=color,
                       linewidth=1.4))


def render_curves(csv_path: Path, out_path: Path, *, title: str,
                  subtitle: str, callout: str, callout_color: str):
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].to_numpy()

    fig = plt.figure(figsize=(16, 9), facecolor=NAVY_TOP)
    _gradient_bg(fig)

    # Title block
    fig.text(0.5, 0.93, title, color=WHITE, fontsize=30, ha="center",
             fontweight="bold")
    fig.text(0.5, 0.875, subtitle, color=GREY, fontsize=17, ha="center",
             style="italic")

    # Two axes
    ax_loss = fig.add_axes([0.07, 0.20, 0.42, 0.58])
    ax_acc = fig.add_axes([0.54, 0.20, 0.42, 0.58])

    # Loss
    ax_loss.plot(epochs, df["train_loss"], color=TRAIN, linewidth=2.6,
                 marker="o", markersize=5, label="train")
    ax_loss.plot(epochs, df["val_loss"], color=VAL, linewidth=2.6,
                 marker="o", markersize=5, label="val")
    ax_loss.set_title("Loss", color=WHITE, fontsize=20, fontweight="bold",
                      pad=14)
    _style_axes(ax_loss, ylabel="cross-entropy")
    _legend(ax_loss)

    # Accuracy
    ax_acc.plot(epochs, df["train_acc"], color=TRAIN, linewidth=2.6,
                marker="o", markersize=5, label="train")
    ax_acc.plot(epochs, df["val_acc"], color=VAL, linewidth=2.6,
                marker="o", markersize=5, label="val")
    ax_acc.set_title("Accuracy", color=WHITE, fontsize=20, fontweight="bold",
                     pad=14)
    _style_axes(ax_acc, ylabel="accuracy")
    _legend(ax_acc)

    _callout(fig, callout, callout_color, y=0.04)

    fig.savefig(out_path, dpi=160, facecolor=NAVY_TOP)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    render_curves(
        REPORTS / "history_baseline.csv",
        OUT / "curves_baseline_dark.png",
        title="Baseline CNN training curves",
        subtitle="30 epochs · trains smoothly · loss flattens after epoch 20",
        callout="Loss plateaus and val accuracy stalls near 0.55 - capacity ceiling, not overfitting.",
        callout_color=TRAIN,
    )
    render_curves(
        REPORTS / "history_resnet18_aug.csv",
        OUT / "curves_resnet18_dark.png",
        title="ResNet18 training curves (aug + class weights)",
        subtitle="11 epochs · early-stopped at epoch 7",
        callout="Val loss rises after epoch 1 while train loss keeps falling - classic over-fitting.",
        callout_color=RED,
    )


if __name__ == "__main__":
    main()
