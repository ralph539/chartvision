"""Generate slide-ready visual assets for the final presentation.

All assets are produced at 16:9 or square at 300 dpi, with a slide-friendly aesthetic
(large text, high contrast, clean dark backgrounds where appropriate).

Outputs (../report_final/figures/slides/):
    title_dashboard.png        # cover-slide hero with the headline numbers
    pipeline_flow.png          # 6-stage horizontal pipeline diagram
    class_overview.png         # 5 patterns + bull/bear/neutral chips
    ablation_bars.png          # horizontal bar chart of the 5 training variants
    gradcam_hero.png           # one big annotated Grad-CAM example
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from src.data.dataset import make_loaders
from src.data.render_charts import render_window
from src.models.resnet18 import build_resnet18
from src.training.gradcam import GradCAM, last_conv_layer, _upsample
from src.utils.config import load_config, resolve_path

OUT = Path("/home/khairallah/Work_Environment/UPC/Computer_Vision/Short Project-20260512/report_final/figures/slides")
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()

# ---------- shared palette ----------
NAVY   = "#0A0E1A"
PURPLE = "#1B1233"
BLUE   = "#1F4E79"
GREEN  = "#22E36A"
RED    = "#FF3B5C"
YELLOW = "#F7C24A"
GREY   = "#94A3B8"
WHITE  = "#FFFFFF"


def _gradient_bg(ax, top: str = NAVY, bottom: str = PURPLE):
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bg", [bottom, top])
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1],
              origin="lower", interpolation="bilinear", zorder=-10)


# ---------- 1. Title dashboard ----------
def title_dashboard():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    _gradient_bg(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # main title
    ax.text(0.5, 0.78, "ChartVision", color=WHITE, fontsize=72,
            fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.66, "Candlestick Pattern Recognition  +  Grad-CAM Interpretability",
            color="#CBD5E1", fontsize=22, ha="center", va="center", style="italic")

    # stat tiles
    stats = [
        ("30",       "US tickers",          GREEN),
        ("3,417",    "labelled chart images", YELLOW),
        ("5",        "classes",             "#A78BFA"),
        ("60.2%",    "test accuracy",       "#60A5FA"),
        ("+0.13",    "macro-F1 vs baseline", GREEN),
    ]
    n = len(stats)
    w = 0.16; gap = 0.018
    total = n * w + (n - 1) * gap
    x0 = (1 - total) / 2
    y = 0.30
    for i, (big, small, col) in enumerate(stats):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, y), w, 0.22,
                             boxstyle="round,pad=0.005,rounding_size=0.012",
                             linewidth=2, edgecolor=col, facecolor=col, alpha=0.10)
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.155, big, color=col, fontsize=42,
                fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2, y + 0.06, small, color="#E5E7EB", fontsize=13,
                ha="center", va="center")

    # footer
    ax.text(0.5, 0.10, "Computer Vision Short Project  •  Master FIB UPC  •  Spring 2026",
            color="#94A3B8", fontsize=14, ha="center", va="center")
    ax.text(0.5, 0.055, "Ralph Khairallah  (Exchange  •  Group C)",
            color="#94A3B8", fontsize=12, ha="center", va="center", style="italic")
    fig.savefig(OUT / "title_dashboard.png", dpi=200, facecolor=NAVY)
    plt.close(fig)
    print("title_dashboard.png written")


# ---------- 2. Pipeline flow ----------
def pipeline_flow():
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    _gradient_bg(ax)
    ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")

    stages = [
        ("1", "Fetch OHLC",       "yfinance\n30 tickers", BLUE),
        ("2", "Rule-based labels","4 patterns\n+ negatives", BLUE),
        ("3", "Render charts",    "mplfinance\n224×224 PNG", BLUE),
        ("4", "Build splits",     "by date\ntrain<2022", BLUE),
        ("5", "Train",            "Baseline + ResNet18\nclass weights, aug", GREEN),
        ("6", "Evaluate",         "Grad-CAM\ncalibration, F1", "#A78BFA"),
    ]
    nx = len(stages); gap = 0.30
    w = (16 - (nx + 1) * gap) / nx; h = 2.7
    y = 1.4
    for i, (n, t, sub, col) in enumerate(stages):
        x = gap + i * (w + gap)
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.04,rounding_size=0.10",
                             linewidth=2.5, edgecolor=col, facecolor=col, alpha=0.15)
        ax.add_patch(box)
        ax.text(x + 0.12, y + h - 0.45, n, color=col, fontsize=22, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.95, t, color=WHITE, fontsize=15,
                fontweight="bold", ha="center")
        ax.text(x + w / 2, y + h / 2 - 0.30, sub, color="#CBD5E1",
                fontsize=11, ha="center", va="center")
        if i < nx - 1:
            xn = gap + (i + 1) * (w + gap)
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2),
                                          (xn - 0.02, y + h / 2),
                                          arrowstyle="-|>", mutation_scale=18,
                                          linewidth=2, color="#64748B"))
    ax.text(8, 0.55, "1 YAML config  •  fixed seed  •  fully reproducible",
            color="#94A3B8", fontsize=13, ha="center", style="italic")
    fig.savefig(OUT / "pipeline_flow.png", dpi=200, facecolor=NAVY)
    plt.close(fig)
    print("pipeline_flow.png written")


# ---------- 3. Class overview ----------
def class_overview():
    sample_dir = resolve_path(cfg["paths"]["figures_dir"]) / "samples_clean"
    panels = [
        ("head_and_shoulders", "Head & Shoulders", "Reversal",   "Bearish", RED),
        ("double_top",         "Double Top",       "Reversal",   "Bearish", RED),
        ("double_bottom",      "Double Bottom",    "Reversal",   "Bullish", GREEN),
        ("bull_flag",          "Bull Flag",        "Continuation","Bullish", GREEN),
        ("no_pattern",         "No Pattern",       "Negative class","Neutral", GREY),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6))
    fig.patch.set_facecolor(NAVY)
    for ax, (folder, name, family, bias, col) in zip(axes, panels):
        p = sample_dir / f"{folder}_0.png"
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#334155"); s.set_linewidth(1.5)
        ax.set_title(name, color=WHITE, fontsize=15, fontweight="bold", pad=10)
        ax.text(0.5, -0.10, family, transform=ax.transAxes, color="#CBD5E1",
                fontsize=11, ha="center", style="italic")
        ax.text(0.5, -0.20, f" {bias} ", transform=ax.transAxes,
                color=NAVY, fontsize=11, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=col, edgecolor="none"))
    plt.tight_layout()
    fig.savefig(OUT / "class_overview.png", dpi=200, facecolor=NAVY)
    plt.close(fig)
    print("class_overview.png written")


# ---------- 4. Ablation bars ----------
def ablation_bars():
    variants = [
        ("ResNet18 fine-tune (aug + class weights)", 0.602, True),
        ("ResNet18 fine-tune (layer4 + FC)",         0.543, False),
        ("Baseline CNN (strong aug + class weights)", 0.482, False),
        ("Baseline CNN (light aug, no weights)",     0.471, False),
        ("ResNet18 linear probe (FC only)",          0.306, False),
    ]
    fig = plt.figure(figsize=(16, 7.5))
    ax = fig.add_axes([0.30, 0.08, 0.66, 0.84])
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    names  = [v[0] for v in variants]
    scores = [v[1] for v in variants]
    cols   = [GREEN if v[2] else "#4F6385" for v in variants]
    y = np.arange(len(variants))
    bars = ax.barh(y, scores, color=cols, edgecolor=WHITE, linewidth=1.2, height=0.62)
    for i, (b, s) in enumerate(zip(bars, scores)):
        ax.text(s + 0.012, b.get_y() + b.get_height() / 2, f"{s:.3f}",
                color=WHITE, fontsize=18, fontweight="bold", va="center")
    ax.set_yticks(y); ax.set_yticklabels(names, color=WHITE, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("Test-set macro-F1", color="#CBD5E1", fontsize=14)
    ax.tick_params(axis="x", colors="#CBD5E1", labelsize=12)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#334155")
    ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Ablation: which training regime won?", color=WHITE,
                 fontsize=22, fontweight="bold", pad=18, loc="left")
    # winner chip well to the right of the winning score
    ax.text(0.78, 0, "WINNER", color=NAVY, fontsize=12, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, edgecolor="none"))
    fig.savefig(OUT / "ablation_bars.png", dpi=200, facecolor=NAVY)
    plt.close(fig)
    print("ablation_bars.png written")


# ---------- 5. Grad-CAM hero ----------
def gradcam_hero():
    ckpt = torch.load(resolve_path("checkpoints/resnet18_best.pt"),
                      map_location="cpu", weights_only=False)
    classes, size, name = ckpt["classes"], ckpt["image_size"], ckpt["model"]
    model = build_resnet18(len(classes), mode="finetune")
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    cam = GradCAM(model, last_conv_layer(model, name))
    _, _, te, _ = make_loaders(cfg, size, batch_size=1, num_workers=0)
    # find a high-confidence correct head_and_shoulders prediction
    target = classes.index("head_and_shoulders")
    best = None
    for x, y in te:
        if int(y.item()) != target: continue
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1).squeeze(0)
            pred = int(p.argmax().item()); conf = float(p[pred].item())
        if pred == target and conf > 0.7:
            heat = cam(x, pred)
            img = x.squeeze(0).permute(1, 2, 0).numpy()
            heat_up = _upsample(heat, img.shape[:2])
            if best is None or conf > best[2]:
                best = (img, heat_up, conf)
            if conf > 0.9:
                break
    if best is None:
        print("no good gradcam example found")
        return
    img, heat_up, conf = best

    fig = plt.figure(figsize=(16, 8.5))
    ax_bg = fig.add_axes([0, 0, 1, 1]); _gradient_bg(ax_bg); ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1); ax_bg.axis("off")
    # title FIRST so axes don't overlap it
    fig.text(0.5, 0.92, "Grad-CAM on a correctly classified Head & Shoulders",
             color=WHITE, fontsize=24, fontweight="bold", ha="center")
    fig.text(0.5, 0.86, f"ResNet18 prediction confidence  {conf*100:.0f}%",
             color="#CBD5E1", fontsize=15, ha="center", style="italic")
    ax_l = fig.add_axes([0.06, 0.16, 0.40, 0.62])
    ax_r = fig.add_axes([0.54, 0.16, 0.40, 0.62])
    for a, title in [(ax_l, "Input chart"), (ax_r, "Where the model looked")]:
        a.imshow(img)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#334155"); s.set_linewidth(1.5)
        a.set_title(title, color=WHITE, fontsize=16, fontweight="bold", pad=8)
    ax_r.imshow(heat_up, cmap="jet", alpha=0.55)
    fig.text(0.5, 0.07,
             "The heatmap localises on the 'head' (the highest central peak),",
             color="#CBD5E1", fontsize=13, ha="center", style="italic")
    fig.text(0.5, 0.035,
             "validating that the network is using the pattern's defining structure.",
             color="#CBD5E1", fontsize=13, ha="center", style="italic")
    fig.savefig(OUT / "gradcam_hero.png", dpi=200, facecolor=NAVY)
    plt.close(fig)
    print("gradcam_hero.png written")


if __name__ == "__main__":
    title_dashboard()
    pipeline_flow()
    class_overview()
    ablation_bars()
    gradcam_hero()
    print("\nAll slide assets in:", OUT)
