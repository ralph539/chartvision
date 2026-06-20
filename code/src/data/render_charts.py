"""Render OHLC windows into candlestick PNGs.

Two modes:
  - "dataset"  : the images the CNN trains on. Catchy but CONSISTENT (gradient background +
                 glowing candles, applied identically to every image so no label leaks).
                 No axes, text, volume, or annotations.
  - "showcase" : eye-candy for the report / slides / video. Adds volume bars, a moving-average
                 line, and optional pattern annotations. Higher resolution.

Candles are drawn directly with matplotlib (not mplfinance) so we control the glow/gradient.

Usage:
    python -m src.data.render_charts                 # ~20 dataset-style samples to inspect
    python -m src.data.render_charts --showcase      # also render a few showcase charts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.data.rule_labelers import double_top
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _gradient_background(ax, top: str, bottom: str, xlim, ylim) -> None:
    """Paint a smooth vertical gradient behind the candles."""
    cmap = LinearSegmentedColormap.from_list("bg", [bottom, top])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(
        grad, aspect="auto", cmap=cmap, origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=0, interpolation="bilinear",
    )


def _glow(color: str, layers: int, max_lw: float, alpha: float):
    """Return path-effects that fake a soft glow by stacking translucent strokes."""
    eff = []
    for k in range(layers, 0, -1):
        eff.append(pe.Stroke(linewidth=max_lw * k / layers, foreground=color, alpha=alpha))
    eff.append(pe.Normal())
    return eff


def _draw_candles(ax, o, h, l, c, cfg: dict, glow: bool) -> None:
    """Draw candlesticks (wick line + body rectangle) with optional glow."""
    r = cfg["render"]
    up, down, wick = r["up_color"], r["down_color"], r["wick_color"]
    w = r["candle_width"]
    n = len(c)
    for i in range(n):
        color = up if c[i] >= o[i] else down
        # wick
        line, = ax.plot([i, i], [l[i], h[i]], color=wick, linewidth=0.8, solid_capstyle="round", zorder=2)
        # body
        body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
        height = max(body_hi - body_lo, (h[i] - l[i]) * 0.001)  # avoid zero-height doji
        rect = Rectangle((i - w / 2, body_lo), w, height, facecolor=color,
                         edgecolor=color, linewidth=0.5, zorder=3)
        ax.add_patch(rect)
        if glow:
            g = _glow(color, r["glow_layers"], r["glow_max_linewidth"], r["glow_alpha"])
            rect.set_path_effects(g)
            line.set_path_effects(_glow(wick, r["glow_layers"], r["glow_max_linewidth"] * 0.5, r["glow_alpha"] * 0.6))


def render_window(window: pd.DataFrame, out_path: Path, cfg: dict) -> None:
    """Render one OHLC window to a dataset-style PNG of exactly image_size x image_size."""
    r = cfg["render"]
    size, dpi = r["image_size"], r["dpi"]
    o = window["Open"].to_numpy(float)
    h = window["High"].to_numpy(float)
    low = window["Low"].to_numpy(float)
    c = window["Close"].to_numpy(float)

    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the whole figure, no margins
    pad = 0.04 * (h.max() - low.min())
    xlim = (-1, len(c))
    ylim = (low.min() - pad, h.max() + pad)
    _gradient_background(ax, r["bg_top"], r["bg_bottom"], xlim, ylim)
    _draw_candles(ax, o, h, low, c, cfg, glow=r["glow"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def render_showcase(window: pd.DataFrame, out_path: Path, cfg: dict,
                    title: str | None = None, annotate: list[tuple[int, str]] | None = None) -> None:
    """Render a high-res, annotated showcase chart for the report/slides (NOT for training)."""
    r = cfg["render"]
    o = window["Open"].to_numpy(float)
    h = window["High"].to_numpy(float)
    low = window["Low"].to_numpy(float)
    c = window["Close"].to_numpy(float)
    vol = window["Volume"].to_numpy(float) if "Volume" in window else None

    fig = plt.figure(figsize=(10, 5.5), dpi=140)
    ax = fig.add_axes([0.02, 0.22 if vol is not None else 0.02, 0.96, 0.70 if title else 0.76])
    pad = 0.06 * (h.max() - low.min())
    xlim = (-1, len(c))
    ylim = (low.min() - pad, h.max() + pad)
    _gradient_background(ax, r["bg_top"], r["bg_bottom"], xlim, ylim)
    _draw_candles(ax, o, h, low, c, cfg, glow=True)

    # moving average (20-bar) as a glowing line
    if len(c) >= 20:
        ma = pd.Series(c).rolling(20).mean().to_numpy()
        line, = ax.plot(range(len(c)), ma, color="#FACC15", linewidth=1.8, zorder=4, alpha=0.9)
        line.set_path_effects(_glow("#FACC15", 3, 7, 0.12))

    # annotations (e.g. mark the two tops)
    if annotate:
        for idx, label in annotate:
            ax.scatter([idx], [h[idx]], s=120, facecolors="none",
                       edgecolors="#FFFFFF", linewidths=1.8, zorder=6)
            ax.annotate(label, (idx, h[idx]), textcoords="offset points", xytext=(0, 12),
                        ha="center", color="#FFFFFF", fontsize=11, fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        fig.text(0.04, 0.94, title, color="#E5E7EB", fontsize=15, fontweight="bold")

    # volume bars in a thin panel underneath
    if vol is not None:
        axv = fig.add_axes([0.02, 0.04, 0.96, 0.15])
        _gradient_background(axv, r["bg_top"], r["bg_bottom"], (-1, len(c)), (0, vol.max() * 1.1))
        colors = [r["up_color"] if c[i] >= o[i] else r["down_color"] for i in range(len(c))]
        axv.bar(range(len(c)), vol, color=colors, width=0.7, alpha=0.8)
        axv.set_xlim(-1, len(c))
        axv.set_ylim(0, vol.max() * 1.1)
        axv.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=r["bg_top"])
    plt.close(fig)


def _scan_for_double_tops(df: pd.DataFrame, cfg: dict) -> list[int]:
    """Start indices of windows where the double_top rule fires (after de-duplication)."""
    from src.data.rule_labelers import dedup_hits
    size = cfg["window"]["size"]
    stride = cfg["window"]["stride"]
    params = cfg["labelers"]["double_top"]
    hits = []
    for start in range(0, len(df) - size, stride):
        window = df.iloc[start : start + size]
        is_pat, anchor = double_top(window, params)
        if is_pat:
            hits.append((start, start + anchor))
    return dedup_hits(hits, min_gap=size // 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sample candlestick charts.")
    parser.add_argument("--n", type=int, default=20, help="number of dataset-style samples")
    parser.add_argument("--showcase", action="store_true", help="also render showcase charts")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["figures_dir"]) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["window"]["size"]
    tickers = cfg["data"]["smoke_test_tickers"]

    n_dt, n_rand = 0, 0
    for ticker in tickers:
        csv = raw_dir / f"{ticker}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        hits = _scan_for_double_tops(df, cfg)
        logger.info("%s: %d double_top windows (after dedup)", ticker, len(hits))
        for start, _ in hits:
            if n_dt >= args.n // 2:
                break
            window = df.iloc[start : start + size]
            render_window(window, out_dir / f"double_top__{ticker}_{window.index[0].date()}.png", cfg)
            n_dt += 1

    while n_dt + n_rand < args.n:
        ticker = np.random.choice(tickers)
        df = pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        start = int(np.random.randint(0, len(df) - size))
        window = df.iloc[start : start + size]
        is_pat, _ = double_top(window, cfg["labelers"]["double_top"])
        label = "double_top" if is_pat else "no_pattern"
        render_window(window, out_dir / f"{label}__{ticker}_{window.index[0].date()}_rand.png", cfg)
        n_rand += 1

    logger.info("Rendered %d dataset samples to %s", n_dt + n_rand, out_dir)

    if args.showcase:
        sc_dir = resolve_path(cfg["paths"]["figures_dir"]) / "showcase"
        df = pd.read_csv(raw_dir / f"{tickers[0]}.csv", index_col=0, parse_dates=True)
        hits = _scan_for_double_tops(df, cfg)
        if hits:
            start, anchor = hits[0]
            window = df.iloc[start : start + size]
            _, a = double_top(window, cfg["labelers"]["double_top"])
            render_showcase(window, sc_dir / "showcase_double_top.png", cfg,
                            title=f"Double Top  ·  {tickers[0]}  ·  {window.index[0].date()}",
                            annotate=[(a, "Top 2")])
            logger.info("Rendered showcase chart to %s", sc_dir)


if __name__ == "__main__":
    main()
