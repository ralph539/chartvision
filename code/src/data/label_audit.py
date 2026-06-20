"""Heuristic label-noise audit: score every labelled window by how *strongly* it satisfies
the pattern's defining geometric feature, then summarise per class.

Limitation: this only measures rule-internal consistency, not whether a human would call it
textbook. But it gives a defensible 'label-quality score distribution' for the report, far
better than nothing.

Usage:
    python -m src.data.label_audit
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.rule_labelers import _local_maxima, _local_minima
from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _read_window(raw_dir: Path, ticker: str, start: int, size: int) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
    return df.iloc[start : start + size]


def salience(name: str, w: pd.DataFrame) -> float:
    H, L, C = w["High"].to_numpy(float), w["Low"].to_numpy(float), w["Close"].to_numpy(float)
    if name == "double_top":
        pks = _local_maxima(H, 6); wh = H.max(); best = 0.0
        for a in range(len(pks)):
            for b in range(a + 1, len(pks)):
                i, j = sorted((pks[a], pks[b])); lo = min(H[i], H[j])
                if j - i < 12 or abs(H[i] - H[j]) / max(H[i], H[j]) > 0.02: continue
                if (wh - lo) / wh > 0.02: continue
                trough = L[i + 1 : j].min() if j > i + 1 else lo
                best = max(best, (lo - trough) / lo)
        return best
    if name == "double_bottom":
        trs = _local_minima(L, 6); wl = L.min(); best = 0.0
        for a in range(len(trs)):
            for b in range(a + 1, len(trs)):
                i, j = sorted((trs[a], trs[b])); hi = max(L[i], L[j])
                if j - i < 12 or abs(L[i] - L[j]) / max(L[i], L[j]) > 0.02: continue
                if (hi - wl) / wl > 0.02: continue
                peak = H[i + 1 : j].max() if j > i + 1 else hi
                best = max(best, (peak - hi) / hi)
        return best
    if name == "head_and_shoulders":
        pks = _local_maxima(H, 4); best = 0.0
        for a in range(len(pks)):
            for b in range(a + 1, len(pks)):
                for c in range(b + 1, len(pks)):
                    ls, h, rs = pks[a], pks[b], pks[c]
                    if H[h] > H[ls] and H[h] > H[rs]:
                        best = max(best, min((H[h] - H[ls]) / H[h], (H[h] - H[rs]) / H[h]))
        return best
    if name == "bull_flag":
        base = np.minimum.accumulate(C)
        return float(((H - base) / base).max())
    return 0.0


def main() -> None:
    cfg = load_config()
    raw = resolve_path(cfg["paths"]["raw_dir"])
    size = cfg["window"]["size"]
    rep = resolve_path(cfg["paths"]["reports_dir"])
    fig = resolve_path(cfg["paths"]["figures_dir"])
    manifest = pd.read_csv(resolve_path(cfg["paths"]["processed_dir"]) / "manifest.csv")

    rows = []
    for r in manifest.itertuples():
        if r.label == "no_pattern":
            rows.append({"label": r.label, "salience": np.nan})
            continue
        w = _read_window(raw, r.ticker, int(r.start), size)
        rows.append({"label": r.label, "salience": salience(r.label, w)})
    audit = pd.DataFrame(rows)
    audit.to_csv(rep / "label_audit.csv", index=False)

    summary = audit.dropna().groupby("label")["salience"].describe()
    print("Salience score by class (higher = the pattern's defining feature is stronger):")
    print(summary.round(3))

    # box-plot per class
    pats = ["head_and_shoulders", "double_top", "double_bottom", "bull_flag"]
    data = [audit.loc[audit["label"] == p, "salience"].dropna().values for p in pats]
    figm, ax = plt.subplots(figsize=(8.5, 4))
    bp = ax.boxplot(data, labels=pats, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="#F7C24A",
                                   markeredgecolor="#1F2937", markersize=6))
    for box, col in zip(bp["boxes"], ["#1F4E79", "#1F4E79", "#1F4E79", "#1F4E79"]):
        box.set(facecolor=col, alpha=0.35, edgecolor=col)
    ax.set_ylabel("salience score (geometric strength)")
    ax.set_title("Label-quality audit: how strongly each labelled window satisfies the rule",
                 fontweight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout(); figm.savefig(fig / "label_audit_boxplot.png", dpi=150, facecolor="white")
    plt.close(figm)
    logger.info("Saved label-audit CSV + boxplot.")


if __name__ == "__main__":
    main()
