"""Regenerate the report's sample figure with the *clearest* example per class.

For each pattern we score every labelled window by how strongly it shows the pattern's
defining feature (deep trough, tall head, big pole, ...) and keep the top two. This makes the
figure actually teach what each class looks like, instead of showing random noisy examples.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.rule_labelers import _local_maxima, _local_minima
from src.data.render_charts import render_window
from src.utils.config import load_config, resolve_path

cfg = load_config()
SIZE = cfg["window"]["size"]
RAW = resolve_path(cfg["paths"]["raw_dir"])
OUT = resolve_path(cfg["paths"]["figures_dir"]) / "samples_clean"
OUT.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(resolve_path(cfg["paths"]["processed_dir"]) / "manifest.csv")

_dfcache: dict[str, pd.DataFrame] = {}


def get_window(ticker: str, start: int) -> pd.DataFrame:
    if ticker not in _dfcache:
        _dfcache[ticker] = pd.read_csv(RAW / f"{ticker}.csv", index_col=0, parse_dates=True)
    return _dfcache[ticker].iloc[start : start + SIZE]


def salience(name: str, w: pd.DataFrame) -> float:
    """Higher = the pattern's defining feature stands out more clearly."""
    H, L, C = w["High"].to_numpy(float), w["Low"].to_numpy(float), w["Close"].to_numpy(float)
    if name == "double_top":
        pks = _local_maxima(H, 5)
        wh = H.max()
        best = -1.0
        for a in range(len(pks)):
            for b in range(a + 1, len(pks)):
                i, j = sorted((pks[a], pks[b]))
                h1, h2 = H[i], H[j]
                lo = min(h1, h2)
                if j - i < 10 or abs(h1 - h2) / max(h1, h2) > 0.04:
                    continue                              # need two ~equal tops, apart
                if (wh - lo) / wh > 0.03:
                    continue                              # the pair must be the window top
                trough = L[i + 1 : j].min() if j > i + 1 else lo
                equality = 1 - abs(h1 - h2) / max(h1, h2)
                best = max(best, (lo - trough) / lo * equality)
        return best                                       # deep valley + equal tops at the top
    if name == "double_bottom":
        trs = _local_minima(L, 5)
        wl = L.min()
        best = -1.0
        for a in range(len(trs)):
            for b in range(a + 1, len(trs)):
                i, j = sorted((trs[a], trs[b]))
                l1, l2 = L[i], L[j]
                hi = max(l1, l2)
                if j - i < 10 or abs(l1 - l2) / max(l1, l2) > 0.04:
                    continue
                if (hi - wl) / wl > 0.03:
                    continue                              # the pair must be the window floor
                peak = H[i + 1 : j].max() if j > i + 1 else hi
                equality = 1 - abs(l1 - l2) / max(l1, l2)
                best = max(best, (peak - hi) / hi * equality)
        return best
    if name == "head_and_shoulders":
        pks = _local_maxima(H, 4)
        best = -1.0
        for a in range(len(pks)):
            for b in range(a + 1, len(pks)):
                for c in range(b + 1, len(pks)):
                    ls, h, rs = pks[a], pks[b], pks[c]
                    if H[h] > H[ls] and H[h] > H[rs]:
                        best = max(best, min((H[h] - H[ls]) / H[h], (H[h] - H[rs]) / H[h]))
        return best                                      # taller head vs shoulders
    if name == "bull_flag":
        base = np.minimum.accumulate(C)
        return float(((H - base) / base).max())          # bigger pole
    return 0.0


def pick(name: str, k: int = 2) -> list[tuple[str, int]]:
    sub = manifest[manifest["label"] == name]
    if name == "no_pattern":
        s = sub.sample(k, random_state=1)
        return list(zip(s["ticker"], s["start"]))
    scored = [(salience(name, get_window(r.ticker, r.start)), r.ticker, r.start)
              for r in sub.itertuples()]
    scored.sort(key=lambda x: -x[0])
    return [(t, int(st)) for _, t, st in scored[:k]]


ORDER = ["head_and_shoulders", "double_top", "double_bottom", "bull_flag", "no_pattern"]
picks = {name: pick(name) for name in ORDER}

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for col, name in enumerate(ORDER):
    for row in range(2):
        ticker, start = picks[name][row]
        w = get_window(ticker, start)
        p = OUT / f"{name}_{row}.png"
        render_window(w, p, cfg)
        ax = axes[row, col]
        ax.imshow(Image.open(p))
        ax.axis("off")
        ax.set_title(name if row == 0 else f"{name} (ex. 2)", fontsize=10, color="#1F2937")
plt.tight_layout(pad=0.5)
fig.savefig(resolve_path("../report_checkpoint/figures/samples.png"), dpi=110, facecolor="white")
print("clean samples.png written")
for name in ORDER:
    print(f"  {name}: {picks[name]}")
