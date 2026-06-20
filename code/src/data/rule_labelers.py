"""Rule-based pattern labelers operating on a rolling window of OHLC bars.

Each labeler takes a window (pandas DataFrame with Open/High/Low/Close) and a params dict, and
returns ``(is_pattern, anchor_index)``:
    is_pattern   -- True if the pattern is present
    anchor_index -- the bar index (within the window) that "confirms" the pattern, else -1.

Criteria are deliberately conservative and fully geometric, so they are explainable and
unit-testable. Thresholds come from configs/default.yaml (labelers.<pattern>).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------- helpers --------------------------------------------

def _local_maxima(values: np.ndarray, order: int) -> list[int]:
    """Indices of local maxima (highest within +/-order bars), with plateaus collapsed."""
    n = len(values)
    raw = [i for i in range(n) if values[i] == values[max(0, i - order):min(n, i + order + 1)].max()]
    return _collapse(raw, values, order, prefer_max=True)


def _local_minima(values: np.ndarray, order: int) -> list[int]:
    """Indices of local minima (lowest within +/-order bars), with plateaus collapsed."""
    n = len(values)
    raw = [i for i in range(n) if values[i] == values[max(0, i - order):min(n, i + order + 1)].min()]
    return _collapse(raw, values, order, prefer_max=False)


def _collapse(raw: list[int], values: np.ndarray, order: int, prefer_max: bool) -> list[int]:
    """Collapse runs of indices within `order` of each other to the most extreme one."""
    if not raw:
        return []
    pick = (lambda ks: max(ks, key=lambda k: values[k])) if prefer_max else (lambda ks: min(ks, key=lambda k: values[k]))
    out, cluster = [], [raw[0]]
    for p in raw[1:]:
        if p - cluster[-1] <= order:
            cluster.append(p)
        else:
            out.append(pick(cluster))
            cluster = [p]
    out.append(pick(cluster))
    return out


def dedup_hits(hits: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    """Non-max suppression on overlapping window hits.

    Consecutive rolling windows share most of their bars, so one real pattern is detected many
    times. Keep hits whose start indices are at least ``min_gap`` bars apart.
    """
    if not hits:
        return []
    hits = sorted(hits, key=lambda x: x[0])
    kept = [hits[0]]
    for start, anchor in hits[1:]:
        if start - kept[-1][0] >= min_gap:
            kept.append((start, anchor))
    return kept


# ----------------------------- labelers -------------------------------------------

def double_top(window: pd.DataFrame, params: dict) -> tuple[bool, int]:
    """Detect a Double Top (two equal-height tops, trough between, reversal afterwards).

    Criteria:
      1. Two peaks separated by >= ``min_peak_separation`` bars.
      2. Peaks within ``peak_tolerance`` of each other in height.
      3. Trough (lowest Low) between them >= ``min_trough_drop`` below the lower peak.
      4. No higher high between the peaks.
      5. The peaks are the top of the window (within ``top_margin`` of the window's max high)
         -- rules out a bump in the middle of an ongoing uptrend.
      6. Price drops >= ``confirm_drop`` below the lower peak after the 2nd peak (reversal).
    Anchor: the second peak.
    """
    high = window["High"].to_numpy(float)
    low = window["Low"].to_numpy(float)
    n = len(high)
    if n < params["min_peak_separation"] + 2:
        return False, -1

    tol, min_drop = params["peak_tolerance"], params["min_trough_drop"]
    min_sep, top_margin, confirm = params["min_peak_separation"], params["top_margin"], params["confirm_drop"]
    order = max(2, min_sep // 2)
    window_high = high.max()

    peaks = _local_maxima(high, order)
    if len(peaks) < 2:
        return False, -1

    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            i, j = sorted((peaks[a], peaks[b]))
            h1, h2 = high[i], high[j]
            lower = min(h1, h2)
            if j - i < min_sep:
                continue
            if abs(h1 - h2) / max(h1, h2) > tol:
                continue
            trough = low[i + 1:j].min() if j > i + 1 else lower
            if (lower - trough) / lower < min_drop:
                continue
            if (high[i + 1:j].max() if j > i + 1 else 0.0) > lower * (1 + tol):
                continue
            # 5. peaks are the window top
            if (window_high - lower) / window_high > top_margin:
                continue
            # 6. confirmation drop after the 2nd peak
            after = low[j + 1:]
            if after.size == 0 or (lower - after.min()) / lower < confirm:
                continue
            return True, int(j)
    return False, -1


def double_bottom(window: pd.DataFrame, params: dict) -> tuple[bool, int]:
    """Detect a Double Bottom: the mirror of Double Top (two equal lows, peak between, rally).

    Anchor: the second bottom.
    """
    high = window["High"].to_numpy(float)
    low = window["Low"].to_numpy(float)
    n = len(low)
    if n < params["min_trough_separation"] + 2:
        return False, -1

    tol, min_rise = params["trough_tolerance"], params["min_peak_rise"]
    min_sep, bot_margin, confirm = params["min_trough_separation"], params["bottom_margin"], params["confirm_rise"]
    order = max(2, min_sep // 2)
    window_low = low.min()

    troughs = _local_minima(low, order)
    if len(troughs) < 2:
        return False, -1

    for a in range(len(troughs)):
        for b in range(a + 1, len(troughs)):
            i, j = sorted((troughs[a], troughs[b]))
            l1, l2 = low[i], low[j]
            higher = max(l1, l2)
            if j - i < min_sep:
                continue
            if abs(l1 - l2) / max(l1, l2) > tol:
                continue
            peak = high[i + 1:j].max() if j > i + 1 else higher
            if (peak - higher) / higher < min_rise:
                continue
            if (low[i + 1:j].min() if j > i + 1 else np.inf) < higher * (1 - tol):
                continue
            # bottoms are the floor of the window
            if (higher - window_low) / window_low > bot_margin:
                continue
            # confirmation rally after the 2nd bottom
            after = high[j + 1:]
            if after.size == 0 or (after.max() - higher) / higher < confirm:
                continue
            return True, int(j)
    return False, -1


def head_and_shoulders(window: pd.DataFrame, params: dict) -> tuple[bool, int]:
    """Detect Head & Shoulders: left shoulder < head > right shoulder, shoulders ~equal.

    Criteria:
      1. Three peaks LS < H > RS with >= ``min_separation`` bars between consecutive peaks.
      2. Head >= ``head_prominence`` above BOTH shoulders.
      3. Shoulders within ``shoulder_tolerance`` of each other.
      4. The head is the window's highest high (within ``top_margin``).
    Anchor: the right shoulder.
    """
    high = window["High"].to_numpy(float)
    n = len(high)
    sep, head_prom = params["min_separation"], params["head_prominence"]
    sh_tol, top_margin = params["shoulder_tolerance"], params["top_margin"]
    if n < 3 * sep:
        return False, -1
    order = max(2, sep // 2)
    window_high = high.max()

    peaks = _local_maxima(high, order)
    if len(peaks) < 3:
        return False, -1

    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            for c in range(b + 1, len(peaks)):
                ls, h, rs = peaks[a], peaks[b], peaks[c]
                if h - ls < sep or rs - h < sep:
                    continue
                hl, hh, hr = high[ls], high[h], high[rs]
                # head is the highest, and the window's top
                if hh <= hl or hh <= hr:
                    continue
                if (window_high - hh) / window_high > top_margin:
                    continue
                if (hh - hl) / hh < head_prom or (hh - hr) / hh < head_prom:
                    continue
                if abs(hl - hr) / max(hl, hr) > sh_tol:
                    continue
                return True, int(rs)
    return False, -1


def bull_flag(window: pd.DataFrame, params: dict) -> tuple[bool, int]:
    """Detect a Bull Flag: a strong rise (pole) then a tight downward/sideways consolidation.

    Criteria:
      1. Pole: a >= ``pole_gain`` rise within <= ``pole_max_len`` bars, ending at a local high.
      2. Flag: the following ``flag_min_len``..``flag_max_len`` bars stay in a tight band
         (high-low range <= ``flag_max_range`` of the pole top) and retrace at most
         ``flag_max_pullback`` of the pole.
    Anchor: the last bar of the flag (the breakout point).
    """
    close = window["Close"].to_numpy(float)
    high = window["High"].to_numpy(float)
    low = window["Low"].to_numpy(float)
    n = len(close)
    gain, pole_len = params["pole_gain"], params["pole_max_len"]
    fmin, fmax = params["flag_min_len"], params["flag_max_len"]
    max_range, max_pull = params["flag_max_range"], params["flag_max_pullback"]
    order = max(2, pole_len // 3)
    pole_min = max(5, pole_len // 3)  # the pole can be shorter than pole_max_len
    if n < pole_min + fmin:
        return False, -1

    for p in range(pole_min, n - fmin):
        base = close[max(0, p - pole_len):p + 1].min()
        top = high[p]
        if (top - base) / base < gain:
            continue
        # p must be the top of the pole (a local high)
        if high[p] < high[max(0, p - order):p + 1].max():
            continue
        # Grow the flag bar by bar; stop as soon as it stops being a tight consolidation.
        # The breakout (price rising back above the pole top) naturally ends the flag.
        flag_hi, flag_lo, valid_len = -np.inf, np.inf, 0
        for k in range(1, fmax + 1):
            idx = p + k
            if idx >= n:
                break
            flag_hi = max(flag_hi, high[idx])
            flag_lo = min(flag_lo, low[idx])
            if flag_hi > top * 1.01:                       # broke above the pole -> flag over
                break
            if (flag_hi - flag_lo) / top > max_range:       # band too wide
                break
            if (top - flag_lo) / (top - base) > max_pull:   # pullback too deep
                break
            valid_len = k
        if valid_len >= fmin:
            return True, int(p + valid_len)
    return False, -1


# Registry so build_dataset can iterate over labelers by class name.
LABELERS = {
    "double_top": double_top,
    "double_bottom": double_bottom,
    "head_and_shoulders": head_and_shoulders,
    "bull_flag": bull_flag,
}
