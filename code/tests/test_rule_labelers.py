"""Unit tests for the rule-based labelers, using synthetic OHLC where we know the answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.rule_labelers import (
    bull_flag,
    dedup_hits,
    double_bottom,
    double_top,
    head_and_shoulders,
)

# Thresholds mirror configs/default.yaml so the tests are self-contained.
DT = {"peak_tolerance": 0.015, "min_trough_drop": 0.07, "min_peak_separation": 12,
      "top_margin": 0.01, "confirm_drop": 0.04}
DB = {"trough_tolerance": 0.015, "min_peak_rise": 0.07, "min_trough_separation": 12,
      "bottom_margin": 0.01, "confirm_rise": 0.04}
HS = {"shoulder_tolerance": 0.03, "head_prominence": 0.03, "min_separation": 7, "top_margin": 0.02}
BF = {"pole_gain": 0.10, "pole_max_len": 15, "flag_min_len": 8, "flag_max_len": 25,
      "flag_max_range": 0.07, "flag_max_pullback": 0.5}


def _ohlc(close: np.ndarray, wiggle: float = 0.0, volume: float = 1e6) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = close * (1.0 + wiggle)
    low = close * (1.0 - wiggle)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": np.full(len(close), volume)})


def _seg(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, n)


# ----------------------------- double_top -----------------------------------------

def test_double_top_clean():
    close = np.concatenate([
        _seg(100, 120, 15), _seg(120, 106, 10)[1:], _seg(106, 120, 12)[1:], _seg(120, 104, 12)[1:],
    ])
    ok, anchor = double_top(_ohlc(close), DT)
    assert ok and anchor > 0


def test_double_top_no_confirmation_fails():
    """Two equal tops but price stays high at the end (no reversal) -> rejected."""
    close = np.concatenate([
        _seg(100, 120, 15), _seg(120, 110, 10)[1:], _seg(110, 120, 12)[1:], _seg(120, 119, 12)[1:],
    ])
    ok, _ = double_top(_ohlc(close), DT)
    assert ok is False


def test_double_top_uptrend_bump_fails():
    """Two equal bumps mid-uptrend, but the window keeps rising to a new high -> not a top."""
    close = np.concatenate([
        _seg(100, 115, 12), _seg(115, 110, 8)[1:], _seg(110, 115, 8)[1:], _seg(115, 140, 20)[1:],
    ])
    ok, _ = double_top(_ohlc(close), DT)
    assert ok is False


def test_double_top_monotonic_uptrend_fails():
    ok, anchor = double_top(_ohlc(_seg(100, 160, 60)), DT)
    assert ok is False and anchor == -1


def test_double_top_trough_too_shallow_fails():
    close = np.concatenate([
        _seg(100, 120, 15), _seg(120, 117.6, 10)[1:], _seg(117.6, 120, 12)[1:], _seg(120, 110, 12)[1:],
    ])
    ok, _ = double_top(_ohlc(close), DT)
    assert ok is False


# ----------------------------- double_bottom --------------------------------------

def test_double_bottom_clean():
    close = np.concatenate([
        _seg(120, 100, 15), _seg(100, 113, 10)[1:], _seg(113, 100, 12)[1:], _seg(100, 116, 12)[1:],
    ])
    ok, anchor = double_bottom(_ohlc(close), DB)
    assert ok and anchor > 0


def test_double_bottom_downtrend_fails():
    ok, anchor = double_bottom(_ohlc(_seg(160, 100, 60)), DB)
    assert ok is False and anchor == -1


# ----------------------------- head_and_shoulders ---------------------------------

def test_head_and_shoulders_clean():
    close = np.concatenate([
        _seg(100, 112, 10),     # left shoulder up
        _seg(112, 104, 6)[1:],  # dip
        _seg(104, 124, 9)[1:],  # head up (highest)
        _seg(124, 104, 9)[1:],  # dip
        _seg(104, 112, 8)[1:],  # right shoulder (~= left)
        _seg(112, 98, 10)[1:],  # break down
    ])
    ok, anchor = head_and_shoulders(_ohlc(close), HS)
    assert ok and anchor > 0


def test_head_and_shoulders_double_top_fails():
    """Two equal peaks (no taller head in the middle) is not H&S."""
    close = np.concatenate([
        _seg(100, 120, 15), _seg(120, 108, 10)[1:], _seg(108, 120, 12)[1:], _seg(120, 105, 12)[1:],
    ])
    ok, _ = head_and_shoulders(_ohlc(close), HS)
    assert ok is False


# ----------------------------- bull_flag ------------------------------------------

def test_bull_flag_clean():
    close = np.concatenate([
        _seg(100, 118, 12),       # pole: +18%
        _seg(118, 113, 10)[1:],   # flag: gentle pullback
        _seg(113, 116, 8)[1:],    # flag drifts sideways
        _seg(116, 130, 12)[1:],   # breakout
    ])
    ok, anchor = bull_flag(_ohlc(close), BF)
    assert ok and anchor > 0


def test_bull_flag_no_pole_fails():
    """A slow drift with no strong pole is not a bull flag."""
    ok, _ = bull_flag(_ohlc(_seg(100, 104, 60)), BF)
    assert ok is False


def test_bull_flag_deep_pullback_fails():
    """Strong pole but the 'flag' gives back almost all the gain -> not a flag."""
    close = np.concatenate([
        _seg(100, 118, 12), _seg(118, 101, 12)[1:], _seg(101, 104, 20)[1:],
    ])
    ok, _ = bull_flag(_ohlc(close), BF)
    assert ok is False


# ----------------------------- dedup ----------------------------------------------

def test_dedup_hits():
    hits = [(0, 5), (3, 8), (6, 11), (40, 45), (42, 47)]
    kept = dedup_hits(hits, min_gap=30)
    assert kept == [(0, 5), (40, 45)]


def test_flat_line_all_negative():
    flat = _ohlc(np.full(60, 100.0))
    assert double_top(flat, DT)[0] is False
    assert double_bottom(flat, DB)[0] is False
    assert head_and_shoulders(flat, HS)[0] is False
    assert bull_flag(flat, BF)[0] is False
