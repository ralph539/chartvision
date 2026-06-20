"""Build an interactive Plotly candlestick chart (HTML) for the report/slides/video.

This is a *showcase* artifact, not part of the training data. It lets a viewer hover, zoom,
and pan a labelled chart window, with the pattern's key points marked. Examples are pulled
straight from the real dataset manifest, so they always match a true labelled window.

Usage:
    python -m src.data.interactive_demo --pattern bull_flag                 # a random bull_flag
    python -m src.data.interactive_demo --pattern double_bottom --ticker MSFT --date 2020-08-12
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import plotly.graph_objects as go

from src.data.rule_labelers import LABELERS
from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def build(pattern: str, ticker: str | None, date: str | None) -> None:
    cfg = load_config()
    size = cfg["window"]["size"]
    raw = resolve_path(cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["figures_dir"]) / "interactive"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = cfg["render"]

    # pick a labelled example from the real dataset manifest
    manifest = pd.read_csv(resolve_path(cfg["paths"]["processed_dir"]) / "manifest.csv")
    sub = manifest[manifest["label"] == pattern]
    if ticker:
        sub = sub[sub["ticker"] == ticker]
    if date:
        sub = sub[sub["start_date"] == date]
    if sub.empty:
        logger.warning("No %s window in the manifest for ticker=%s date=%s", pattern, ticker, date)
        return
    row = sub.sample(1, random_state=cfg["seed"]).iloc[0]
    ticker = row["ticker"]
    df = pd.read_csv(raw / f"{ticker}.csv", index_col=0, parse_dates=True)
    start = int(row["start"])
    anchor = int(row["anchor"])  # absolute bar index of the confirmation point
    win = df.iloc[start : start + size]

    fig = go.Figure(
        data=[go.Candlestick(
            x=win.index,
            open=win["Open"], high=win["High"], low=win["Low"], close=win["Close"],
            increasing_line_color=r["up_color"], decreasing_line_color=r["down_color"],
            name=ticker,
        )]
    )
    # 20-bar moving average
    ma = win["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=win.index, y=ma, line=dict(color="#FACC15", width=2), name="MA20"))
    # mark the anchor (pattern confirmation point)
    off = min(max(anchor - start, 0), size - 1)
    a_date = win.index[off]
    fig.add_trace(go.Scatter(
        x=[a_date], y=[win["High"].iloc[off]],
        mode="markers+text", text=["anchor"], textposition="top center",
        marker=dict(size=14, color="#FFFFFF", symbol="circle-open", line=dict(width=2)),
        name="anchor",
    ))
    fig.update_layout(
        title=f"{pattern}  ·  {ticker}  ·  {win.index[0].date()} to {win.index[-1].date()}",
        template="plotly_dark",
        paper_bgcolor=r["bg_top"], plot_bgcolor=r["bg_top"],
        xaxis_rangeslider_visible=True,
        font=dict(color="#E5E7EB"),
        height=600,
    )
    out = out_dir / f"interactive_{pattern}_{ticker}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    logger.info("Wrote interactive chart -> %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build an interactive Plotly chart.")
    p.add_argument("--pattern", default="bull_flag", choices=list(LABELERS))
    p.add_argument("--ticker", default=None)
    p.add_argument("--date", default=None, help="window start_date, e.g. 2020-08-12")
    args = p.parse_args()
    build(args.pattern, args.ticker, args.date)


if __name__ == "__main__":
    main()
