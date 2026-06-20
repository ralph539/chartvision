"""Download daily OHLC data from Yahoo Finance and cache it as per-ticker CSVs.

Usage:
    python -m src.data.fetch_ohlc                 # fetch the smoke-test tickers (5)
    python -m src.data.fetch_ohlc --all           # fetch the full universe (~30)
    python -m src.data.fetch_ohlc --force         # re-download even if cached
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_one(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    raw_dir: Path,
    force: bool = False,
) -> pd.DataFrame | None:
    """Download one ticker's OHLC and cache it. Returns the DataFrame (or None on failure)."""
    out_path = raw_dir / f"{ticker}.csv"
    if out_path.exists() and not force:
        logger.info("%s already cached, skipping (use --force to re-download)", ticker)
        return pd.read_csv(out_path, index_col=0, parse_dates=True)

    logger.info("Downloading %s (%s to %s)...", ticker, start, end)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        logger.warning("No data returned for %s", ticker)
        return None

    # yfinance can return a MultiIndex column header for a single ticker; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[[c for c in OHLC_COLUMNS if c in df.columns]].dropna()
    df.index.name = "Date"
    df.to_csv(out_path)
    logger.info("Saved %s rows for %s -> %s", len(df), ticker, out_path.name)
    return df


def sanity_check(ticker: str, df: pd.DataFrame) -> None:
    """Log a quick sanity summary so we can eyeball that the data is reasonable."""
    n = len(df)
    first, last = df.index.min().date(), df.index.max().date()
    has_nan = bool(df[OHLC_COLUMNS[:4]].isna().any().any())
    bad_hl = int((df["High"] < df["Low"]).sum())
    logger.info(
        "  [%s] rows=%d  range=%s..%s  any_nan=%s  high<low_rows=%d",
        ticker, n, first, last, has_nan, bad_hl,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLC data from Yahoo Finance.")
    parser.add_argument("--all", action="store_true", help="fetch the full ticker universe")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = cfg["data"]["tickers"] if args.all else cfg["data"]["smoke_test_tickers"]
    logger.info("Fetching %d tickers into %s", len(tickers), raw_dir)

    ok, failed = [], []
    for ticker in tickers:
        df = fetch_one(
            ticker,
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
            cfg["data"]["interval"],
            raw_dir,
            force=args.force,
        )
        if df is None or df.empty:
            failed.append(ticker)
        else:
            ok.append(ticker)
            sanity_check(ticker, df)

    logger.info("Done. %d ok, %d failed.", len(ok), len(failed))
    if failed:
        logger.warning("Failed tickers: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
