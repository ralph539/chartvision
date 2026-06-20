"""Build the full image dataset: scan -> label -> balance -> temporal split -> render.

For every ticker we slide a 60-bar window. A window is:
  - a POSITIVE example if exactly ONE labeler fires (multiple firing = ambiguous -> skipped),
  - a NO_PATTERN candidate if NO labeler fires.
Overlapping detections of the same pattern are de-duplicated. Each kept window is rendered to
data/processed/<class>/<ticker>_<startdate>.png and recorded in a manifest CSV.

Usage:
    python -m src.data.build_dataset                       # full universe, default caps
    python -m src.data.build_dataset --max-per-class 500   # smaller (faster) build
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.render_charts import render_window
from src.data.rule_labelers import LABELERS, dedup_hits
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def assign_split(date: pd.Timestamp, cfg: dict) -> str:
    """Temporal split by the window's last date."""
    if date <= pd.Timestamp(cfg["splits"]["train_end"]):
        return "train"
    if date <= pd.Timestamp(cfg["splits"]["val_end"]):
        return "val"
    return "test"


def label_window(window: pd.DataFrame, cfg: dict) -> tuple[str | None, int]:
    """Return (class_name, anchor) for a window, or (None, -1) if ambiguous.

    Exactly one labeler firing -> that class. None firing -> no_pattern. More than one -> skip.
    """
    fired = []
    for name, fn in LABELERS.items():
        ok, anchor = fn(window, cfg["labelers"][name])
        if ok:
            fired.append((name, anchor))
    if len(fired) == 1:
        return fired[0]
    if len(fired) == 0:
        return "no_pattern", -1
    return None, -1  # ambiguous: more than one pattern present


def scan_ticker(ticker: str, df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Collect candidate windows for one ticker (positives deduped per class, plus negatives)."""
    size, stride = cfg["window"]["size"], cfg["window"]["stride"]
    per_class_hits: dict[str, list[tuple[int, int]]] = {c: [] for c in LABELERS}
    negatives: list[int] = []

    for start in range(0, len(df) - size, stride):
        window = df.iloc[start : start + size]
        cls, anchor = label_window(window, cfg)
        if cls is None:
            continue
        if cls == "no_pattern":
            negatives.append(start)
        else:
            per_class_hits[cls].append((start, start + anchor))

    rows: list[dict] = []
    min_gap = size // 2
    for cls, hits in per_class_hits.items():
        for start, anchor in dedup_hits(hits, min_gap=min_gap):
            end_date = df.index[start + size - 1]
            rows.append({
                "ticker": ticker, "start": start, "start_date": df.index[start].date(),
                "end_date": end_date.date(), "anchor": anchor, "label": cls,
                "split": assign_split(end_date, cfg),
            })
    # negatives: dedup the same way, then we subsample later during balancing
    for start in dedup_hits([(s, s) for s in negatives], min_gap=min_gap):
        s = start[0]
        end_date = df.index[s + size - 1]
        rows.append({
            "ticker": ticker, "start": s, "start_date": df.index[s].date(),
            "end_date": end_date.date(), "anchor": -1, "label": "no_pattern",
            "split": assign_split(end_date, cfg),
        })
    return rows


def balance(manifest: pd.DataFrame, max_per_class: int, neg_cap: int, seed: int) -> pd.DataFrame:
    """Cap each class (sampling within each split so all splits keep every class)."""
    rng = np.random.default_rng(seed)
    kept = []
    for cls in manifest["label"].unique():
        cap = neg_cap if cls == "no_pattern" else max_per_class
        sub = manifest[manifest["label"] == cls]
        # distribute the cap across splits proportionally to availability
        for split in ["train", "val", "test"]:
            s = sub[sub["split"] == split]
            frac = {"train": 0.75, "val": 0.10, "test": 0.15}[split]
            k = min(len(s), int(round(cap * frac)))
            if k > 0:
                kept.append(s.iloc[rng.choice(len(s), size=k, replace=False)])
    return pd.concat(kept).sort_values(["label", "split", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ChartVision image dataset.")
    parser.add_argument("--max-per-class", type=int, default=700,
                        help="cap per positive class (total across splits)")
    parser.add_argument("--neg-cap", type=int, default=1400, help="cap for no_pattern")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    proc_dir = resolve_path(cfg["paths"]["processed_dir"])

    # 1) scan every ticker for candidate windows
    all_rows: list[dict] = []
    for ticker in cfg["data"]["tickers"]:
        csv = raw_dir / f"{ticker}.csv"
        if not csv.exists():
            logger.warning("%s not cached, skipping", ticker)
            continue
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        rows = scan_ticker(ticker, df, cfg)
        all_rows.extend(rows)
        logger.info("%s: %d candidate windows", ticker, len(rows))

    manifest = pd.DataFrame(all_rows)
    logger.info("Raw candidates: %d", len(manifest))
    logger.info("Raw class counts:\n%s", manifest["label"].value_counts().to_string())

    # 2) balance
    manifest = balance(manifest, args.max_per_class, args.neg_cap, cfg["seed"])
    logger.info("After balancing: %d", len(manifest))

    # 3) render every kept window
    proc_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(manifest.itertuples(), 1):
        df = pd.read_csv(raw_dir / f"{row.ticker}.csv", index_col=0, parse_dates=True)
        window = df.iloc[row.start : row.start + cfg["window"]["size"]]
        out = proc_dir / row.label / f"{row.ticker}_{row.start_date}.png"
        render_window(window, out, cfg)
        paths.append(str(out.relative_to(resolve_path("."))))
        if i % 200 == 0:
            logger.info("  rendered %d/%d", i, len(manifest))
    manifest["path"] = paths

    # 4) save manifest + summary
    manifest_path = proc_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = manifest.pivot_table(index="label", columns="split", aggfunc="size", fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(resolve_path(cfg["paths"]["reports_dir"]) / "dataset_summary.csv")
    logger.info("Dataset built: %d images -> %s", len(manifest), proc_dir)
    logger.info("Summary (rows=class, cols=split):\n%s", summary.to_string())


if __name__ == "__main__":
    main()
