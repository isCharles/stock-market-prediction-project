from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from data.transforms import get_relative_change


@dataclass(frozen=True)
class DatasetSpec:
    tickers: Sequence[str]
    start: str
    end: str


def _today_yyyy_mm_dd() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def cache_dir(root: Path) -> Path:
    d = root / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_csv_path(root: Path, ticker: str, start: str, end: str) -> Path:
    safe_end = end.replace(":", "-")
    return cache_dir(root) / f"{ticker}_{start}_{safe_end}.csv"


def fetch_yahoo_daily(
    ticker: str,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    end = end or _today_yyyy_mm_dd()
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned from Yahoo Finance for ticker={ticker}")

    # yfinance>=1.0 may return a MultiIndex column layout, e.g. (PriceField, Ticker).
    # Normalize to single-level columns: Open/High/Low/Close/Adj Close/Volume (when available).
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Common case: first level is the price field, second level is ticker.
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def load_or_fetch_daily(
    root: Path,
    ticker: str,
    start: str,
    end: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    end = end or _today_yyyy_mm_dd()
    path = cached_csv_path(root, ticker, start, end)
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df

    df = fetch_yahoo_daily(ticker=ticker, start=start, end=end)
    df.to_csv(path)
    return df


def add_return_column(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    if price_col not in df.columns:
        # Newer yfinance versions may not provide Adj Close by default; fallback to Close.
        if price_col == "Adj Close" and "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(f"Missing column {price_col} in dataframe. Have: {list(df.columns)}")
    out = df.copy()
    out["Return"] = get_relative_change(out[price_col]).astype(float)
    return out


def add_calendar_features(
    df: pd.DataFrame,
    *,
    add_dow_onehot: bool = True,
    add_month_onehot: bool = True,
) -> pd.DataFrame:
    """
    Add lightweight calendar features for Model C:
      - day_of_week one-hot (Mon..Fri) as dow_0..dow_4 based on pandas weekday (0=Mon)
      - month one-hot (1..12) as month_1..month_12
    """
    out = df.copy()
    if out.index.name != "Date":
        # still works, but keep semantics consistent
        out.index.name = "Date"

    if add_dow_onehot:
        dow = out.index.dayofweek  # 0..6
        # Map weekend to -1; but Yahoo daily usually has only weekdays. Keep robust:
        dow_clipped = np.where(dow <= 4, dow, -1)
        for k in range(5):
            out[f"dow_{k}"] = (dow_clipped == k).astype(float)

    if add_month_onehot:
        month = out.index.month  # 1..12
        for m in range(1, 13):
            out[f"month_{m}"] = (month == m).astype(float)

    return out


def dataset_md(
    spec: DatasetSpec,
    per_ticker_paths: Dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# Dataset")
    lines.append("")
    lines.append("## Source")
    lines.append("- Source: Yahoo Finance (via `yfinance`)")
    lines.append("")
    lines.append("## Time range")
    lines.append(f"- start: {spec.start}")
    lines.append(f"- end: {spec.end}")
    lines.append("")
    lines.append("## Fields")
    lines.append("- open/high/low/close/adj close/volume")
    lines.append("- Return: daily return computed as `pct_change(Adj Close)` with first value filled as 0.0")
    lines.append("")
    lines.append("## Files")
    for t, p in per_ticker_paths.items():
        lines.append(f"- {t}: `{p.as_posix()}`")
    lines.append("")
    lines.append("## Missing values / preprocessing")
    lines.append("- Missing values: none expected from Yahoo daily; if present, rows with NaNs are dropped.")
    lines.append("- Standardisation: per-fold, using train split mean/std of Return only (calendar one-hots are left as 0/1).")
    lines.append("- Splitting: expanding-window time-series CV (see `run_all.py` outputs for exact params).")
    lines.append("")
    return "\n".join(lines)


