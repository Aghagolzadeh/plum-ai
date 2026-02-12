"""Download and cache daily OHLCV data from Massive (Polygon) API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

try:
    import massive
except ImportError as exc:  # pragma: no cover
    massive = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

RAW_DIR = Path("data/raw")


def _get_client() -> "massive.RESTClient":
    if massive is None:
        raise ImportError(
            "massive package is required for data downloads. Install dependencies first."
        ) from _IMPORT_ERROR

    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise EnvironmentError("MASSIVE_API_KEY is not set in the environment.")
    return massive.RESTClient(api_key=api_key)


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker's daily bars from Massive and return normalized DataFrame."""
    client = _get_client()
    aggs = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start,
        to=end,
        limit=50000,
    ):
        aggs.append(agg)

    if not aggs:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    records = []
    for a in aggs:
        ts = getattr(a, "timestamp", getattr(a, "t", None))
        if ts is None:
            continue
        records.append(
            {
                "date": pd.to_datetime(ts, unit="ms").normalize(),
                "open": float(getattr(a, "open", getattr(a, "o", 0.0))),
                "high": float(getattr(a, "high", getattr(a, "h", 0.0))),
                "low": float(getattr(a, "low", getattr(a, "l", 0.0))),
                "close": float(getattr(a, "close", getattr(a, "c", 0.0))),
                "volume": float(getattr(a, "volume", getattr(a, "v", 0.0))),
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def cache_ticker(df: pd.DataFrame, ticker: str) -> Path:
    """Persist ticker DataFrame to parquet in data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def download_and_cache(tickers: Iterable[str], start: str, end: str) -> None:
    for ticker in tqdm(list(tickers), desc="Downloading"):
        df = download_ticker(ticker=ticker, start=start, end=end)
        path = cache_ticker(df=df, ticker=ticker)
        print(f"Saved {ticker} -> {path} ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV data from Massive.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-01", help="End date YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_cache(tickers=args.tickers, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
