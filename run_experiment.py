"""End-to-end experiment runner for rule-based and RL strategy comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.metrics import compute_all_metrics
from analysis.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_return_distribution,
    plot_rl_heatmap,
    plot_rolling_sharpe,
    plot_trade_freq_vs_perf,
)
from data.download import cache_ticker, download_ticker
from engine.backtester import BacktestConfig, Backtester
from features.indicators import compute_indicators
from rl.train import run_policy, train_walk_forward_ppo
from strategies.deterministic import (
    BuyAndHoldStrategy,
    DeterministicThresholdStrategy,
    VolatilityScaledThresholdStrategy,
)
from strategies.probabilistic import (
    KellyMeanReversionStrategy,
    ProbabilisticLinearStrategy,
    ProbabilisticSigmoidStrategy,
    ZScoreMeanReversionStrategy,
)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategy + RL comparison experiment.")
    parser.add_argument("--tickers", nargs="+", default=["TTD", "AAPL", "MSFT", "NVDA"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="If raw parquet is missing, download from Massive automatically.",
    )
    return parser.parse_args()


def print_runtime_banner() -> None:
    """Show script path + library versions to help diagnose stale local code copies."""
    print("[plum-ai] run_experiment.py loaded from:", Path(__file__).resolve())
    print("[plum-ai] pandas version:", pd.__version__)


def assert_indicator_patch_present() -> None:
    """Fail fast if running a stale checkout that still uses deprecated fillna(method=...)."""
    src = Path(compute_indicators.__code__.co_filename)
    text = src.read_text(encoding="utf-8")
    if 'fillna(method="bfill")' in text or "fillna(method='bfill')" in text:
        raise RuntimeError(
            "Detected stale features/indicators.py using fillna(method='bfill'). "
            "Please update your local checkout and rerun."
        )


def ensure_ticker_data(ticker: str, start: str, end: str, auto_download: bool) -> Path:
    path = RAW_DIR / f"{ticker}.parquet"
    if path.exists():
        return path

    if not auto_download:
        raise FileNotFoundError(
            f"Data file not found: {path}. Run `python -m data.download --tickers {ticker} --start {start} --end {end}` "
            "or rerun this script with --auto-download."
        )

    print(f"Missing {path}; downloading {ticker} ({start} -> {end})...")
    df = download_ticker(ticker=ticker, start=start, end=end)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check symbol/date range/API access.")
    cache_ticker(df=df, ticker=ticker)
    return path


def load_ticker_data(ticker: str, start: str, end: str, auto_download: bool) -> pd.DataFrame:
    path = ensure_ticker_data(ticker=ticker, start=start, end=end, auto_download=auto_download)
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        df["ticker"] = ticker
    return compute_indicators(df)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def main(tickers: list[str], start: str, end: str, auto_download: bool) -> None:
    print_runtime_banner()
    assert_indicator_patch_present()

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    backtester = Backtester(BacktestConfig())
    strategies = [
        BuyAndHoldStrategy(),
        DeterministicThresholdStrategy(),
        ProbabilisticLinearStrategy(),
        ProbabilisticSigmoidStrategy(),
        ZScoreMeanReversionStrategy(),
        VolatilityScaledThresholdStrategy(),
        KellyMeanReversionStrategy(),
    ]

    summary_rows = []
    all_curves = []
    rl_heatmap_frames = []

    for ticker in tickers:
        df = load_ticker_data(ticker=ticker, start=start, end=end, auto_download=auto_download)
        train_df, val_df, test_df = time_split(df)

        for strategy in strategies:
            result = backtester.run(test_df, strategy=strategy, ticker=ticker)
            perf = result["performance"].copy()
            perf["strategy"] = strategy.name
            perf["ticker"] = ticker

            metrics = compute_all_metrics(perf)
            metrics.update({"strategy": strategy.name, "ticker": ticker})
            summary_rows.append(metrics)
            all_curves.append(perf)

        model = train_walk_forward_ppo(train_df=train_df, val_df=val_df)
        rl_perf, rl_actions = run_policy(model=model, test_df=test_df)
        if not rl_perf.empty:
            rl_perf["strategy"] = "ppo_actor_critic"
            rl_perf["ticker"] = ticker
            metrics = compute_all_metrics(rl_perf)
            metrics.update({"strategy": "ppo_actor_critic", "ticker": ticker})
            summary_rows.append(metrics)
            all_curves.append(rl_perf)

        if not rl_actions.empty:
            rl_heatmap_frames.append(rl_actions)

    summary_df = pd.DataFrame(summary_rows).sort_values(["ticker", "Sharpe"], ascending=[True, False])
    curves_df = pd.concat(all_curves, ignore_index=True)

    summary_path = RESULTS_DIR / "summary.csv"
    curves_path = RESULTS_DIR / "equity_curves.parquet"
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_parquet(curves_path, index=False)

    plot_equity_curves(curves_df, PLOTS_DIR)
    plot_drawdowns(curves_df, PLOTS_DIR)
    plot_return_distribution(curves_df, PLOTS_DIR)
    plot_rolling_sharpe(curves_df, PLOTS_DIR)
    plot_trade_freq_vs_perf(summary_df, PLOTS_DIR)
    if rl_heatmap_frames:
        plot_rl_heatmap(pd.concat(rl_heatmap_frames, ignore_index=True), PLOTS_DIR)

    print(f"Saved summary: {summary_path}")
    print(f"Saved curves: {curves_path}")
    print(f"Saved plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    args = parse_args()
    main(tickers=args.tickers, start=args.start, end=args.end, auto_download=args.auto_download)
