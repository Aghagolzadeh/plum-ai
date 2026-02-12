"""End-to-end experiment runner for rule-based and RL strategy comparison."""

from __future__ import annotations

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


def load_ticker_data(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run download module first.")
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        df["ticker"] = ticker
    return compute_indicators(df)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def main(tickers: list[str] | None = None) -> None:
    tickers = tickers or ["TTD", "AAPL", "MSFT", "NVDA"]

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
        df = load_ticker_data(ticker)
        train_df, val_df, test_df = time_split(df)

        # Classical strategies only evaluated on held-out test set to avoid leakage.
        for strategy in strategies:
            result = backtester.run(test_df, strategy=strategy, ticker=ticker)
            perf = result["performance"].copy()
            perf["strategy"] = strategy.name
            perf["ticker"] = ticker

            metrics = compute_all_metrics(perf)
            metrics.update({"strategy": strategy.name, "ticker": ticker})
            summary_rows.append(metrics)
            all_curves.append(perf)

        # RL walk-forward: train on train, validate on val, test on future split.
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
    main()
