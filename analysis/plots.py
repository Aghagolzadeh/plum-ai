"""Plot generation utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_equity_curves(curves: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, g in curves.groupby("strategy"):
        ax.plot(g["date"], g["equity"], label=strategy, alpha=0.9)
    ax.set_title("Equity Curves")
    ax.legend(loc="best", fontsize=8)
    out = out_dir / "equity_curves_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_drawdowns(curves: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    dd_df = curves.copy()
    dd_df["drawdown"] = dd_df.groupby("strategy")["equity"].transform(lambda x: x / x.cummax() - 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, g in dd_df.groupby("strategy"):
        ax.plot(g["date"], g["drawdown"], label=strategy)
    ax.set_title("Drawdowns")
    ax.legend(loc="best", fontsize=8)
    out = out_dir / "drawdowns.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_return_distribution(curves: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=curves, x="daily_return", hue="strategy", bins=60, stat="density", ax=ax)
    ax.set_title("Distribution of Daily Returns")
    out = out_dir / "returns_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rolling_sharpe(curves: pd.DataFrame, out_dir: Path, window: int = 63) -> Path:
    _ensure_dir(out_dir)
    rs = curves.copy()
    rs["rolling_sharpe"] = rs.groupby("strategy")["daily_return"].transform(
        lambda x: (x.rolling(window).mean() / x.rolling(window).std(ddof=0)) * (252**0.5)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, g in rs.groupby("strategy"):
        ax.plot(g["date"], g["rolling_sharpe"], label=strategy)
    ax.set_title("Rolling Sharpe")
    ax.legend(loc="best", fontsize=8)
    out = out_dir / "rolling_sharpe.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_trade_freq_vs_perf(summary: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=summary, x="Turnover", y="Sharpe", hue="strategy", ax=ax)
    ax.set_title("Trade Frequency vs Performance")
    out = out_dir / "trade_freq_vs_performance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rl_heatmap(actions_df: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    pivot = actions_df.pivot_table(index="vol_bin", columns="z_bin", values="action", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("RL Policy Action Heatmap vs Z-score")
    out = out_dir / "rl_action_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
