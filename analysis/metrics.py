"""Performance metric calculations for strategy comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    std = r.std(ddof=0)
    if std == 0 or len(r) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / std)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    dstd = downside.std(ddof=0)
    if dstd == 0 or len(r) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dstd)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / mdd


def turnover(position: pd.Series) -> float:
    if position.empty:
        return 0.0
    return float(position.diff().abs().fillna(0.0).mean())


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cutoff = r.quantile(alpha)
    tail = r[r <= cutoff]
    if tail.empty:
        return 0.0
    return float(tail.mean())


def compute_all_metrics(perf: pd.DataFrame) -> dict[str, float]:
    equity = perf["equity"]
    returns = perf["daily_return"]
    position = perf["position"]

    return {
        "CAGR": cagr(equity),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "MaxDrawdown": max_drawdown(equity),
        "Calmar": calmar_ratio(equity),
        "Turnover": turnover(position),
        "WinRate": win_rate(returns),
        "CVaR_5": cvar(returns, alpha=0.05),
    }
