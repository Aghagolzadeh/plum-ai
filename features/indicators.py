"""Feature engineering helpers for backtesting state."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(
    df: pd.DataFrame,
    return_window: int = 20,
    vol_window: int = 20,
    momentum_window: int = 10,
    atr_window: int = 14,
    liquidity_window: int = 20,
) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("date").reset_index(drop=True)

    data["returns"] = data["close"].pct_change().fillna(0.0)
    data["rolling_mean"] = data["close"].rolling(return_window).mean()
    data["rolling_volatility"] = data["returns"].rolling(vol_window).std(ddof=0)

    rolling_std_price = data["close"].rolling(return_window).std(ddof=0)
    data["zscore"] = (data["close"] - data["rolling_mean"]) / rolling_std_price.replace(0, np.nan)
    data["zscore"] = data["zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cum_max = data["close"].cummax().replace(0, np.nan)
    data["drawdown"] = (data["close"] - cum_max) / cum_max
    data["drawdown"] = data["drawdown"].fillna(0.0)

    data["momentum"] = data["close"].pct_change(momentum_window).fillna(0.0)

    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(atr_window).mean().fillna(0.0)

    dollar_volume = data["close"] * data["volume"]
    data["liquidity_proxy"] = (
        dollar_volume / dollar_volume.rolling(liquidity_window).mean().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data["rolling_volatility"] = data["rolling_volatility"].fillna(0.0)
    data["rolling_mean"] = data["rolling_mean"].bfill().fillna(data["close"])

    return data
