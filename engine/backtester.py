"""Vectorized-style event loop backtester with cash/share accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyState


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_leverage: float = 1.5


class Backtester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame, strategy: Strategy, ticker: str = "UNKNOWN") -> dict[str, Any]:
        data = df.sort_values("date").reset_index(drop=True).copy()

        cash = self.config.initial_cash
        shares = 0.0

        dates: list[pd.Timestamp] = []
        equity_curve: list[float] = []
        positions: list[float] = []
        daily_returns: list[float] = []
        trade_log: list[dict[str, Any]] = []

        prev_equity = self.config.initial_cash

        for i, row in data.iterrows():
            date = pd.to_datetime(row["date"])
            close = float(row["close"])
            portfolio_value = cash + shares * close
            current_position_fraction = 0.0 if portfolio_value == 0 else (shares * close) / portfolio_value

            state = StrategyState(
                price=close,
                returns=float(row.get("returns", 0.0)),
                zscore=float(row.get("zscore", 0.0)),
                volatility=float(row.get("rolling_volatility", 0.0)),
                momentum=float(row.get("momentum", 0.0)),
                position=current_position_fraction,
                cash=cash,
                rolling_mean=float(row.get("rolling_mean", close)),
            )

            target_frac = float(strategy.decide(state))
            target_frac = max(-self.config.max_leverage, min(self.config.max_leverage, target_frac))

            # No lookahead: trade at today's close with costs applied to rebalancing delta.
            target_notional = target_frac * portfolio_value
            current_notional = shares * close
            delta_notional = target_notional - current_notional

            if abs(delta_notional) > 1e-12:
                trade_shares = delta_notional / close
                slippage = abs(delta_notional) * (self.config.slippage_bps / 10_000)
                tx_cost = abs(delta_notional) * (self.config.transaction_cost_bps / 10_000)
                total_cost = slippage + tx_cost

                cash -= delta_notional
                cash -= total_cost
                shares += trade_shares

                trade_log.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "strategy": strategy.name,
                        "price": close,
                        "trade_shares": trade_shares,
                        "trade_notional": delta_notional,
                        "tx_cost": tx_cost,
                        "slippage": slippage,
                        "post_cash": cash,
                        "post_shares": shares,
                    }
                )

            equity = cash + shares * close
            ret = 0.0 if i == 0 else (equity / prev_equity) - 1.0
            prev_equity = equity

            dates.append(date)
            equity_curve.append(equity)
            positions.append(0.0 if equity == 0 else (shares * close) / equity)
            daily_returns.append(ret)

        perf = pd.DataFrame(
            {
                "date": dates,
                "equity": equity_curve,
                "daily_return": daily_returns,
                "position": positions,
            }
        )

        return {
            "equity_curve": perf[["date", "equity"]],
            "daily_returns": perf[["date", "daily_return"]],
            "positions": perf[["date", "position"]],
            "trade_log": pd.DataFrame(trade_log),
            "performance": perf,
            "final_equity": equity_curve[-1] if equity_curve else self.config.initial_cash,
        }
