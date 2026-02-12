"""Deterministic threshold-based baseline strategies."""

from __future__ import annotations

from dataclasses import dataclass

from strategies.base import Strategy, StrategyState


@dataclass
class BuyAndHoldStrategy(Strategy):
    name: str = "buy_and_hold"

    def decide(self, state: StrategyState) -> float:
        return 1.0


@dataclass
class DeterministicThresholdStrategy(Strategy):
    buy_drop_pct: float = 0.02
    sell_rise_pct: float = 0.02
    name: str = "deterministic_threshold"

    def decide(self, state: StrategyState) -> float:
        if state.returns <= -abs(self.buy_drop_pct):
            return 1.0
        if state.returns >= abs(self.sell_rise_pct):
            return 0.0
        return state.position


@dataclass
class VolatilityScaledThresholdStrategy(Strategy):
    k: float = 1.0
    name: str = "volatility_scaled_threshold"

    def decide(self, state: StrategyState) -> float:
        threshold = max(1e-6, state.volatility * self.k)
        if state.returns <= -threshold:
            return 1.0
        if state.returns >= threshold:
            return 0.0
        return state.position
