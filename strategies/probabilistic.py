"""Probabilistic and normalized mean-reversion strategies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategies.base import Strategy, StrategyState


@dataclass
class ProbabilisticLinearStrategy(Strategy):
    threshold: float = 0.03
    seed: int = 42
    name: str = "prob_linear"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def decide(self, state: StrategyState) -> float:
        drop_pct = max(0.0, -state.returns)
        p_buy = float(np.clip(drop_pct / max(self.threshold, 1e-9), 0.0, 1.0))
        return 1.0 if self.rng.random() < p_buy else 0.0


@dataclass
class ProbabilisticSigmoidStrategy(Strategy):
    alpha: float = -1.5
    seed: int = 42
    name: str = "prob_sigmoid"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def decide(self, state: StrategyState) -> float:
        score = state.zscore * self.alpha
        p_buy = 1.0 / (1.0 + np.exp(-score))
        return 1.0 if self.rng.random() < p_buy else 0.0


@dataclass
class ZScoreMeanReversionStrategy(Strategy):
    entry_z: float = 1.0
    exit_z: float = 0.2
    name: str = "zscore_mean_reversion"

    def decide(self, state: StrategyState) -> float:
        if state.zscore <= -self.entry_z:
            return 1.0
        if state.zscore >= self.entry_z:
            return -1.0
        if abs(state.zscore) <= self.exit_z:
            return 0.0
        return state.position


@dataclass
class KellyMeanReversionStrategy(Strategy):
    lookback_mean: float = 0.001
    lookback_var: float = 0.0004
    max_leverage: float = 1.5
    name: str = "kelly_mean_reversion"

    def decide(self, state: StrategyState) -> float:
        edge = -state.zscore * self.lookback_mean
        var = max(self.lookback_var, state.volatility**2, 1e-8)
        f_star = edge / var
        return self.clamp(f_star, -self.max_leverage, self.max_leverage)
