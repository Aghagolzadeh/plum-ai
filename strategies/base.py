"""Strategy interfaces and shared state typing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StrategyState:
    price: float
    returns: float
    zscore: float
    volatility: float
    momentum: float
    position: float
    cash: float
    rolling_mean: float = 0.0


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, state: StrategyState) -> float:
        """Return desired target position fraction in [-leverage, leverage]."""

    def clamp(self, value: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))
