"""Gymnasium environment wrapping a single ticker feature dataframe."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        transaction_cost_bps: float = 5.0,
        risk_penalty: float = 0.1,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_penalty = risk_penalty
        self.max_steps = max_steps or len(self.data) - 1

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.step_idx = 0
        self.position = 0.0

    def _obs(self) -> np.ndarray:
        row = self.data.iloc[self.step_idx]
        return np.array(
            [
                float(row.get("zscore", 0.0)),
                float(row.get("momentum", 0.0)),
                float(row.get("rolling_volatility", 0.0)),
                float(self.position),
                float(row.get("returns", 0.0)),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 0.0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        target = float(np.clip(action[0], -1.0, 1.0))
        row = self.data.iloc[self.step_idx]

        transaction_penalty = abs(target - self.position) * (self.transaction_cost_bps / 10_000)
        pnl = target * float(row.get("returns", 0.0))
        risk = self.risk_penalty * (target**2) * float(row.get("rolling_volatility", 0.0))
        reward = pnl - transaction_penalty - risk

        self.position = target
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        truncated = False

        obs = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"pnl": pnl, "transaction_penalty": transaction_penalty, "risk_penalty": risk}
        return obs, float(reward), done, truncated, info
