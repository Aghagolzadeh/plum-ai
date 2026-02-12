"""PPO walk-forward training and inference utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import TradingEnv


def train_walk_forward_ppo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_dir: str = "results/checkpoints",
    total_timesteps: int = 20_000,
) -> PPO:
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
    eval_env = DummyVecEnv([lambda: TradingEnv(val_df)])

    early_stop = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=0)
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=early_stop,
        eval_freq=1000,
        best_model_save_path=str(out),
        deterministic=True,
        verbose=0,
    )

    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(str(out / "ppo_last"))
    return model


def run_policy(model: PPO, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = TradingEnv(test_df)
    obs, _ = env.reset()

    equity = 1.0
    rows = []
    actions = []

    for i in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        equity *= 1 + reward
        row = test_df.iloc[i]
        action_val = float(np.clip(action[0], -1.0, 1.0))

        rows.append(
            {
                "date": row["date"],
                "equity": equity,
                "daily_return": reward,
                "position": action_val,
            }
        )
        actions.append(
            {
                "zscore": float(row.get("zscore", 0.0)),
                "volatility": float(row.get("rolling_volatility", 0.0)),
                "action": action_val,
            }
        )
        if done:
            break

    perf = pd.DataFrame(rows)
    actions_df = pd.DataFrame(actions)
    if not actions_df.empty:
        actions_df["z_bin"] = pd.qcut(actions_df["zscore"], q=10, duplicates="drop")
        actions_df["vol_bin"] = pd.qcut(actions_df["volatility"], q=10, duplicates="drop")
    return perf, actions_df
