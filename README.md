# plum-ai backtesting framework

A modular Python 3.11 research framework to compare deterministic rules, stochastic policies, and RL (PPO actor-critic) trading policies across multiple tickers and non-overlapping time splits.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

```bash
export MASSIVE_API_KEY=XXXX
python -m data.download --tickers TTD AAPL MSFT NVDA --start 2020-01-01 --end 2026-01-01
```

## Run experiment

```bash
python run_experiment.py
```

Outputs:

- `results/summary.csv`
- `results/equity_curves.parquet`
- plots in `results/plots/`

## Scientific safeguards

- no lookahead in strategy decision loop (decisions use current row state only)
- multi-ticker support for survivorship-robust evaluation sets
- walk-forward train/validation/test for RL and test-only evaluation for rule-based methods
- explicit train/test split to prevent leakage
