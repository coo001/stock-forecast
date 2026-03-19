"""Synthetic OHLCV data generator for offline development and testing.

Generates Samsung Electronics-like price data using Geometric Brownian Motion
with parameters calibrated to approximate the stock's historical characteristics:
  - Starting price : ~60,000 KRW
  - Annual drift   : ~8%
  - Annual vol     : ~25%
  - Intra-day spread: ~0.5%

This module is the single source of truth for synthetic data.
Both ``run_pipeline.py --synthetic`` and the demo scripts import from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_samsung_ohlcv(
    n: int = 2500,
    seed: int = 42,
    start: str = "2015-01-02",
    start_price: float = 60_000.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
) -> pd.DataFrame:
    """Generate synthetic Samsung Electronics-like daily OHLCV data.

    Args:
        n: Number of trading days to generate.
        seed: Random seed for reproducibility.
        start: First date in the returned index (business days).
        start_price: Opening price of the series (KRW).
        annual_drift: Expected annual return (GBM mu).
        annual_vol: Annual volatility (GBM sigma).

    Returns:
        DataFrame with DatetimeIndex (business days) and columns
        [open, high, low, close, volume], matching the format of
        ``src.data.loader.load_ohlcv``.
    """
    rng = np.random.default_rng(seed)

    daily_mu = annual_drift / 252
    daily_sigma = annual_vol / np.sqrt(252)

    log_rets = rng.normal(daily_mu, daily_sigma, n)
    close = start_price * np.exp(np.cumsum(log_rets))

    intraday_spread = close * 0.005
    high = close + np.abs(rng.normal(0, intraday_spread, n))
    low = close - np.abs(rng.normal(0, intraday_spread, n))
    open_ = close + rng.normal(0, intraday_spread * 0.5, n)

    # Lognormal volume centred on ~10 M shares/day
    volume = np.abs(rng.lognormal(mean=np.log(1e7), sigma=0.5, size=n))

    idx = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "date"
    return df


def recommended_n(initial_train_days: int, step_days: int, n_folds: int = 10) -> int:
    """Return a synthetic row count that comfortably fits the backtest window.

    Args:
        initial_train_days: Walk-forward initial window size.
        step_days: Walk-forward step size.
        n_folds: Desired minimum number of out-of-sample folds.

    Returns:
        Suggested number of synthetic rows (always at least 1500).
    """
    return max(initial_train_days + step_days * n_folds, 1500)
