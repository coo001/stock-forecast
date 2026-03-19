"""Pure-function technical indicator calculations.

All functions take a DataFrame or Series and return a new Series/DataFrame
without mutating the input.  Every function shifts features by ``horizon``
days (default 1) to prevent look-ahead leakage: features at row t are
derived only from data available at t, and the target is close[t+horizon].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series, window: int = 1) -> pd.Series:
    """Compute ``log(close[t] / close[t - window])``."""
    return np.log(close / close.shift(window)).rename(f"log_ret_{window}d")


def moving_average_ratio(close: pd.Series, window: int) -> pd.Series:
    """Price relative to its rolling mean: ``close / MA(window) - 1``."""
    ma = close.rolling(window, min_periods=window).mean()
    return (close / ma - 1).rename(f"ma_ratio_{window}d")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI (0–100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range normalised by close (ATR%)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_val = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return (atr_val / close).rename(f"atr_pct_{window}")


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its rolling mean: ``volume / MA(window) - 1``."""
    ma = volume.rolling(window, min_periods=window).mean()
    return (volume / ma - 1).rename(f"vol_ratio_{window}d")


def make_target(close: pd.Series, horizon: int = 1, kind: str = "next_day_log_return") -> pd.Series:
    """Construct the forecast target, aligned to the feature timestamp t.

    The returned Series at index t represents the outcome at t+horizon,
    so the last ``horizon`` rows will be NaN (no future data available).

    Args:
        close: Closing price series.
        horizon: Number of trading days ahead to forecast.
        kind: One of:
            - ``next_day_log_return``  (continuous, for regression)
            - ``next_day_direction``   (±1, for classification)
            - ``next_5d_log_return``   (continuous, 5-day horizon)

    Returns:
        pd.Series with the same index as ``close``.

    Raises:
        ValueError: if ``kind`` is not recognised.
    """
    if kind in ("next_day_log_return", "next_5d_log_return"):
        h = horizon if kind == "next_day_log_return" else 5
        ret = np.log(close.shift(-h) / close)
        return ret.rename(f"target_{kind}")
    elif kind == "next_day_direction":
        ret = np.log(close.shift(-horizon) / close)
        direction = np.sign(ret).astype(float)
        direction[direction == 0] = 1.0  # rare tie → long bias
        return direction.rename("target_next_day_direction")
    else:
        raise ValueError(
            f"Unknown target kind '{kind}'. "
            "Choose from: next_day_log_return, next_day_direction, next_5d_log_return"
        )
