"""Feature engineering pipeline.

Assembles all technical features into a single DataFrame and
attaches the forecast target.  The returned DataFrame is ready
for walk-forward splitting – no further transformations required
except optional z-score normalisation inside each training fold.

Leakage guarantee:
    Every feature at row t is computed from data at indices ≤ t.
    The target at row t is the outcome at t + horizon (future data),
    so it is only used as a label, never as a feature.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.features.indicators import (
    atr,
    log_returns,
    make_target,
    moving_average_ratio,
    rsi,
    volume_ratio,
)

logger = logging.getLogger(__name__)


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    *,
    return_windows: list[int] = (1, 5, 10, 20),
    ma_windows: list[int] = (5, 20, 60),
    rsi_window: int = 14,
    atr_window: int = 14,
    volume_ma_window: int = 20,
    target_kind: str = "next_day_log_return",
    horizon: int = 1,
) -> pd.DataFrame:
    """Build the full feature + target DataFrame.

    Args:
        ohlcv: Raw OHLCV DataFrame with columns [open, high, low, close, volume].
        return_windows: Lookback windows (days) for log-return features.
        ma_windows: Lookback windows for moving-average ratio features.
        rsi_window: RSI period.
        atr_window: ATR period.
        volume_ma_window: Period for volume ratio.
        target_kind: Forecast target type (see ``indicators.make_target``).
        horizon: Forecast horizon in trading days.

    Returns:
        DataFrame with feature columns and a ``target`` column.
        Rows with any NaN in features or target are dropped.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    parts: list[pd.Series] = []

    # Log returns
    for w in return_windows:
        parts.append(log_returns(close, w))

    # MA ratios
    for w in ma_windows:
        parts.append(moving_average_ratio(close, w))

    # RSI
    parts.append(rsi(close, rsi_window))

    # ATR %
    parts.append(atr(high, low, close, atr_window))

    # Volume ratio
    parts.append(volume_ratio(volume, volume_ma_window))

    # Assemble features
    feature_df = pd.concat(parts, axis=1)

    # Target (computed without touching feature columns)
    target_series = make_target(close, horizon=horizon, kind=target_kind)
    feature_df["target"] = target_series

    n_before = len(feature_df)
    feature_df.dropna(inplace=True)
    n_dropped = n_before - len(feature_df)
    logger.info(
        "Feature matrix: %d rows, %d feature cols (dropped %d NaN rows)",
        len(feature_df),
        len(feature_df.columns) - 1,
        n_dropped,
    )
    return feature_df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (all columns except 'target')."""
    return [c for c in df.columns if c != "target"]
