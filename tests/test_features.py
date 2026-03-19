"""Unit tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    atr,
    log_returns,
    make_target,
    moving_average_ratio,
    rsi,
    volume_ratio,
)
from src.features.pipeline import build_feature_matrix, feature_columns


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def price_series() -> pd.Series:
    """200-day synthetic close price with no NaN."""
    rng = np.random.default_rng(0)
    prices = 60_000 + np.cumsum(rng.normal(0, 500, 200))
    prices = np.abs(prices)  # ensure positive
    return pd.Series(prices, index=pd.date_range("2020-01-01", periods=200, freq="B"), name="close")


@pytest.fixture()
def ohlcv_df(price_series: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = price_series
    spread = close * 0.01
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 200, len(close)),
            "high": close + np.abs(rng.normal(0, 300, len(close))),
            "low": close - np.abs(rng.normal(0, 300, len(close))),
            "close": close,
            "volume": np.abs(rng.normal(1e7, 2e6, len(close))),
        },
        index=close.index,
    )


# ── Indicator tests ────────────────────────────────────────────────────────────

def test_log_returns_length(price_series):
    result = log_returns(price_series, 1)
    assert len(result) == len(price_series)
    assert result.isna().sum() == 1  # first row is NaN


def test_log_returns_values(price_series):
    result = log_returns(price_series, 1)
    expected = np.log(price_series.iloc[1] / price_series.iloc[0])
    assert abs(result.iloc[1] - expected) < 1e-10


def test_ma_ratio_all_nan_initially(price_series):
    result = moving_average_ratio(price_series, 20)
    assert result.iloc[:19].isna().all()


def test_rsi_bounds(price_series):
    result = rsi(price_series)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr_positive(ohlcv_df):
    result = atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
    assert (result.dropna() > 0).all()


def test_volume_ratio_name(ohlcv_df):
    result = volume_ratio(ohlcv_df["volume"], 20)
    assert result.name == "vol_ratio_20d"


# ── Target tests ───────────────────────────────────────────────────────────────

def test_target_no_lookahead(price_series):
    """Target at t+1 must be NaN for the last row (no future data)."""
    target = make_target(price_series, horizon=1)
    assert pd.isna(target.iloc[-1])


def test_target_direction_values(price_series):
    direction = make_target(price_series, horizon=1, kind="next_day_direction")
    valid = direction.dropna()
    assert set(valid.unique()).issubset({-1.0, 1.0})


def test_target_unknown_kind_raises(price_series):
    with pytest.raises(ValueError, match="Unknown target kind"):
        make_target(price_series, kind="bad_kind")


# ── Pipeline tests ─────────────────────────────────────────────────────────────

def test_build_feature_matrix_columns(ohlcv_df):
    df = build_feature_matrix(ohlcv_df)
    assert "target" in df.columns
    assert len(feature_columns(df)) > 0


def test_build_feature_matrix_no_nan(ohlcv_df):
    df = build_feature_matrix(ohlcv_df)
    assert df.isna().sum().sum() == 0


def test_build_feature_matrix_target_not_in_features(ohlcv_df):
    df = build_feature_matrix(ohlcv_df)
    assert "target" not in feature_columns(df)


def test_build_feature_matrix_index_monotonic(ohlcv_df):
    df = build_feature_matrix(ohlcv_df)
    assert df.index.is_monotonic_increasing
