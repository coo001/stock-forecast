"""Tests for src/features/external_merge.py

All tests use synthetic data — no network calls, no API keys required.
The internal ``_fetch_from_source`` function is monkey-patched so tests
verify merge logic (alignment, lag, transformation) in isolation.

Coverage:
- ExternalSeriesConfig validation
- _align_series: daily, monthly, sparse data
- _apply_feature_type: all four types
- lag correctness: lag=1, lag=2
- no-leakage guarantee
- NaN warmup rows
- column naming
- multiple series
- monthly forward-fill
- partial failure (one series fails, rest succeed)
- cache interaction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.external_merge import (
    ExternalSeriesConfig,
    _align_series,
    _apply_feature_type,
    merge_external_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _samsung_index(n: int = 60) -> pd.DatetimeIndex:
    """Samsung-like business-day index."""
    return pd.bdate_range("2020-01-02", periods=n)


def _daily_series(
    idx: pd.DatetimeIndex,
    start_val: float = 100.0,
    seed: int = 0,
) -> pd.Series:
    """Deterministic daily price-like series on the given index."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, len(idx))
    values = start_val * np.exp(np.cumsum(returns))
    return pd.Series(values, index=idx, name="test_sym")


def _monthly_series(start_val: float = 1.75) -> pd.Series:
    """Monthly series with first-of-month DatetimeIndex (e.g. central bank rate)."""
    idx = pd.date_range("2019-12-01", periods=18, freq="MS")
    values = [start_val + i * 0.05 for i in range(len(idx))]
    return pd.Series(values, index=idx, name="monthly_sym")


def _ohlcv(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    idx = _samsung_index(n)
    close = 60_000 + np.cumsum(rng.normal(0, 500, n))
    return pd.DataFrame(
        {
            "open":   close + rng.normal(0, 100, n),
            "high":   close + np.abs(rng.normal(0, 100, n)),
            "low":    close - np.abs(rng.normal(0, 100, n)),
            "close":  close,
            "volume": np.abs(rng.lognormal(16, 0.5, n)),
        },
        index=idx,
    )


def _cfg(
    name="test",
    source="market",
    symbol="SYM",
    lag_days=1,
    feature_type="level",
    frequency="daily",
) -> ExternalSeriesConfig:
    return ExternalSeriesConfig(
        name=name, source=source, symbol=symbol,
        lag_days=lag_days, feature_type=feature_type, frequency=frequency,
    )


# ── ExternalSeriesConfig validation ───────────────────────────────────────────

def test_config_rejects_lag_zero():
    with pytest.raises(ValueError, match="lag_days must be >= 1"):
        ExternalSeriesConfig(name="x", source="market", symbol="S", lag_days=0)


def test_config_rejects_negative_lag():
    with pytest.raises(ValueError, match="lag_days must be >= 1"):
        ExternalSeriesConfig(name="x", source="market", symbol="S", lag_days=-3)


def test_config_rejects_invalid_feature_type():
    with pytest.raises(ValueError, match="feature_type must be one of"):
        ExternalSeriesConfig(name="x", source="market", symbol="S", feature_type="raw")


def test_config_rejects_invalid_source():
    with pytest.raises(ValueError, match="source must be one of"):
        ExternalSeriesConfig(name="x", source="bloomberg", symbol="S")


def test_config_valid_accepts_all_feature_types():
    for ft in ("level", "diff", "pct_change", "log_return"):
        cfg = ExternalSeriesConfig(name="ok", source="market", symbol="S", feature_type=ft)
        assert cfg.feature_type == ft


# ── _align_series ─────────────────────────────────────────────────────────────

def test_align_daily_index_matches_samsung_exactly():
    idx = _samsung_index(30)
    raw = _daily_series(idx)
    aligned = _align_series(raw, idx)
    assert list(aligned.index) == list(idx)


def test_align_monthly_index_matches_samsung_exactly():
    idx = _samsung_index(60)
    monthly = _monthly_series()
    aligned = _align_series(monthly, idx)
    assert list(aligned.index) == list(idx)


def test_align_monthly_no_nan_after_first_value():
    idx = _samsung_index(60)
    monthly = _monthly_series()
    aligned = _align_series(monthly, idx)
    first_valid = aligned.first_valid_index()
    assert aligned.loc[first_valid:].isna().sum() == 0


def test_align_monthly_values_constant_within_month():
    idx = _samsung_index(60)  # ~3 months
    monthly = _monthly_series()
    aligned = _align_series(monthly, idx)
    non_null = aligned.dropna()
    # Most consecutive days should have the same value (forward-fill)
    diffs = non_null.diff().dropna()
    flat_fraction = (diffs == 0.0).sum() / len(diffs)
    assert flat_fraction > 0.7, "Monthly data should be mostly constant between announcements"


def test_align_sparse_series_forward_fills_gaps():
    idx = _samsung_index(20)
    sparse_idx = idx[::3]  # every 3rd date
    raw = pd.Series(np.arange(1.0, len(sparse_idx) + 1), index=sparse_idx)
    aligned = _align_series(raw, idx)
    assert aligned.notna().sum() == len(idx)


# ── _apply_feature_type ───────────────────────────────────────────────────────

def _s3() -> pd.Series:
    return pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))


def test_feature_type_level_unchanged():
    s = _s3()
    result = _apply_feature_type(s, "level", "t")
    pd.testing.assert_series_equal(result, s)


def test_feature_type_diff():
    result = _apply_feature_type(_s3(), "diff", "t")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(-11.0)


def test_feature_type_pct_change():
    result = _apply_feature_type(_s3(), "pct_change", "t")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(-0.10, rel=1e-4)


def test_feature_type_log_return():
    result = _apply_feature_type(_s3(), "log_return", "t")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert result.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_feature_type_invalid_raises():
    with pytest.raises(ValueError, match="Unknown feature_type"):
        _apply_feature_type(_s3(), "zscore", "t")


# ── Lag correctness ───────────────────────────────────────────────────────────

def test_lag1_level_feature_at_T_equals_raw_at_T_minus_1(monkeypatch, tmp_path):
    """ext[T] = raw[T-1] when lag=1, feature_type='level'."""
    idx = _samsung_index(30)
    raw = _daily_series(idx, start_val=100.0)
    ohlcv = _ohlcv(30)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfg = _cfg(name="spy", lag_days=1, feature_type="level")
    result = merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    col = "ext_spy"
    # Row 0 → NaN (no T-1 available); rows 1..n-1 → raw[i-1]
    assert np.isnan(result[col].iloc[0])
    for i in range(1, len(idx)):
        assert result[col].iloc[i] == pytest.approx(raw.iloc[i - 1])


def test_lag2_level_feature_at_T_equals_raw_at_T_minus_2(monkeypatch, tmp_path):
    """ext[T] = raw[T-2] when lag=2, feature_type='level'."""
    idx = _samsung_index(30)
    raw = _daily_series(idx, start_val=50.0, seed=1)
    ohlcv = _ohlcv(30)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfg = _cfg(name="spy2", lag_days=2, feature_type="level")
    result = merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    col = "ext_spy2"
    assert result[col].iloc[:2].isna().all()
    for i in range(2, len(idx)):
        assert result[col].iloc[i] == pytest.approx(raw.iloc[i - 2])


# ── No-leakage guarantee ──────────────────────────────────────────────────────

def test_no_leakage_log_return_lag1(monkeypatch, tmp_path):
    """After lag=1 + log_return, feature[T] = log(raw[T-1] / raw[T-2]).

    The value at T (raw[T]) must not contribute to the feature at T.
    """
    idx = _samsung_index(40)
    raw = _daily_series(idx)
    ohlcv = _ohlcv(40)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfg = _cfg(name="k", lag_days=1, feature_type="log_return")
    result = merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    col = "ext_k"
    for i in range(2, len(idx)):
        feat = result[col].iloc[i]
        if np.isnan(feat):
            continue
        # Feature at T = log_return at T-1 = log(raw[T-1] / raw[T-2])
        expected = np.log(raw.iloc[i - 1] / raw.iloc[i - 2])
        assert feat == pytest.approx(expected, rel=1e-6), f"Leakage at row {i}"


def test_first_lag_rows_are_nan(monkeypatch, tmp_path):
    """The first lag_days rows must always be NaN (warmup period)."""
    idx = _samsung_index(30)
    raw = _daily_series(idx)
    ohlcv = _ohlcv(30)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    for lag in (1, 2, 3):
        cfg = _cfg(name=f"lag{lag}", lag_days=lag, feature_type="level")
        result = merge_external_features(
            ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
            cache_dir=str(tmp_path),
        )
        col = f"ext_lag{lag}"
        assert result[col].iloc[:lag].isna().all(), f"Expected NaN in first {lag} rows"
        assert result[col].iloc[lag:].notna().all(), f"Expected values after row {lag}"


# ── Output column names ───────────────────────────────────────────────────────

def test_column_names_are_ext_prefixed(monkeypatch, tmp_path):
    ohlcv = _ohlcv(30)
    raw = _daily_series(_samsung_index(30))

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfgs = [
        _cfg(name="kospi",  lag_days=1, feature_type="log_return"),
        _cfg(name="usdkrw", lag_days=1, feature_type="log_return"),
        _cfg(name="vix",    lag_days=1, feature_type="level"),
    ]
    result = merge_external_features(
        ohlcv, cfgs, start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    assert "ext_kospi"  in result.columns
    assert "ext_usdkrw" in result.columns
    assert "ext_vix"    in result.columns


def test_original_ohlcv_columns_preserved(monkeypatch, tmp_path):
    ohlcv = _ohlcv(30)
    raw = _daily_series(_samsung_index(30))

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfg = _cfg(name="x", lag_days=1)
    result = merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns


def test_empty_series_list_returns_copy(tmp_path):
    ohlcv = _ohlcv(30)
    result = merge_external_features(
        ohlcv, [], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )
    pd.testing.assert_frame_equal(result, ohlcv)


def test_input_ohlcv_not_mutated(monkeypatch, tmp_path):
    ohlcv = _ohlcv(30)
    raw = _daily_series(_samsung_index(30))
    original_cols = list(ohlcv.columns)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raw)

    cfg = _cfg(name="x", lag_days=1)
    merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )
    assert list(ohlcv.columns) == original_cols


# ── Multiple series ───────────────────────────────────────────────────────────

def test_multiple_series_produce_independent_columns(monkeypatch, tmp_path):
    ohlcv = _ohlcv(40)
    idx = _samsung_index(40)
    raws = {
        "alpha": _daily_series(idx, start_val=100.0, seed=0),
        "beta":  _daily_series(idx, start_val=200.0, seed=1),
        "gamma": _daily_series(idx, start_val=50.0,  seed=2),
    }

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: raws[cfg.name])

    # Each cfg must have a unique symbol so cache keys don't collide
    cfgs = [
        ExternalSeriesConfig(name=n, source="market", symbol=f"SYM_{n}",
                             lag_days=1, feature_type="level")
        for n in raws
    ]
    result = merge_external_features(
        ohlcv, cfgs, start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    for n, raw in raws.items():
        col = f"ext_{n}"
        assert col in result.columns
        for i in range(1, len(idx)):
            assert result[col].iloc[i] == pytest.approx(raw.iloc[i - 1])


# ── Monthly forward-fill ──────────────────────────────────────────────────────

def test_monthly_data_mostly_constant_between_announcements(monkeypatch, tmp_path):
    idx = _samsung_index(60)
    monthly = _monthly_series(start_val=2.0)
    ohlcv = _ohlcv(60)

    import src.features.external_merge as em
    monkeypatch.setattr(em, "_fetch_from_source", lambda cfg, start, end: monthly)

    cfg = _cfg(name="rate", lag_days=5, feature_type="level", frequency="monthly")
    result = merge_external_features(
        ohlcv, [cfg], start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    col = "ext_rate"
    non_null = result[col].dropna()
    diffs = non_null.diff().dropna()
    flat_fraction = (diffs == 0.0).sum() / len(diffs)
    assert flat_fraction > 0.7


# ── Fail-safe behaviour ───────────────────────────────────────────────────────

def test_failed_series_does_not_abort_merge(monkeypatch, tmp_path):
    """If one series fails, the others must still be merged successfully."""
    ohlcv = _ohlcv(30)
    raw = _daily_series(_samsung_index(30))

    import src.features.external_merge as em

    def mock_fetch(cfg, start, end):
        if cfg.name == "broken":
            raise RuntimeError("simulated API failure")
        return raw

    monkeypatch.setattr(em, "_fetch_from_source", mock_fetch)

    # Use distinct symbols so cache keys don't collide between the two series
    cfgs = [
        ExternalSeriesConfig(name="ok",     source="market", symbol="SYM_OK",
                             lag_days=1, feature_type="level"),
        ExternalSeriesConfig(name="broken", source="market", symbol="SYM_BROKEN",
                             lag_days=1, feature_type="level"),
    ]
    result = merge_external_features(
        ohlcv, cfgs, start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    assert "ext_ok"     in result.columns
    assert "ext_broken" not in result.columns


def test_all_series_failing_returns_ohlcv_copy(monkeypatch, tmp_path):
    ohlcv = _ohlcv(30)

    import src.features.external_merge as em
    monkeypatch.setattr(
        em, "_fetch_from_source",
        lambda cfg, start, end: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    cfgs = [_cfg(name="a", lag_days=1), _cfg(name="b", lag_days=1)]
    result = merge_external_features(
        ohlcv, cfgs, start="2020-01-01", end="2020-04-01",
        cache_dir=str(tmp_path),
    )

    # Should have no ext_ columns
    ext_cols = [c for c in result.columns if c.startswith("ext_")]
    assert ext_cols == []
    pd.testing.assert_frame_equal(result[ohlcv.columns], ohlcv)


# ── Cache interaction ─────────────────────────────────────────────────────────

def test_cache_is_used_on_second_call(monkeypatch, tmp_path):
    """Second merge call with same config must use the cache, not the client."""
    idx = _samsung_index(30)
    raw = _daily_series(idx)
    ohlcv = _ohlcv(30)
    call_count = {"n": 0}

    import src.features.external_merge as em

    def counting_fetch(cfg, start, end):
        call_count["n"] += 1
        return raw

    monkeypatch.setattr(em, "_fetch_from_source", counting_fetch)

    cfg = _cfg(name="cached", lag_days=1, feature_type="level")
    kw = dict(start="2020-01-01", end="2020-04-01", cache_dir=str(tmp_path))
    merge_external_features(ohlcv, [cfg], **kw)
    merge_external_features(ohlcv, [cfg], **kw)

    assert call_count["n"] == 1, "Second call should have used the cache"
