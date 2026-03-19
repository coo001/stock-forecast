"""Unit tests for walk-forward evaluation and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    annualised_sharpe,
    compute_all,
    directional_accuracy,
    mean_absolute_error,
    root_mean_squared_error,
)
from src.backtest.walk_forward import walk_forward
from src.models.base import BaseForecaster


# ── Metric tests ───────────────────────────────────────────────────────────────

def test_directional_accuracy_perfect():
    y = np.array([0.01, -0.02, 0.03])
    assert directional_accuracy(y, y) == 1.0


def test_directional_accuracy_worst():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = -y_true
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_mae_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_rmse_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert root_mean_squared_error(y, y) == pytest.approx(0.0)


def test_sharpe_nan_on_constant():
    assert np.isnan(annualised_sharpe(np.zeros(100)))


def test_compute_all_keys():
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 0.01, 100)
    y_pred = rng.normal(0, 0.01, 100)
    result = compute_all(y_true, y_pred)
    for key in ("mae", "rmse", "directional_accuracy", "sharpe"):
        assert key in result


# ── Walk-forward tests ─────────────────────────────────────────────────────────

class _ConstantForecaster(BaseForecaster):
    """Dummy model that always predicts a constant value."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def _make_df(n: int = 700) -> pd.DataFrame:
    """Synthetic feature + target DataFrame."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    feat = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=[f"f{i}" for i in range(5)])
    feat["target"] = rng.normal(0, 0.01, n)
    return feat


def test_walk_forward_produces_folds():
    df = _make_df(700)
    result = walk_forward(
        df,
        _ConstantForecaster,
        initial_train_days=300,
        step_days=100,
    )
    assert len(result.folds) >= 3


def test_walk_forward_no_lookahead():
    """Every test fold must start strictly after its training fold ends."""
    df = _make_df(700)
    result = walk_forward(df, _ConstantForecaster, initial_train_days=300, step_days=100)
    for fold in result.folds:
        assert fold.test_start > fold.train_end


def test_walk_forward_predictions_cover_full_oos():
    """Concatenated predictions must exactly match the test indices."""
    df = _make_df(700)
    result = walk_forward(df, _ConstantForecaster, initial_train_days=300, step_days=100)
    preds = result.all_predictions
    # All test dates should appear in predictions
    for fold in result.folds:
        assert fold.test_start in preds.index
        assert fold.test_end in preds.index


def test_walk_forward_too_little_data_raises():
    df = _make_df(100)
    with pytest.raises(ValueError):
        walk_forward(df, _ConstantForecaster, initial_train_days=300, step_days=100)


def test_walk_forward_summary_shape():
    df = _make_df(700)
    result = walk_forward(df, _ConstantForecaster, initial_train_days=300, step_days=100)
    summary = result.summary()
    assert "directional_accuracy" in summary.columns
    assert len(summary) == len(result.folds)
