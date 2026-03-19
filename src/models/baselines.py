"""Naive baseline forecasters for ablation comparison.

Three models that require no hyperparameter tuning:

- ZeroPredictor:       always predicts 0.0 (pure noise hypothesis)
- PrevReturnPredictor: predicts the previous day's log return (momentum naive)
- RidgeForecaster:     sklearn Ridge regression (linear baseline)

All implement BaseForecaster and are safe for walk_forward().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.base import BaseForecaster


class ZeroPredictor(BaseForecaster):
    """Always predicts zero — tests whether any model beats random-walk."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZeroPredictor":
        self._n_features = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class PrevReturnPredictor(BaseForecaster):
    """Predicts using the previous day's log return (feature log_ret_1d).

    Falls back to 0.0 if ``log_ret_1d`` is not present in the feature matrix.
    """

    _COL = "log_ret_1d"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PrevReturnPredictor":
        self._has_col = self._COL in X.columns
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._has_col:
            return X[self._COL].to_numpy()
        return np.zeros(len(X))


class RidgeForecaster(BaseForecaster):
    """Thin wrapper around sklearn Ridge for a linear baseline.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength (default 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        self._model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values)
