"""LightGBM baseline forecasting model.

Predicts next-day log return.  The model is intentionally kept simple:
no hyperparameter search, no feature selection – those belong in the
research notebooks or future AutoGen experiment agents.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class LGBMForecaster(BaseForecaster):
    """Gradient-boosted tree regressor wrapping LightGBM.

    Args:
        params: LightGBM parameters dict.  If None, conservative defaults
                are used that work reasonably for financial time-series.
        early_stopping_rounds: Pass a hold-out eval set to trigger early
                stopping (disabled by default in walk-forward mode).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or _default_params()
        self._model = None

    # ── BaseForecaster interface ───────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        """Train LightGBM on (X, y).

        Args:
            X: Feature matrix (rows = observations, columns = features).
            y: Target series aligned with X.

        Returns:
            self
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm") from exc

        logger.debug("Fitting LGBM on %d rows, %d features", len(X), X.shape[1])
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted log returns for X.

        Args:
            X: Feature matrix with the same columns as the training set.

        Returns:
            1-D array of predicted values.

        Raises:
            RuntimeError: if ``fit`` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._model.predict(X)

    # ── Extras ────────────────────────────────────────────────────────────────

    @property
    def feature_importance(self) -> pd.Series | None:
        """Feature importances (gain) after fitting, or None."""
        if self._model is None:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_name_,
            name="importance",
        ).sort_values(ascending=False)

    @classmethod
    def from_config(cls, lgbm_cfg: dict[str, Any]) -> "LGBMForecaster":
        """Instantiate from the ``lgbm`` section of ``config/default.yaml``."""
        return cls(params=lgbm_cfg)


def _default_params() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }
