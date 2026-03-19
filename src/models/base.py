"""Abstract base class for all forecasting models.

Every concrete model must implement ``fit`` and ``predict``.
This contract lets the walk-forward harness treat all models uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Minimal interface every model must satisfy."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Train the model on (X, y) and return self."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for X as a 1-D numpy array."""
        ...

    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """Convenience: fit then predict."""
        return self.fit(X_train, y_train).predict(X_test)
