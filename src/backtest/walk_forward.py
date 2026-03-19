"""Walk-forward validation for time-series forecasting.

Walk-forward (also called anchored/expanding-window) backtesting is the
only valid validation strategy for financial time-series: we train on all
data up to a cutoff t, predict the next step period, then expand the
training window by ``step_days`` and repeat.

                  │←── train ──────────────────────────────→│← test →│
    fold 0:       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░
    fold 1:       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░
    fold 2:       ...

Leakage guarantee:
    The test rows at fold k are strictly after all training rows at fold k.
    The model is re-fitted from scratch at each fold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_all
from src.features.pipeline import feature_columns
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: dict[str, float]
    predictions: pd.Series  # index = test dates, values = predicted returns


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""

    folds: list[FoldResult] = field(default_factory=list)

    @property
    def all_predictions(self) -> pd.Series:
        """Concatenated out-of-sample predictions across all folds."""
        return pd.concat([f.predictions for f in self.folds]).sort_index()

    def summary(self) -> pd.DataFrame:
        """Per-fold metric table."""
        rows = []
        for f in self.folds:
            row = {
                "fold": f.fold,
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
                "n_train": f.n_train,
                "n_test": f.n_test,
            }
            row.update(f.metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("fold")

    def aggregate_metrics(self) -> dict[str, float]:
        """Mean metrics across all folds (macro average)."""
        summary = self.summary()
        metric_cols = ["mae", "rmse", "directional_accuracy", "sharpe", "ic"]
        existing = [c for c in metric_cols if c in summary.columns]
        return summary[existing].mean().to_dict()


def walk_forward(
    df: pd.DataFrame,
    model_factory: type[BaseForecaster] | callable,
    *,
    initial_train_days: int = 504,
    step_days: int = 63,
    min_train_days: int = 252,
    model_kwargs: dict | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        df: Feature + target DataFrame with DatetimeIndex, sorted ascending.
            Must contain a ``target`` column.
        model_factory: Callable that returns a fresh ``BaseForecaster`` instance.
        initial_train_days: Number of rows in the first training window.
        step_days: Number of rows to advance the window each fold.
        min_train_days: Skip the fold if training rows would be fewer than this.
        model_kwargs: Keyword arguments passed to ``model_factory``.

    Returns:
        WalkForwardResult with per-fold metrics and concatenated predictions.

    Raises:
        ValueError: If ``df`` has fewer rows than ``initial_train_days + step_days``.
    """
    model_kwargs = model_kwargs or {}
    feat_cols = feature_columns(df)
    target_col = "target"

    n = len(df)
    if n < initial_train_days + step_days:
        raise ValueError(
            f"DataFrame has {n} rows but initial_train_days + step_days = "
            f"{initial_train_days + step_days}. Not enough data."
        )

    result = WalkForwardResult()
    fold = 0
    train_end_idx = initial_train_days  # exclusive upper bound (Python slice)

    while train_end_idx < n:
        test_end_idx = min(train_end_idx + step_days, n)

        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]

        if len(train_df) < min_train_days or len(test_df) == 0:
            break

        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test = test_df[feat_cols]
        y_test = test_df[target_col].to_numpy()

        model: BaseForecaster = model_factory(**model_kwargs)
        y_pred = model.fit_predict(X_train, y_train, X_test)

        metrics = compute_all(y_test, y_pred)

        fold_result = FoldResult(
            fold=fold,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            n_train=len(train_df),
            n_test=len(test_df),
            metrics=metrics,
            predictions=pd.Series(y_pred, index=test_df.index, name="prediction"),
        )
        result.folds.append(fold_result)

        logger.info(
            "Fold %d | train %s→%s (%d rows) | test %s→%s (%d rows) | "
            "dir_acc=%.3f  sharpe=%.2f",
            fold,
            fold_result.train_start.date(),
            fold_result.train_end.date(),
            fold_result.n_train,
            fold_result.test_start.date(),
            fold_result.test_end.date(),
            fold_result.n_test,
            metrics.get("directional_accuracy", float("nan")),
            metrics.get("sharpe", float("nan")),
        )

        train_end_idx += step_days
        fold += 1

    logger.info(
        "Walk-forward complete: %d folds, %d total test observations",
        len(result.folds),
        sum(f.n_test for f in result.folds),
    )
    return result
