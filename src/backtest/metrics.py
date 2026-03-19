"""Evaluation metrics for forecasting models.

All functions are pure – they take arrays/series and return scalars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign.

    When y_true == 0, the row is skipped (no information day).

    Args:
        y_true: Actual log returns.
        y_pred: Predicted log returns.

    Returns:
        Float in [0, 1].
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE in the same units as the target."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (IC) between predictions and realised returns.

    Returns NaN if the series is constant or too short.
    """
    from scipy.stats import spearmanr  # optional dependency

    if len(y_true) < 2:
        return float("nan")
    corr, _ = spearmanr(y_pred, y_true)
    return float(corr)


def annualised_sharpe(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Sharpe ratio of a daily-return series, annualised by sqrt(252).

    Args:
        daily_returns: Strategy daily P&L (not log returns from the model,
                       but realised P&L e.g. sign(pred) * actual_return).
        risk_free: Daily risk-free rate (default 0).

    Returns:
        Annualised Sharpe ratio.
    """
    excess = daily_returns - risk_free
    std = np.std(excess, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(252))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the full metric suite and return a dict.

    Args:
        y_true: Actual log returns.
        y_pred: Predicted log returns.

    Returns:
        Dict with keys: mae, rmse, directional_accuracy, ic, sharpe.
    """
    strategy_returns = np.sign(y_pred) * y_true  # long/short signal × actual

    metrics: dict[str, float] = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "sharpe": annualised_sharpe(strategy_returns),
    }

    try:
        metrics["ic"] = information_coefficient(y_true, y_pred)
    except ImportError:
        metrics["ic"] = float("nan")

    return metrics
