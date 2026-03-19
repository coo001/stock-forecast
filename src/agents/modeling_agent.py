"""ModelingAgent — feature engineering and walk-forward evaluation.

Responsibilities:
- Call ``src.features.pipeline.build_feature_matrix`` with plan's feature config
- Call ``src.backtest.walk_forward.walk_forward`` with plan's backtest config
- Collect feature importance from the last-fitted model
- Serialise WalkForwardResult → ModelingResult (JSON-safe)

This agent delegates ALL computation to existing modules. It adds only:
- Input/output type safety
- Logging of progress
- Serialisation from pandas/numpy → Pydantic
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.schemas import ExecutionPlan, FoldSummary, ModelingResult
from src.backtest.walk_forward import WalkForwardResult, walk_forward
from src.features.pipeline import build_feature_matrix, feature_columns
from src.models.lgbm_model import LGBMForecaster

logger = logging.getLogger(__name__)


@dataclass
class ModelingInput:
    """Carrier passed from orchestrator to ModelingAgent."""
    plan: ExecutionPlan
    ohlcv: pd.DataFrame


class ModelingAgent(BaseAgent[ModelingInput, ModelingResult]):
    """Builds features and runs walk-forward evaluation using LGBMForecaster."""

    name = "ModelingAgent"
    timeout_seconds = 600.0   # walk-forward with many folds can take minutes

    # Side-channel: last WalkForwardResult kept for EvaluationAgent
    last_wf_result: WalkForwardResult | None = None

    def _run(self, input: ModelingInput) -> ModelingResult:
        plan = input.plan
        ohlcv = input.ohlcv
        fc = plan.feature_config
        bc = plan.backtest_config
        mc = plan.lgbm_config

        # ── Feature engineering ───────────────────────────────────────────────
        logger.info("[ModelingAgent] building feature matrix …")
        feat_df = build_feature_matrix(
            ohlcv,
            return_windows=fc["return_windows"],
            ma_windows=fc["ma_windows"],
            rsi_window=fc["rsi_window"],
            atr_window=fc["atr_window"],
            volume_ma_window=fc["volume_ma_window"],
            target_kind=plan.target_kind,
            horizon=plan.horizon,
        )
        feat_names = feature_columns(feat_df)
        logger.info("[ModelingAgent] %d rows × %d features", len(feat_df), len(feat_names))

        # ── Walk-forward ──────────────────────────────────────────────────────
        logger.info("[ModelingAgent] starting walk-forward …")

        # Keep a reference to the last model for feature importance
        last_model: list[LGBMForecaster] = []

        def model_factory() -> LGBMForecaster:
            m = LGBMForecaster(params={**mc, "verbose": -1})
            last_model.clear()
            last_model.append(m)
            return m

        wf_result = walk_forward(
            feat_df,
            model_factory,
            initial_train_days=bc["initial_train_days"],
            step_days=bc["step_days"],
            min_train_days=bc["min_train_days"],
        )
        ModelingAgent.last_wf_result = wf_result

        # ── Feature importance ────────────────────────────────────────────────
        top_features: list[str] = []
        if last_model:
            imp = last_model[0].feature_importance
            if imp is not None:
                top_features = imp.head(5).index.tolist()

        # ── Serialise folds ───────────────────────────────────────────────────
        fold_summaries = []
        for f in wf_result.folds:
            fold_summaries.append(FoldSummary(
                fold=f.fold,
                train_start=str(f.train_start.date()),
                train_end=str(f.train_end.date()),
                test_start=str(f.test_start.date()),
                test_end=str(f.test_end.date()),
                n_train=f.n_train,
                n_test=f.n_test,
                mae=_safe(f.metrics.get("mae")),
                rmse=_safe(f.metrics.get("rmse")),
                directional_accuracy=_safe(f.metrics.get("directional_accuracy")),
                sharpe=_safe(f.metrics.get("sharpe")),
                ic=_safe(f.metrics.get("ic")),
            ))

        result = ModelingResult(
            experiment_id=plan.experiment_id,
            model_name="LGBMForecaster",
            target_kind=plan.target_kind,
            n_features=len(feat_names),
            n_folds=len(wf_result.folds),
            n_oos_observations=sum(f.n_test for f in wf_result.folds),
            folds=fold_summaries,
            feature_names=feat_names,
            top_features=top_features,
        )

        logger.info(
            "[ModelingAgent] done: %d folds, %d OOS observations",
            result.n_folds, result.n_oos_observations,
        )
        return result


def _safe(v: float | None) -> float:
    """Replace None or NaN with 0.0 for JSON serialisation."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)
