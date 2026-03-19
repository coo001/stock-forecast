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

        # ── External feature merge (optional) ────────────────────────────────
        ext_cols: list[str] = []
        dropped_features_log: dict[str, str] = {}
        attempted_names: list[str] = []

        ext_enabled = bool(fc.get("external_data_enabled"))
        configured_series = fc.get("external_series") or []
        logger.info(
            "[ModelingAgent] external_data_enabled=%s, configured series=%d",
            ext_enabled, len(configured_series),
        )

        if ext_enabled and configured_series:
            from src.features.external_merge import (
                ExternalSeriesConfig,
                merge_external_features,
            )
            series_cfgs = [ExternalSeriesConfig(**s) for s in configured_series]
            attempted_names = [f"ext_{s.name}" for s in series_cfgs]
            fetch_start = ohlcv.index[0].strftime("%Y-%m-%d")
            fetch_end = ohlcv.index[-1].strftime("%Y-%m-%d")
            logger.info(
                "[ModelingAgent] merging %d external series (%s … %s)",
                len(series_cfgs), fetch_start, fetch_end,
            )
            ohlcv = merge_external_features(
                ohlcv,
                series_cfgs,
                start=fetch_start,
                end=fetch_end,
                cache_dir=fc.get("external_cache_dir", "data/external"),
                cache_ttl_hours=float(fc.get("external_cache_ttl_hours", 24.0)),
            )
            ext_cols = [c for c in ohlcv.columns if c.startswith("ext_")]

            # Log which configured series failed to merge (never appeared)
            missing_after_merge = [n for n in attempted_names if n not in ext_cols]
            for name in missing_after_merge:
                dropped_features_log[name] = "fetch failed (API error / missing key)"

            if ext_cols:
                logger.info("[ModelingAgent] external columns merged: %s", ext_cols)
            else:
                logger.warning(
                    "[ModelingAgent] external_data_enabled=True but NO ext_* columns "
                    "were merged. Check API keys, symbols, and date ranges. "
                    "Configured series: %s",
                    [s.name for s in series_cfgs],
                )
        elif ext_enabled and not configured_series:
            logger.warning(
                "[ModelingAgent] external_data_enabled=True but no series configured "
                "under external_data.series in the config file."
            )

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

        # Rejoin external features — build_feature_matrix builds its own
        # DataFrame from scratch (only uses close/high/low/volume), so
        # ext_* columns must be added back by index alignment.
        if ext_cols:
            feat_df = feat_df.join(ohlcv[ext_cols], how="left")

            # Drop columns that are entirely NaN (failed fetches / no coverage).
            dead_cols = [c for c in ext_cols if feat_df[c].isna().all()]
            if dead_cols:
                logger.warning(
                    "[ModelingAgent] dropping %d all-NaN ext columns "
                    "(fetch failed or no data coverage): %s",
                    len(dead_cols), dead_cols,
                )
                for c in dead_cols:
                    dropped_features_log[c] = "all-NaN (no data / no internet / no coverage)"
                feat_df.drop(columns=dead_cols, inplace=True)
                ext_cols = [c for c in ext_cols if c not in dead_cols]

            # Warn about high-NaN columns (keep them — LightGBM handles NaN)
            for c in ext_cols:
                nan_pct = feat_df[c].isna().mean()
                if nan_pct > 0.5:
                    logger.warning(
                        "[ModelingAgent] ext col '%s' is %.0f%% NaN — "
                        "low coverage, may not contribute to model",
                        c, nan_pct * 100,
                    )

            # Drop rows where non-external columns are NaN (target + tech warmup).
            # LightGBM handles NaN natively; partial ext NaN is acceptable.
            non_ext = [c for c in feat_df.columns if not c.startswith("ext_")]
            feat_df.dropna(subset=non_ext, inplace=True)

            logger.info(
                "[ModelingAgent] after external join: %d rows, %d ext cols",
                len(feat_df), len(ext_cols),
            )

        feat_names = feature_columns(feat_df)
        n_tech = len(feat_names) - len(ext_cols)
        logger.info(
            "[ModelingAgent] feature matrix: %d rows × %d features "
            "(%d technical + %d external)",
            len(feat_df), len(feat_names), n_tech, len(ext_cols),
        )

        # Compute per-column missing ratio for external features
        ext_missing: dict[str, float] = {}
        for col in ext_cols:
            if col in feat_df.columns:
                ext_missing[col] = float(feat_df[col].isna().mean())
        if ext_missing:
            logger.info(
                "[ModelingAgent] external missing ratios: %s",
                {k: f"{v*100:.1f}%" for k, v in ext_missing.items()},
            )

        # Enforce: enabled but 0 ext columns → warning
        if ext_enabled and not ext_cols:
            logger.warning(
                "[ModelingAgent] external_data_enabled=True but 0 ext_* "
                "columns ended up in the feature matrix after merge. "
                "All configured series may have failed silently."
            )

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
            external_columns=ext_cols,
            external_missing_ratios=ext_missing,
            attempted_external_features=attempted_names,
            dropped_features_log=dropped_features_log,
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
