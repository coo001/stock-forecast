"""Pydantic schemas for all agent inputs and outputs.

Every agent in the pipeline communicates exclusively through these types.
This eliminates ambiguity in handoffs and makes outputs trivially serialisable
to JSON for logging, storage, and future AutoGen tool-calling integration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Orchestrator input ─────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Top-level configuration for a single forecasting experiment.

    This is the only input to the orchestrator; it maps closely to
    config/default.yaml but can be overridden per experiment.
    """

    ticker: str = "005930.KS"
    start_date: str = "2015-01-01"
    end_date: str | None = None
    interval: str = "1d"
    cache_dir: str = "data/raw"

    target_kind: str = "next_day_log_return"
    horizon: int = 1

    return_windows: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    ma_windows: list[int] = Field(default_factory=lambda: [5, 20, 60])
    rsi_window: int = 14
    atr_window: int = 14
    volume_ma_window: int = 20

    initial_train_days: int = 504
    step_days: int = 63
    min_train_days: int = 252

    lgbm_params: dict[str, Any] = Field(default_factory=dict)

    # Populated by the orchestrator at runtime
    experiment_id: str = ""


# ── PlannerAgent ───────────────────────────────────────────────────────────────

class ExecutionPlan(BaseModel):
    """Validated, fully-resolved experiment plan produced by PlannerAgent."""

    experiment_id: str
    ticker: str
    target_kind: str
    horizon: int
    steps: list[str]                 # ordered pipeline step names
    feature_config: dict[str, Any]
    backtest_config: dict[str, Any]
    lgbm_config: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    created_at: str = ""
    llm_reasoning: str | None = None  # LLM explanation of feature choices (if enabled)


# ── DataAgent ─────────────────────────────────────────────────────────────────

class DataSummary(BaseModel):
    """OHLCV data quality summary produced by DataAgent."""

    ticker: str
    n_rows: int
    date_start: str
    date_end: str
    trading_days_per_year: float
    close_min: float
    close_max: float
    close_mean: float
    volume_mean: float
    missing_close_pct: float         # % of rows dropped due to NaN close
    sufficient_for_backtest: bool    # n_rows >= initial_train_days + step_days
    source: str                      # "cache" | "download" | "synthetic"


# ── ModelingAgent ─────────────────────────────────────────────────────────────

class FoldSummary(BaseModel):
    """Serialisable summary of a single walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe: float
    ic: float


class ModelingResult(BaseModel):
    """Walk-forward evaluation output from ModelingAgent."""

    model_config = {"protected_namespaces": ()}  # allow model_name field

    experiment_id: str
    model_name: str
    target_kind: str
    n_features: int
    n_folds: int
    n_oos_observations: int
    folds: list[FoldSummary]
    feature_names: list[str]
    top_features: list[str]           # top-5 by importance (if available)


# ── EvaluationAgent ───────────────────────────────────────────────────────────

class EvaluationReport(BaseModel):
    """Metric interpretation and anomaly flags from EvaluationAgent."""

    experiment_id: str
    aggregate_metrics: dict[str, float]
    directional_accuracy_mean: float
    sharpe_mean: float
    verdict: str                      # "poor" | "marginal" | "acceptable" | "strong"
    flags: list[str]                  # e.g. ["high variance across folds", "near-random DA"]
    recommendations: list[str]
    # LLM-enriched fields (None when running without LLM)
    llm_interpretation: str | None = None
    llm_recommendations: list[str] = Field(default_factory=list)


# ── ReportAgent ───────────────────────────────────────────────────────────────

class ExperimentReport(BaseModel):
    """Final structured report combining all agent outputs."""

    experiment_id: str
    generated_at: str
    config: ExperimentConfig
    plan: ExecutionPlan
    data_summary: DataSummary
    modeling_result: ModelingResult
    evaluation: EvaluationReport

    def to_text(self) -> str:
        """Human-readable summary for console output."""
        e = self.evaluation
        m = self.modeling_result
        d = self.data_summary
        lines = [
            "=" * 70,
            f"EXPERIMENT REPORT  [{self.experiment_id}]",
            f"Generated: {self.generated_at}",
            "=" * 70,
            f"Ticker     : {self.config.ticker}",
            f"Target     : {self.config.target_kind}  (horizon={self.config.horizon}d)",
            f"Data       : {d.date_start} → {d.date_end}  ({d.n_rows} rows, src={d.source})",
            f"Features   : {m.n_features}",
            f"Model      : {m.model_name}",
            f"OOS obs.   : {m.n_oos_observations}  over {m.n_folds} folds",
            "-" * 70,
            "METRICS (mean across folds)",
        ]
        for k, v in e.aggregate_metrics.items():
            lines.append(f"  {k:<28} {v:>10.4f}")
        lines += [
            "-" * 70,
            f"VERDICT    : {e.verdict.upper()}",
        ]
        if e.flags:
            lines.append("FLAGS      :")
            for f in e.flags:
                lines.append(f"  [!] {f}")
        if e.llm_interpretation:
            lines += ["-" * 70, "LLM ANALYSIS:", f"  {e.llm_interpretation}"]
        rec_list = e.llm_recommendations if e.llm_recommendations else e.recommendations
        if rec_list:
            lines.append("NEXT STEPS :")
            for r in rec_list:
                lines.append(f"  ->  {r}")
        if m.top_features:
            lines.append(f"TOP FEATURES: {', '.join(m.top_features[:5])}")
        if self.plan.llm_reasoning:
            lines += ["-" * 70, f"PLANNER (LLM): {self.plan.llm_reasoning}"]
        lines.append("=" * 70)
        return "\n".join(lines)
