"""Unit tests for the agent layer.

Tests run fully offline: no yfinance, no LLM API calls.
Synthetic DataFrames are used throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import AgentError, BaseAgent
from src.agents.data_agent import DataAgent
from src.agents.evaluation_agent import EvaluationAgent, EvaluationInput
from src.agents.modeling_agent import ModelingAgent, ModelingInput
from src.agents.orchestrator import run_experiment
from src.agents.planner_agent import PlannerAgent
from src.agents.report_agent import ReportAgent, ReportInput
from src.agents.schemas import (
    ExperimentConfig,
    ExecutionPlan,
    FoldSummary,
    ModelingResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 60_000 + np.cumsum(rng.normal(0, 500, n))
    close = np.abs(close)
    spread = close * 0.005
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, spread, n),
            "high": close + np.abs(rng.normal(0, spread, n)),
            "low": close - np.abs(rng.normal(0, spread, n)),
            "close": close,
            "volume": np.abs(rng.lognormal(16, 0.5, n)),
        },
        index=idx,
    )


@pytest.fixture()
def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        ticker="005930.KS",
        initial_train_days=252,
        step_days=42,
        min_train_days=126,
    )


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


# ── BaseAgent tests ───────────────────────────────────────────────────────────

class _AlwaysOK(BaseAgent[str, str]):
    name = "AlwaysOK"
    def _run(self, input: str) -> str:
        return input.upper()


class _AlwaysFails(BaseAgent[str, str]):
    name = "AlwaysFails"
    max_retries = 1
    def _run(self, input: str) -> str:
        raise ValueError("boom")


def test_base_agent_success():
    assert _AlwaysOK().run("hello") == "HELLO"


def test_base_agent_raises_agent_error():
    with pytest.raises(AgentError) as exc_info:
        _AlwaysFails().run("x")
    assert exc_info.value.agent_name == "AlwaysFails"


def test_base_agent_as_autogen_tool():
    tool = _AlwaysOK().as_autogen_tool()
    assert tool["name"] == "AlwaysOK"
    assert callable(tool["callable"])
    assert tool["callable"]("hi") == "HI"


# ── PlannerAgent tests ────────────────────────────────────────────────────────

def test_planner_produces_plan(default_config):
    plan = PlannerAgent().run(default_config)
    assert plan.experiment_id
    assert plan.ticker == "005930.KS"
    assert "DataAgent" in plan.steps
    assert "ModelingAgent" in plan.steps


def test_planner_rejects_bad_target(default_config):
    cfg = default_config.model_copy(update={"target_kind": "nonsense"})
    with pytest.raises(AgentError) as exc_info:
        PlannerAgent().run(cfg)
    assert "Unknown target_kind" in str(exc_info.value.cause)


def test_planner_warns_on_mismatched_horizon(default_config):
    cfg = default_config.model_copy(
        update={"target_kind": "next_5d_log_return", "horizon": 1}
    )
    plan = PlannerAgent().run(cfg)
    assert any("horizon" in w for w in plan.warnings)


def test_planner_assigns_experiment_id(default_config):
    cfg = default_config.model_copy(update={"experiment_id": ""})
    plan = PlannerAgent().run(cfg)
    assert len(plan.experiment_id) > 0


# ── DataAgent tests ───────────────────────────────────────────────────────────

def test_data_agent_with_preloaded_df(default_config, ohlcv):
    plan = PlannerAgent().run(default_config)
    agent = DataAgent()
    summary = agent.run_with_df(plan, ohlcv_df=ohlcv)

    assert summary.n_rows == len(ohlcv)
    assert summary.source == "synthetic"
    assert summary.sufficient_for_backtest
    assert agent.last_df is not None


def test_data_agent_flags_insufficient_data(default_config):
    plan = PlannerAgent().run(default_config)
    tiny_df = _synthetic_ohlcv(n=50)  # way too small
    agent = DataAgent()
    summary = agent.run_with_df(plan, ohlcv_df=tiny_df)
    assert not summary.sufficient_for_backtest


# ── ModelingAgent tests ───────────────────────────────────────────────────────

def _make_plan(config: ExperimentConfig) -> ExecutionPlan:
    return PlannerAgent().run(config)


def test_modeling_agent_runs(default_config, ohlcv):
    plan = _make_plan(default_config)
    result = ModelingAgent().run(ModelingInput(plan=plan, ohlcv=ohlcv))

    assert result.n_folds >= 1
    assert result.n_features > 0
    assert result.n_oos_observations > 0
    assert result.model_name == "LGBMForecaster"


def test_modeling_agent_folds_have_metrics(default_config, ohlcv):
    plan = _make_plan(default_config)
    result = ModelingAgent().run(ModelingInput(plan=plan, ohlcv=ohlcv))

    for fold in result.folds:
        assert 0.0 <= fold.directional_accuracy <= 1.0
        assert fold.mae >= 0.0


# ── EvaluationAgent tests ─────────────────────────────────────────────────────

def _dummy_modeling_result(da: float, sharpe: float, n_folds: int = 5) -> ModelingResult:
    folds = [
        FoldSummary(
            fold=i,
            train_start="2016-01-01", train_end="2017-12-31",
            test_start="2018-01-01", test_end="2018-03-31",
            n_train=504, n_test=63,
            mae=0.005, rmse=0.007,
            directional_accuracy=da,
            sharpe=sharpe,
            ic=0.03,
        )
        for i in range(n_folds)
    ]
    return ModelingResult(
        experiment_id="test",
        model_name="LGBMForecaster",
        target_kind="next_day_log_return",
        n_features=12,
        n_folds=n_folds,
        n_oos_observations=n_folds * 63,
        folds=folds,
        feature_names=[f"f{i}" for i in range(12)],
        top_features=["log_ret_1d", "rsi_14"],
    )


def _dummy_plan(config: ExperimentConfig) -> ExecutionPlan:
    return PlannerAgent().run(config)


def test_evaluation_strong(default_config):
    plan = _dummy_plan(default_config)
    mr = _dummy_modeling_result(da=0.60, sharpe=1.0)
    report = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=mr))
    assert report.verdict == "strong"


def test_evaluation_poor(default_config):
    plan = _dummy_plan(default_config)
    mr = _dummy_modeling_result(da=0.49, sharpe=-0.5)
    report = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=mr))
    assert report.verdict == "poor"
    assert any("near random" in f for f in report.flags)


def test_evaluation_negative_sharpe_flagged(default_config):
    plan = _dummy_plan(default_config)
    mr = _dummy_modeling_result(da=0.55, sharpe=-0.2)
    report = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=mr))
    assert any("negative" in f for f in report.flags)


def test_evaluation_recommendations_not_empty(default_config):
    plan = _dummy_plan(default_config)
    mr = _dummy_modeling_result(da=0.51, sharpe=0.1)
    report = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=mr))
    assert len(report.recommendations) > 0


# ── Full pipeline (orchestrator) tests ────────────────────────────────────────

def test_full_pipeline_synthetic(default_config, ohlcv, tmp_path):
    report = run_experiment(default_config, ohlcv_df=ohlcv, reports_dir=tmp_path)

    assert report.experiment_id
    assert report.data_summary.n_rows == len(ohlcv)
    assert report.modeling_result.n_folds >= 1
    assert report.evaluation.verdict in ("poor", "marginal", "acceptable", "strong")


def test_full_pipeline_writes_json(default_config, ohlcv, tmp_path):
    report = run_experiment(default_config, ohlcv_df=ohlcv, reports_dir=tmp_path)
    json_path = tmp_path / f"{report.experiment_id}.json"
    assert json_path.exists()
    assert json_path.stat().st_size > 100


def test_full_pipeline_report_text(default_config, ohlcv, tmp_path):
    report = run_experiment(default_config, ohlcv_df=ohlcv, reports_dir=tmp_path)
    text = report.to_text()
    assert "EXPERIMENT REPORT" in text
    assert "VERDICT" in text
    assert report.experiment_id in text


def test_pipeline_insufficient_data_raises(default_config, tmp_path):
    tiny_df = _synthetic_ohlcv(n=50)
    with pytest.raises(RuntimeError, match="Insufficient data"):
        run_experiment(default_config, ohlcv_df=tiny_df, reports_dir=tmp_path)
