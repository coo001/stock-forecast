"""Tests for LLM-integrated agent behaviour.

All tests use MockLLMClient — no real API calls, no env vars required.
The suite verifies:
  - MockLLMClient round-trips correctly
  - EvaluationAgent enriches the rule-based report with LLM output
  - EvaluationAgent falls back gracefully on LLM failure
  - EvaluationAgent rejects invalid verdicts from the LLM
  - PlannerAgent applies LLM feature suggestions when past reports exist
  - PlannerAgent clamps out-of-bounds values from the LLM
  - PlannerAgent falls back when LLM fails
  - PlannerAgent skips LLM when no past reports exist
  - Full orchestrator pipeline runs end-to-end with MockLLMClient
  - LLM fields appear in ExperimentReport.to_text() output
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.agents.evaluation_agent import EvaluationAgent, EvaluationInput
from src.agents.llm_client import LLMClient, LLMError, _MockLLMClient, _parse_json
from src.agents.orchestrator import run_experiment
from src.agents.planner_agent import PlannerAgent
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
def base_config() -> ExperimentConfig:
    return ExperimentConfig(
        ticker="005930.KS",
        initial_train_days=252,
        step_days=42,
        min_train_days=126,
    )


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


def _dummy_plan(config: ExperimentConfig) -> ExecutionPlan:
    return PlannerAgent().run(config)


def _dummy_modeling(n_folds: int = 5, da: float = 0.51, sharpe: float = 0.5) -> ModelingResult:
    folds = [
        FoldSummary(
            fold=i,
            train_start="2016-01-01", train_end="2017-12-31",
            test_start="2018-01-01", test_end="2018-06-30",
            n_train=504, n_test=126,
            mae=0.005, rmse=0.007,
            directional_accuracy=da, sharpe=sharpe, ic=0.03,
        )
        for i in range(n_folds)
    ]
    return ModelingResult(
        experiment_id="test",
        model_name="LGBMForecaster",
        target_kind="next_day_log_return",
        n_features=10,
        n_folds=n_folds,
        n_oos_observations=n_folds * 126,
        folds=folds,
        feature_names=[f"f{i}" for i in range(10)],
        top_features=["log_ret_1d", "rsi_14", "atr_pct_14"],
    )


def _mock_eval_response(verdict: str = "acceptable") -> dict:
    return {
        "interpretation": "Test interpretation sentence.",
        "verdict": verdict,
        "llm_recommendations": ["Rec A", "Rec B"],
    }


def _mock_planner_response() -> dict:
    return {
        "reasoning": "Extending windows based on prior results.",
        "return_windows": [1, 5, 20, 60],
        "ma_windows": [10, 30, 90],
        "rsi_window": 21,
        "atr_window": 10,
        "volume_ma_window": 15,
        "lgbm_overrides": {"num_leaves": 15},
    }


# ── LLMClient / Mock tests ─────────────────────────────────────────────────────

def test_mock_returns_configured_dict():
    response = {"verdict": "strong", "interpretation": "Great."}
    client = LLMClient.mock(response)
    result = client.chat_json("sys", "user")
    assert result["verdict"] == "strong"


def test_mock_cycles_through_list():
    r1 = {"call": 1}
    r2 = {"call": 2}
    client = LLMClient.mock([r1, r2])
    assert client.chat_json("s", "u")["call"] == 1
    assert client.chat_json("s", "u")["call"] == 2
    # exhausted → repeats last
    assert client.chat_json("s", "u")["call"] == 2


def test_mock_returns_copy_not_reference():
    response = {"key": "value"}
    client = LLMClient.mock(response)
    result = client.chat_json("s", "u")
    result["key"] = "mutated"
    # original should be unchanged
    assert client.chat_json("s", "u")["key"] == "value"


def test_parse_json_plain():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_with_markdown_block():
    text = '```json\n{"a": 1}\n```'
    assert _parse_json(text) == {"a": 1}


def test_parse_json_embedded_in_prose():
    text = 'Here is the result: {"a": 1} as requested.'
    assert _parse_json(text) == {"a": 1}


def test_parse_json_raises_on_garbage():
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        _parse_json("not json at all")


def test_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider must be one of"):
        LLMClient(provider="fakecloud")


# ── EvaluationAgent + LLM ─────────────────────────────────────────────────────

def test_evaluation_agent_llm_adds_interpretation(base_config):
    plan = _dummy_plan(base_config)
    modeling = _dummy_modeling()
    client = LLMClient.mock(_mock_eval_response("acceptable"))

    report = EvaluationAgent(llm_client=client).run(EvaluationInput(plan=plan, modeling=modeling))

    assert report.llm_interpretation == "Test interpretation sentence."
    assert report.llm_recommendations == ["Rec A", "Rec B"]


def test_evaluation_agent_llm_can_change_verdict(base_config):
    plan = _dummy_plan(base_config)
    # Rule-based will be "marginal" (DA=0.51, Sharpe=0.5)
    modeling = _dummy_modeling(da=0.51, sharpe=0.5)
    # LLM upgrades to "acceptable"
    client = LLMClient.mock(_mock_eval_response("acceptable"))

    report = EvaluationAgent(llm_client=client).run(EvaluationInput(plan=plan, modeling=modeling))
    assert report.verdict == "acceptable"


def test_evaluation_agent_llm_rejects_invalid_verdict(base_config):
    plan = _dummy_plan(base_config)
    modeling = _dummy_modeling(da=0.55, sharpe=0.6)
    # LLM returns a nonsense verdict
    client = LLMClient.mock({"interpretation": "Good.", "verdict": "EXCELLENT", "llm_recommendations": []})

    report = EvaluationAgent(llm_client=client).run(EvaluationInput(plan=plan, modeling=modeling))
    # Must fall back to rule-based verdict, not "EXCELLENT"
    assert report.verdict in {"poor", "marginal", "acceptable", "strong"}
    assert report.verdict != "EXCELLENT"


def test_evaluation_agent_falls_back_on_llm_error(base_config):
    """If LLM raises, the rule-based report is returned unchanged."""
    plan = _dummy_plan(base_config)
    modeling = _dummy_modeling()

    class _BrokenClient(_MockLLMClient):
        def chat_json(self, *args, **kwargs):
            raise LLMError("simulated API error")

    broken = _BrokenClient({})
    report_no_llm = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=modeling))
    report_broken = EvaluationAgent(llm_client=broken).run(EvaluationInput(plan=plan, modeling=modeling))

    # Verdict must match rule-based; no LLM fields populated
    assert report_broken.verdict == report_no_llm.verdict
    assert report_broken.llm_interpretation is None
    assert report_broken.llm_recommendations == []


def test_evaluation_agent_no_llm_leaves_fields_none(base_config):
    plan = _dummy_plan(base_config)
    modeling = _dummy_modeling()
    report = EvaluationAgent().run(EvaluationInput(plan=plan, modeling=modeling))
    assert report.llm_interpretation is None
    assert report.llm_recommendations == []


# ── PlannerAgent + LLM ─────────────────────────────────────────────────────────

def test_planner_agent_skips_llm_with_no_past_reports(base_config, tmp_path):
    """When reports_dir is empty, LLM is skipped even if client is provided."""
    client = LLMClient.mock(_mock_planner_response())
    plan = PlannerAgent(llm_client=client, reports_dir=tmp_path).run(base_config)
    # LLM should not have been called — feature config unchanged from deterministic
    assert client._call_count == 0
    assert plan.llm_reasoning is None


def test_planner_agent_applies_llm_when_history_exists(base_config, tmp_path):
    """When past reports exist, LLM updates feature_config and sets llm_reasoning."""
    _write_fake_report(tmp_path, "exp_001")

    client = LLMClient.mock(_mock_planner_response())
    plan = PlannerAgent(llm_client=client, reports_dir=tmp_path).run(base_config)

    assert client._call_count == 1
    assert plan.feature_config["return_windows"] == [1, 5, 20, 60]
    assert plan.feature_config["rsi_window"] == 21
    assert plan.llm_reasoning == "Extending windows based on prior results."


def test_planner_agent_clamps_out_of_bounds_windows(base_config, tmp_path):
    """LLM-returned values outside bounds are clamped, not rejected outright."""
    _write_fake_report(tmp_path, "exp_002")
    bad_response = {
        "reasoning": "Testing bounds.",
        "return_windows": [0, 5, 200],   # 0 → 1, 200 → 120
        "ma_windows": [3, 20],            # 3 → 5
        "rsi_window": 50,                 # 50 → 30
        "atr_window": 2,                  # 2 → 7
        "volume_ma_window": 100,          # 100 → 60
        "lgbm_overrides": {},
    }
    client = LLMClient.mock(bad_response)
    plan = PlannerAgent(llm_client=client, reports_dir=tmp_path).run(base_config)

    assert 1 in plan.feature_config["return_windows"]
    assert 120 in plan.feature_config["return_windows"]
    assert all(w >= 5 for w in plan.feature_config["ma_windows"])
    assert plan.feature_config["rsi_window"] == 30
    assert plan.feature_config["atr_window"] == 7
    assert plan.feature_config["volume_ma_window"] == 60


def test_planner_agent_falls_back_on_llm_error(base_config, tmp_path):
    _write_fake_report(tmp_path, "exp_003")

    class _BrokenClient(_MockLLMClient):
        def chat_json(self, *args, **kwargs):
            raise LLMError("simulated error")

    broken = _BrokenClient({})
    plan_no_llm = PlannerAgent(reports_dir=tmp_path).run(base_config)
    plan_broken = PlannerAgent(llm_client=broken, reports_dir=tmp_path).run(base_config)

    assert plan_broken.feature_config == plan_no_llm.feature_config
    assert plan_broken.llm_reasoning is None


def test_planner_agent_applies_lgbm_overrides(base_config, tmp_path):
    _write_fake_report(tmp_path, "exp_004")
    response = {**_mock_planner_response(), "lgbm_overrides": {"num_leaves": 63, "learning_rate": 0.01}}
    client = LLMClient.mock(response)
    plan = PlannerAgent(llm_client=client, reports_dir=tmp_path).run(base_config)

    assert plan.lgbm_config["num_leaves"] == 63
    assert abs(plan.lgbm_config["learning_rate"] - 0.01) < 1e-9


# ── Full pipeline with LLM ─────────────────────────────────────────────────────

def test_full_pipeline_with_mock_llm(base_config, ohlcv, tmp_path):
    """End-to-end with MockLLMClient; LLM fields must appear in the report."""
    # First call → EvaluationAgent (PlannerAgent skips — no history yet)
    client = LLMClient.mock([_mock_eval_response("marginal"), _mock_planner_response()])

    report = run_experiment(
        base_config,
        ohlcv_df=ohlcv,
        reports_dir=tmp_path,
        llm_client=client,
    )

    assert report.evaluation.llm_interpretation is not None
    assert len(report.evaluation.llm_recommendations) > 0


def test_full_pipeline_second_run_uses_planner_llm(base_config, ohlcv, tmp_path):
    """On second run, PlannerAgent finds history and also calls LLM."""
    # Run 1: builds history
    run_experiment(base_config, ohlcv_df=ohlcv, reports_dir=tmp_path)

    # Orchestrator calls: Step 1 PlannerAgent, then Step 4 EvaluationAgent
    client = LLMClient.mock([_mock_planner_response(), _mock_eval_response()])
    report2 = run_experiment(
        base_config,
        ohlcv_df=ohlcv,
        reports_dir=tmp_path,
        llm_client=client,
    )

    # PlannerAgent should have updated feature config from LLM
    assert report2.plan.feature_config["return_windows"] == [1, 5, 20, 60]
    assert report2.plan.llm_reasoning is not None


def test_llm_fields_appear_in_to_text(base_config, ohlcv, tmp_path):
    client = LLMClient.mock(_mock_eval_response())
    report = run_experiment(
        base_config, ohlcv_df=ohlcv, reports_dir=tmp_path, llm_client=client
    )
    text = report.to_text()
    assert "LLM ANALYSIS" in text
    assert "Test interpretation sentence." in text


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_fake_report(reports_dir, experiment_id: str) -> None:
    """Write a minimal ExperimentReport JSON so PlannerAgent finds history."""
    report = {
        "experiment_id": experiment_id,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "config": {"target_kind": "next_day_log_return"},
        "plan": {
            "feature_config": {
                "return_windows": [1, 5, 10, 20],
                "ma_windows": [5, 20, 60],
                "rsi_window": 14, "atr_window": 14, "volume_ma_window": 20,
            }
        },
        "modeling_result": {
            "n_folds": 5, "n_features": 10,
            "top_features": ["log_ret_1d", "rsi_14", "atr_pct_14"],
        },
        "evaluation": {
            "directional_accuracy_mean": 0.51,
            "sharpe_mean": 0.65,
            "aggregate_metrics": {"ic": 0.03},
            "verdict": "marginal",
        },
    }
    path = reports_dir / f"{experiment_id}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
