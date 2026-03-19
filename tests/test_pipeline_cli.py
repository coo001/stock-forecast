"""Tests for run_pipeline.py CLI wiring.

Verifies that the command-line entry point correctly assembles
ExperimentConfig, LLM client, and the agent pipeline — without
touching any real network or LLM API.

Strategy:
- Patch run_experiment so it returns a pre-built ExperimentReport
- Test each CLI flag path in isolation
- Test build_llm_client factory directly
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.llm_client import LLMClient, build_llm_client
from src.agents.schemas import (
    DataSummary,
    EvaluationReport,
    ExperimentConfig,
    ExperimentReport,
    ExecutionPlan,
    ModelingResult,
)


# ── Autouse: bypass preflight startup checks in all CLI tests ─────────────────
# check_required() and require_yfinance() call sys.exit(1) when packages are
# absent; we stub them out so tests focus on CLI logic, not the environment.

@pytest.fixture(autouse=True)
def _no_startup_checks():
    with patch("run_pipeline.check_required"), patch("run_pipeline.require_yfinance"):
        yield


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _minimal_report(experiment_id: str = "test_exp") -> ExperimentReport:
    """Build the smallest valid ExperimentReport for mocking."""
    config = ExperimentConfig(
        experiment_id=experiment_id,
        initial_train_days=252,
        step_days=42,
        min_train_days=126,
    )
    plan = ExecutionPlan(
        experiment_id=experiment_id,
        ticker="005930.KS",
        target_kind="next_day_log_return",
        horizon=1,
        steps=["DataAgent", "ModelingAgent", "EvaluationAgent", "ReportAgent"],
        feature_config={"return_windows": [1, 5], "ma_windows": [5, 20],
                        "rsi_window": 14, "atr_window": 14, "volume_ma_window": 20},
        backtest_config={"initial_train_days": 252, "step_days": 42, "min_train_days": 126},
        lgbm_config={"n_estimators": 100, "random_state": 42},
        created_at="2026-01-01T00:00:00",
    )
    data_summary = DataSummary(
        ticker="005930.KS", n_rows=800,
        date_start="2015-01-02", date_end="2018-03-01",
        trading_days_per_year=252.0,
        close_min=50000.0, close_max=70000.0, close_mean=60000.0,
        volume_mean=1e7, missing_close_pct=0.0,
        sufficient_for_backtest=True, source="cache",
    )
    modeling = ModelingResult(
        experiment_id=experiment_id,
        model_name="LGBMForecaster",
        target_kind="next_day_log_return",
        n_features=10, n_folds=3, n_oos_observations=126,
        folds=[], feature_names=[], top_features=[],
    )
    evaluation = EvaluationReport(
        experiment_id=experiment_id,
        aggregate_metrics={"mae": 0.005, "rmse": 0.007, "directional_accuracy": 0.52,
                           "sharpe": 0.4, "ic": 0.03},
        directional_accuracy_mean=0.52, sharpe_mean=0.4,
        verdict="marginal", flags=[], recommendations=["Add macro features."],
    )
    return ExperimentReport(
        experiment_id=experiment_id,
        generated_at="2026-01-01T00:00:00+00:00",
        config=config, plan=plan,
        data_summary=data_summary, modeling_result=modeling, evaluation=evaluation,
    )


@pytest.fixture()
def minimal_config_yaml(tmp_path) -> Path:
    """Write a minimal valid YAML config to a temp file."""
    content = textwrap.dedent("""\
        data:
          ticker: "005930.KS"
          start_date: "2015-01-01"
          end_date: null
          interval: "1d"
          cache_dir: "data/raw"
        target:
          kind: "next_day_log_return"
          horizon: 1
        features:
          return_windows: [1, 5, 10, 20]
          ma_windows: [5, 20, 60]
          rsi_window: 14
          atr_window: 14
          volume_ma_window: 20
        backtest:
          initial_train_days: 252
          step_days: 42
          min_train_days: 126
        lgbm:
          n_estimators: 100
          random_state: 42
        llm:
          provider: "none"
        logging:
          level: "WARNING"
          format: "%(message)s"
    """)
    p = tmp_path / "test_config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ── build_llm_client factory tests ────────────────────────────────────────────

def test_build_llm_client_none_returns_none():
    assert build_llm_client({"provider": "none"}) is None


def test_build_llm_client_empty_config_returns_none():
    assert build_llm_client({}) is None


def test_build_llm_client_openai_returns_client():
    client = build_llm_client(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}
    )
    assert isinstance(client, LLMClient)
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_build_llm_client_provider_override():
    """CLI --llm-provider overrides config."""
    client = build_llm_client(
        {"provider": "none"},
        provider_override="anthropic",
    )
    assert client is not None
    assert client.provider == "anthropic"


def test_build_llm_client_model_override():
    """CLI --llm-model overrides config model."""
    client = build_llm_client(
        {"provider": "openai", "model": "gpt-4o"},
        model_override="gpt-4o-mini",
    )
    assert client.model == "gpt-4o-mini"


def test_build_llm_client_provider_override_none_wins():
    """Passing provider_override='none' disables LLM even if config says openai."""
    result = build_llm_client(
        {"provider": "openai"},
        provider_override="none",
    )
    assert result is None


# ── Config → ExperimentConfig mapping tests ───────────────────────────────────

def test_build_experiment_config_reads_all_sections(minimal_config_yaml):
    """_build_experiment_config must map every YAML section correctly."""
    import yaml
    import argparse
    from run_pipeline import _build_experiment_config

    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    args = argparse.Namespace(experiment_id="")

    ec = _build_experiment_config(cfg, args)
    assert ec.ticker == "005930.KS"
    assert ec.target_kind == "next_day_log_return"
    assert ec.return_windows == [1, 5, 10, 20]
    assert ec.initial_train_days == 252
    assert ec.lgbm_params["n_estimators"] == 100


def test_build_experiment_config_uses_experiment_id(minimal_config_yaml):
    import yaml, argparse
    from run_pipeline import _build_experiment_config

    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    args = argparse.Namespace(experiment_id="my_exp_001")
    ec = _build_experiment_config(cfg, args)
    assert ec.experiment_id == "my_exp_001"


# ── --no-download flag ─────────────────────────────────────────────────────────

def test_no_download_raises_if_no_cache(minimal_config_yaml, tmp_path):
    """--no-download must raise FileNotFoundError when cache is absent."""
    import argparse, yaml
    from run_pipeline import _load_data

    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    # Point cache_dir to an empty tmp dir so no cache exists
    cfg["data"]["cache_dir"] = str(tmp_path)
    args = argparse.Namespace(no_download=True, force_refresh=False)

    with pytest.raises(FileNotFoundError, match="--no-download"):
        _load_data(cfg, args)


def test_no_download_loads_from_csv_when_cache_exists(minimal_config_yaml, tmp_path):
    """--no-download loads the cache CSV and returns a DataFrame."""
    import argparse, yaml
    from run_pipeline import _load_data
    from src.data.loader import _cache_path
    from src.data.schema import DataConfig

    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    cfg["data"]["cache_dir"] = str(tmp_path)

    # Create a synthetic cache CSV in the right location
    data_config = DataConfig(**cfg["data"])
    cache = _cache_path(data_config)

    rng = np.random.default_rng(0)
    n = 100
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 60_000 + np.cumsum(rng.normal(0, 300, n))
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1e6, 1e7, n).astype(float)},
        index=idx,
    )
    df.index.name = "date"
    df.to_csv(cache)

    args = argparse.Namespace(no_download=True, force_refresh=False)
    result = _load_data(cfg, args)
    assert result is not None
    assert len(result) == n


def test_default_mode_returns_none(minimal_config_yaml):
    """Default (no flags) returns None so DataAgent handles download."""
    import argparse, yaml
    from run_pipeline import _load_data

    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    args = argparse.Namespace(no_download=False, force_refresh=False)
    assert _load_data(cfg, args) is None


# ── End-to-end CLI tests (run_experiment mocked) ──────────────────────────────

def test_main_returns_zero_on_success(minimal_config_yaml, tmp_path):
    """main() returns 0 and prints a report when pipeline succeeds."""
    from run_pipeline import main

    report = _minimal_report("test_success")

    with patch("run_pipeline.run_experiment", return_value=report) as mock_run:
        rc = main([
            "--config", str(minimal_config_yaml),
            "--reports-dir", str(tmp_path),
        ])

    assert rc == 0
    mock_run.assert_called_once()


def test_main_passes_llm_client_to_run_experiment(minimal_config_yaml, tmp_path):
    """--llm-provider cli flag causes a real LLMClient to be passed."""
    from run_pipeline import main

    report = _minimal_report()

    with patch("run_pipeline.run_experiment", return_value=report) as mock_run:
        main([
            "--config", str(minimal_config_yaml),
            "--llm-provider", "openai",
            "--reports-dir", str(tmp_path),
        ])

    _, kwargs = mock_run.call_args
    assert kwargs.get("llm_client") is not None
    assert kwargs["llm_client"].provider == "openai"


def test_main_llm_none_passes_none_client(minimal_config_yaml, tmp_path):
    """provider=none (default) means llm_client=None is passed."""
    from run_pipeline import main

    report = _minimal_report()

    with patch("run_pipeline.run_experiment", return_value=report) as mock_run:
        main(["--config", str(minimal_config_yaml), "--reports-dir", str(tmp_path)])

    _, kwargs = mock_run.call_args
    assert kwargs.get("llm_client") is None


def test_main_passes_experiment_id(minimal_config_yaml, tmp_path):
    """--experiment-id is forwarded into ExperimentConfig."""
    from run_pipeline import main

    report = _minimal_report("cli_given_id")

    with patch("run_pipeline.run_experiment", return_value=report) as mock_run:
        main([
            "--config", str(minimal_config_yaml),
            "--experiment-id", "cli_given_id",
            "--reports-dir", str(tmp_path),
        ])

    positional_config = mock_run.call_args[0][0]
    assert positional_config.experiment_id == "cli_given_id"


def test_main_returns_one_on_pipeline_failure(minimal_config_yaml, tmp_path):
    """main() returns 1 when run_experiment raises."""
    from run_pipeline import main

    with patch("run_pipeline.run_experiment", side_effect=RuntimeError("boom")):
        rc = main(["--config", str(minimal_config_yaml), "--reports-dir", str(tmp_path)])

    assert rc == 1


def test_main_returns_one_on_no_download_missing_cache(minimal_config_yaml, tmp_path):
    """main() returns 1 (not an exception) when --no-download cache is absent."""
    from run_pipeline import main

    # cache_dir is empty tmp_path → no cache exists
    import yaml
    cfg = yaml.safe_load(minimal_config_yaml.read_text())
    cfg["data"]["cache_dir"] = str(tmp_path / "empty")
    cfg_path = tmp_path / "modified.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    rc = main(["--config", str(cfg_path), "--no-download", "--reports-dir", str(tmp_path)])
    assert rc == 1


# ── --synthetic flag ───────────────────────────────────────────────────────────

def test_synthetic_flag_passes_dataframe_to_run_experiment(minimal_config_yaml, tmp_path):
    """--synthetic must generate a DataFrame and pass it to run_experiment."""
    from run_pipeline import main

    report = _minimal_report("synthetic_test")

    captured_kwargs = {}

    def capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return report

    with patch("run_pipeline.run_experiment", side_effect=capture):
        rc = main([
            "--config", str(minimal_config_yaml),
            "--synthetic",
            "--reports-dir", str(tmp_path),
        ])

    assert rc == 0
    assert "ohlcv_df" in captured_kwargs
    df = captured_kwargs["ohlcv_df"]
    assert df is not None
    assert hasattr(df, "columns")
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}


def test_synthetic_flag_returns_zero_on_success(minimal_config_yaml, tmp_path):
    """main() with --synthetic returns 0 when pipeline succeeds."""
    from run_pipeline import main

    report = _minimal_report("synthetic_ok")

    with patch("run_pipeline.run_experiment", return_value=report):
        rc = main([
            "--config", str(minimal_config_yaml),
            "--synthetic",
            "--reports-dir", str(tmp_path),
        ])

    assert rc == 0


def test_synthetic_exclusive_with_no_download(minimal_config_yaml, tmp_path):
    """--synthetic and --no-download are mutually exclusive; argparse should reject."""
    from run_pipeline import main
    import pytest

    with pytest.raises(SystemExit):
        main([
            "--config", str(minimal_config_yaml),
            "--synthetic",
            "--no-download",
        ])
