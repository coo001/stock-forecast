"""Explicit pipeline orchestrator for the Samsung forecasting multi-agent system.

This is the single public entry point for the agent layer::

    from src.agents.orchestrator import run_experiment
    from src.agents.schemas import ExperimentConfig, ExperimentReport

    report = run_experiment(ExperimentConfig(ticker="005930.KS"))

## Design rationale

AutoGen's GroupChat lets agents decide who speaks next, which is powerful
but non-deterministic and hard to debug.  CLAUDE.md requires:
  "Prefer deterministic handoff logic over vague agent chatter"

So this orchestrator drives a fixed pipeline:

    PlannerAgent → DataAgent → ModelingAgent → EvaluationAgent → ReportAgent

Each agent receives the *typed output* of the previous step, not a chat
message.  The orchestrator is responsible for:
  - Instantiating agents
  - Passing data between them (including the non-serialisable DataFrame)
  - Catching AgentError and logging structured failure reasons
  - Returning a complete ExperimentReport

## AutoGen v2 migration path

When LLM-based reasoning is added, the orchestrator can be replaced by an
AutoGen Swarm or GroupChat coordinator while keeping every agent's
``_run()`` implementation unchanged.  The ``as_autogen_tool()`` method on
each agent provides the registration metadata for that migration.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.base_agent import AgentError
from src.agents.data_agent import DataAgent
from src.agents.evaluation_agent import EvaluationAgent, EvaluationInput
from src.agents.llm_client import LLMClient
from src.agents.modeling_agent import ModelingAgent, ModelingInput
from src.agents.planner_agent import PlannerAgent
from src.agents.report_agent import ReportAgent, ReportInput
from src.agents.schemas import ExperimentConfig, ExperimentReport

logger = logging.getLogger(__name__)


def run_experiment(
    config: ExperimentConfig,
    *,
    ohlcv_df: pd.DataFrame | None = None,
    reports_dir: str | Path = "reports",
    llm_client: LLMClient | None = None,
) -> ExperimentReport:
    """Run the full forecasting pipeline as an explicit multi-agent workflow.

    Pipeline:
        PlannerAgent → DataAgent → ModelingAgent → EvaluationAgent → ReportAgent

    Args:
        config: ExperimentConfig specifying ticker, dates, model params, etc.
        ohlcv_df: Optional pre-loaded DataFrame (bypasses DataAgent download).
                  Useful for tests, demos, and offline runs.
        reports_dir: Directory where the JSON report will be written.
        llm_client: Optional LLMClient.  When provided, PlannerAgent uses it to
                    propose feature windows from past experiment history, and
                    EvaluationAgent uses it to add narrative interpretation.
                    When None, both agents operate in deterministic rule-based mode.

    Returns:
        ExperimentReport — fully structured, JSON-serialisable.

    Raises:
        AgentError: if any agent fails all retry attempts.
        RuntimeError: for unexpected orchestration failures.
    """
    t_start = time.monotonic()
    exp_id = config.experiment_id or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    config = config.model_copy(update={"experiment_id": exp_id})

    logger.info("=" * 60)
    logger.info("ORCHESTRATOR  experiment_id=%s", exp_id)
    logger.info("=" * 60)

    if llm_client is not None:
        logger.info("ORCHESTRATOR  LLM enabled: provider=%s model=%s", llm_client.provider, llm_client.model)

    # ── Step 1: PlannerAgent ──────────────────────────────────────────────────
    _log_step(1, "PlannerAgent")
    try:
        plan = PlannerAgent(llm_client=llm_client, reports_dir=reports_dir).run(config)
    except AgentError as exc:
        _log_failure("PlannerAgent", exc)
        raise

    # ── Step 2: DataAgent ─────────────────────────────────────────────────────
    _log_step(2, "DataAgent")
    data_agent = DataAgent()
    try:
        data_summary = data_agent.run_with_df(
            plan,
            ohlcv_df=ohlcv_df,
            data_config_override={
                "start_date": config.start_date,
                "end_date": config.end_date,
                "interval": config.interval,
                "cache_dir": config.cache_dir,
            },
        )
    except AgentError as exc:
        _log_failure("DataAgent", exc)
        raise

    if not data_summary.sufficient_for_backtest:
        raise RuntimeError(
            f"Insufficient data: {data_summary.n_rows} rows. "
            f"Need at least {config.initial_train_days + config.step_days}. "
            "Reduce initial_train_days or extend the date range."
        )

    # data_agent.last_df is the raw DataFrame — not JSON-serialisable, passed directly
    df = data_agent.last_df
    assert df is not None, "DataAgent.last_df must be set after run_with_df()"

    # ── Step 3: ModelingAgent ─────────────────────────────────────────────────
    _log_step(3, "ModelingAgent")
    modeling_agent = ModelingAgent()
    try:
        modeling_result = modeling_agent.run(ModelingInput(plan=plan, ohlcv=df))
    except AgentError as exc:
        _log_failure("ModelingAgent", exc)
        raise

    # ── Step 4: EvaluationAgent ───────────────────────────────────────────────
    _log_step(4, "EvaluationAgent")
    try:
        evaluation = EvaluationAgent(llm_client=llm_client).run(
            EvaluationInput(plan=plan, modeling=modeling_result)
        )
    except AgentError as exc:
        _log_failure("EvaluationAgent", exc)
        raise

    # ── Step 5: ReportAgent ───────────────────────────────────────────────────
    _log_step(5, "ReportAgent")
    try:
        report = ReportAgent(output_dir=reports_dir).run(
            ReportInput(
                config=config,
                plan=plan,
                data_summary=data_summary,
                modeling_result=modeling_result,
                evaluation=evaluation,
            )
        )
    except AgentError as exc:
        _log_failure("ReportAgent", exc)
        raise

    elapsed = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR  complete in %.1fs  verdict=%s", elapsed, evaluation.verdict)
    logger.info("=" * 60)

    return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_step(n: int, name: str) -> None:
    logger.info("── Step %d: %s ──", n, name)


def _log_failure(agent_name: str, exc: AgentError) -> None:
    logger.error(
        "ORCHESTRATOR  step failed: agent=%s  cause=%s: %s",
        agent_name, type(exc.cause).__name__, exc.cause,
    )
