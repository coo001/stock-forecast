"""AutoGen multi-agent orchestration layer for Samsung forecasting.

Public API::

    from src.agents.orchestrator import run_experiment
    from src.agents.schemas import ExperimentConfig, ExperimentReport

    report = run_experiment(ExperimentConfig(ticker="005930.KS"))
    print(report.to_text())

Pipeline (explicit, deterministic):
    PlannerAgent → DataAgent → ModelingAgent → EvaluationAgent → ReportAgent

Each agent:
- Accepts a typed Pydantic input
- Returns a typed Pydantic output
- Has retry (max_retries=2), timeout, and structured logging built-in
- Exposes ``.as_autogen_tool()`` for AutoGen ConversableAgent registration (v2)

AutoGen v2 migration:
    The orchestrator in ``orchestrator.py`` can be swapped for a GroupChat
    or Swarm coordinator. Every agent's ``_run()`` implementation is reusable.
"""

from src.agents.orchestrator import run_experiment
from src.agents.schemas import ExperimentConfig, ExperimentReport

__all__ = ["run_experiment", "ExperimentConfig", "ExperimentReport"]
