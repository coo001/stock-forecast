"""ReportAgent — assembles and persists the final experiment report.

Responsibilities:
- Combine outputs from all upstream agents into ExperimentReport
- Write JSON report to disk (reports/ directory)
- Print a human-readable summary to stdout
- Return structured ExperimentReport for programmatic use

No forecasting logic lives here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.schemas import (
    DataSummary,
    EvaluationReport,
    ExecutionPlan,
    ExperimentConfig,
    ExperimentReport,
    ModelingResult,
)

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")


@dataclass
class ReportInput:
    config: ExperimentConfig
    plan: ExecutionPlan
    data_summary: DataSummary
    modeling_result: ModelingResult
    evaluation: EvaluationReport


class ReportAgent(BaseAgent[ReportInput, ExperimentReport]):
    """Assembles all agent outputs into a final structured report."""

    name = "ReportAgent"
    timeout_seconds = 30.0

    def __init__(self, output_dir: str | Path = _REPORTS_DIR) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def _run(self, input: ReportInput) -> ExperimentReport:
        report = ExperimentReport(
            experiment_id=input.plan.experiment_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            config=input.config,
            plan=input.plan,
            data_summary=input.data_summary,
            modeling_result=input.modeling_result,
            evaluation=input.evaluation,
        )

        self._save_json(report)
        logger.info("[ReportAgent] report assembled for experiment %s", report.experiment_id)
        return report

    def _save_json(self, report: ExperimentReport) -> Path:
        """Persist report as JSON. Creates output_dir if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{report.experiment_id}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("[ReportAgent] report saved → %s", path)
        return path
