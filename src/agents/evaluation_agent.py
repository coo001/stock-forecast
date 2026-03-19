"""EvaluationAgent — interprets walk-forward metrics and flags anomalies.

Responsibilities:
- Aggregate per-fold metrics from ModelingResult  (always deterministic)
- Apply heuristic thresholds to classify model quality  (always deterministic)
- Optionally call an LLM to produce a narrative interpretation and richer
  experiment recommendations

Two-pass design:
    Pass 1 (rule-based, always runs):
        Computes aggregates, flags anomalies, assigns verdict.
        Produces a complete EvaluationReport without any LLM dependency.

    Pass 2 (LLM, optional):
        Sends compact structured metrics to the LLM.
        LLM returns: interpretation (prose), verdict (may confirm/upgrade),
        llm_recommendations (list).
        On any LLM failure: logs a warning, keeps the rule-based report.

This matches the design principle: "keep orchestration deterministic even
after LLM integration" — the LLM enriches, it does not control flow.
"""
from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING

from src.agents.base_agent import BaseAgent
from src.agents.schemas import EvaluationReport, ExecutionPlan, ModelingResult

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Heuristic thresholds ──────────────────────────────────────────────────────
_DA_RANDOM = 0.50
_DA_MARGINAL = 0.52
_DA_ACCEPTABLE = 0.54
_DA_STRONG = 0.58

_SHARPE_POOR = 0.0
_SHARPE_MARGINAL = 0.3
_SHARPE_ACCEPTABLE = 0.5
_SHARPE_STRONG = 0.8

_FOLD_STDEV_HIGH = 0.05

_VALID_VERDICTS = {"poor", "marginal", "acceptable", "strong"}

_SYSTEM_PROMPT = """\
You are a quantitative analyst reviewing a stock return-forecasting backtest.

Respond with ONLY a valid JSON object matching this exact schema — no markdown, \
no explanation outside the JSON:
{
  "interpretation": "<2-3 sentence plain-English analysis of model quality>",
  "verdict": "<one of: poor | marginal | acceptable | strong>",
  "llm_recommendations": [
    "<specific, actionable recommendation>",
    "<specific, actionable recommendation>"
  ]
}

Guidelines for verdict:
  poor       — directional accuracy near 50%, Sharpe < 0
  marginal   — slight edge but inconsistent (DA 52-54% or Sharpe 0.3-0.5)
  acceptable — consistent edge (DA 54-58% and Sharpe 0.5-0.8)
  strong     — clear signal (DA > 58% and Sharpe > 0.8)
"""


class EvaluationInput:
    def __init__(self, plan: ExecutionPlan, modeling: ModelingResult):
        self.plan = plan
        self.modeling = modeling


class EvaluationAgent(BaseAgent[EvaluationInput, EvaluationReport]):
    """Interprets walk-forward results; optionally enriches with LLM analysis."""

    name = "EvaluationAgent"

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        super().__init__()
        self.llm_client = llm_client
        # LLM calls may take up to 30s; increase timeout when enabled
        self.timeout_seconds = 60.0 if llm_client is not None else 10.0

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def _run(self, input: EvaluationInput) -> EvaluationReport:
        report = self._rule_based(input.plan, input.modeling)

        if self.llm_client is not None:
            report = self._llm_enrich(report, input.modeling)

        return report

    # ── Pass 1: deterministic rule-based evaluation ────────────────────────────

    def _rule_based(self, plan: ExecutionPlan, m: ModelingResult) -> EvaluationReport:
        if not m.folds:
            raise ValueError("ModelingResult has no folds to evaluate.")

        da_values = [f.directional_accuracy for f in m.folds]
        sharpe_values = [f.sharpe for f in m.folds]

        da_mean = statistics.mean(da_values)
        sharpe_mean = statistics.mean(sharpe_values)

        agg = {
            "mae": statistics.mean(f.mae for f in m.folds),
            "rmse": statistics.mean(f.rmse for f in m.folds),
            "directional_accuracy": da_mean,
            "sharpe": sharpe_mean,
            "ic": statistics.mean(f.ic for f in m.folds),
        }

        flags: list[str] = []
        recommendations: list[str] = []

        if len(da_values) >= 2:
            da_std = statistics.stdev(da_values)
            if da_std > _FOLD_STDEV_HIGH:
                flags.append(
                    f"High fold-to-fold DA variance (stdev={da_std:.3f}). Model may be unstable."
                )

        if da_mean <= _DA_RANDOM + 0.01:
            flags.append(
                f"Directional accuracy ({da_mean:.3f}) is near random (0.50). "
                "Model likely has no predictive signal."
            )

        if sharpe_mean < _SHARPE_POOR:
            flags.append(
                f"Mean Sharpe ({sharpe_mean:.2f}) is negative. "
                "Long/short strategy loses money on average."
            )

        if m.n_oos_observations < 100:
            flags.append(
                f"Only {m.n_oos_observations} OOS observations. "
                "Results may not be statistically reliable."
            )

        if da_mean < _DA_ACCEPTABLE:
            recommendations.append("Try adding macro/news features (FX rate, VIX, sector index).")
            recommendations.append("Consider longer return windows (60d, 120d) or calendar features.")

        if sharpe_mean < _SHARPE_ACCEPTABLE:
            recommendations.append("Explore asymmetric loss functions (e.g. profit-weighted RMSE).")

        if m.top_features:
            recommendations.append(
                f"Investigate top features: {', '.join(m.top_features[:3])}. "
                "Consider interaction terms or lag variations."
            )

        if da_mean >= _DA_STRONG and sharpe_mean >= _SHARPE_STRONG:
            verdict = "strong"
        elif da_mean >= _DA_ACCEPTABLE and sharpe_mean >= _SHARPE_ACCEPTABLE:
            verdict = "acceptable"
        elif da_mean >= _DA_MARGINAL or sharpe_mean >= _SHARPE_MARGINAL:
            verdict = "marginal"
        else:
            verdict = "poor"

        logger.info("[EvaluationAgent] rule-based: verdict=%s DA=%.3f Sharpe=%.2f", verdict, da_mean, sharpe_mean)

        return EvaluationReport(
            experiment_id=plan.experiment_id,
            aggregate_metrics=agg,
            directional_accuracy_mean=da_mean,
            sharpe_mean=sharpe_mean,
            verdict=verdict,
            flags=flags,
            recommendations=recommendations,
        )

    # ── Pass 2: optional LLM enrichment ───────────────────────────────────────

    def _llm_enrich(self, report: EvaluationReport, m: ModelingResult) -> EvaluationReport:
        """Call LLM to add narrative interpretation and improved recommendations.

        On any failure, returns the original rule-based report unchanged.
        """
        try:
            user = _build_evaluation_prompt(report, m)
            raw = self.llm_client.chat_json(_SYSTEM_PROMPT, user)  # type: ignore[union-attr]
            return _apply_llm_output(report, raw)
        except Exception as exc:
            logger.warning(
                "[EvaluationAgent] LLM enrichment failed (%s: %s); using rule-based report.",
                type(exc).__name__, exc,
            )
            return report


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_evaluation_prompt(report: EvaluationReport, m: ModelingResult) -> str:
    fold_das = [round(f.directional_accuracy, 3) for f in m.folds]
    fold_sharpes = [round(f.sharpe, 2) for f in m.folds]
    agg = report.aggregate_metrics

    return (
        f"Experiment: {report.experiment_id}\n"
        f"Model: {m.model_name}  Target: {m.target_kind}\n"
        f"Walk-forward: {m.n_folds} folds, {m.n_oos_observations} OOS observations\n"
        f"\nAggregate metrics:\n"
        f"  Directional Accuracy : {agg['directional_accuracy']:.4f}  [random=0.50]\n"
        f"  Sharpe (annualised)  : {agg['sharpe']:.3f}\n"
        f"  MAE                  : {agg['mae']:.6f}\n"
        f"  IC (Spearman)        : {agg['ic']:.4f}\n"
        f"\nRule-based verdict: {report.verdict}\n"
        f"Detected flags: {report.flags or 'none'}\n"
        f"Top features: {m.top_features or 'unavailable'}\n"
        f"\nPer-fold directional accuracies: {fold_das}\n"
        f"Per-fold Sharpe ratios: {fold_sharpes}\n"
    )


def _apply_llm_output(report: EvaluationReport, raw: dict) -> EvaluationReport:
    """Merge LLM output into the rule-based report.

    The LLM may confirm or change the verdict; we only accept it if the
    value is in _VALID_VERDICTS (guards against hallucinated values).
    """
    llm_verdict = str(raw.get("verdict", report.verdict)).lower()
    if llm_verdict not in _VALID_VERDICTS:
        logger.warning("[EvaluationAgent] LLM returned invalid verdict %r; keeping rule-based.", llm_verdict)
        llm_verdict = report.verdict

    llm_recommendations = raw.get("llm_recommendations", [])
    if not isinstance(llm_recommendations, list):
        llm_recommendations = []

    updated = report.model_copy(update={
        "verdict": llm_verdict,
        "llm_interpretation": str(raw.get("interpretation", "")),
        "llm_recommendations": [str(r) for r in llm_recommendations],
    })

    logger.info(
        "[EvaluationAgent] LLM enriched: verdict=%s (was=%s)  recs=%d",
        llm_verdict, report.verdict, len(llm_recommendations),
    )
    return updated
