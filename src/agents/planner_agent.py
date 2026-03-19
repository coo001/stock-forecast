"""PlannerAgent — validates experiment config and emits an execution plan.

Responsibilities:
- Validate ExperimentConfig fields (date format, horizon, target kind)
- Resolve defaults (experiment_id, lgbm_params)
- Emit warnings for potentially problematic settings
- Optionally consult an LLM to propose improved feature windows based on
  the history of past experiment reports

Two-pass design:
    Pass 1 (deterministic, always runs):
        Validates config, assigns experiment_id, resolves LGBM defaults.
        Produces a complete, executable ExecutionPlan.

    Pass 2 (LLM, optional):
        Reads past ExperimentReport JSON files from ``reports_dir``.
        Sends compact summaries to the LLM.
        LLM returns: reasoning (prose), updated feature windows.
        Any values outside permitted bounds are clamped (no blind trust).
        On any LLM failure: logs a warning, keeps the deterministic plan.

This makes PlannerAgent the natural learning loop: as experiment history
accumulates, the LLM can suggest increasingly refined feature configs.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.agents.base_agent import BaseAgent
from src.agents.schemas import ExecutionPlan, ExperimentConfig
from src.models.lgbm_model import _default_params as lgbm_defaults

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

_VALID_TARGET_KINDS = {
    "next_day_log_return",
    "next_day_direction",
    "next_5d_log_return",
}

# ── Feature window bounds (guard against LLM hallucinations) ──────────────────
_BOUNDS = {
    "return_windows_min": 1, "return_windows_max": 120,
    "ma_windows_min": 5,     "ma_windows_max": 200,
    "rsi_window_min": 7,     "rsi_window_max": 30,
    "atr_window_min": 7,     "atr_window_max": 30,
    "volume_ma_window_min": 5, "volume_ma_window_max": 60,
}

_MAX_PAST_REPORTS = 5   # cap context length sent to LLM

_SYSTEM_PROMPT = """\
You are a quantitative research agent designing feature engineering experiments \
for a Samsung Electronics stock forecasting pipeline.

Respond with ONLY a valid JSON object — no markdown, no explanation outside JSON:
{
  "reasoning": "<1-2 sentences explaining feature choices>",
  "return_windows": [<int>, ...],
  "ma_windows": [<int>, ...],
  "rsi_window": <int>,
  "atr_window": <int>,
  "volume_ma_window": <int>,
  "lgbm_overrides": {}
}

Constraints:
  return_windows : integers in [1, 120], 2-6 values
  ma_windows     : integers in [5, 200], 2-5 values
  rsi_window     : integer in [7, 30]
  atr_window     : integer in [7, 30]
  volume_ma_window: integer in [5, 60]
  lgbm_overrides : dict of LightGBM param overrides (may be empty)
"""


class PlannerAgent(BaseAgent[ExperimentConfig, ExecutionPlan]):
    """Validates config, optionally uses LLM to suggest improved feature windows."""

    name = "PlannerAgent"

    def __init__(
        self,
        llm_client: "LLMClient | None" = None,
        reports_dir: str | Path = "reports",
    ) -> None:
        super().__init__()
        self.llm_client = llm_client
        self.reports_dir = Path(reports_dir)
        self.timeout_seconds = 60.0 if llm_client is not None else 10.0

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def _run(self, config: ExperimentConfig) -> ExecutionPlan:
        plan = self._deterministic_plan(config)

        if self.llm_client is not None:
            plan = self._llm_suggest(plan, config)

        return plan

    # ── Pass 1: deterministic validation and planning ─────────────────────────

    def _deterministic_plan(self, config: ExperimentConfig) -> ExecutionPlan:
        warnings: list[str] = []
        experiment_id = config.experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        logger.info("[PlannerAgent] experiment_id=%s  ticker=%s", experiment_id, config.ticker)

        if config.target_kind not in _VALID_TARGET_KINDS:
            raise ValueError(
                f"Unknown target_kind '{config.target_kind}'. "
                f"Valid: {sorted(_VALID_TARGET_KINDS)}"
            )

        if config.target_kind == "next_5d_log_return" and config.horizon != 5:
            warnings.append(
                f"target_kind='next_5d_log_return' but horizon={config.horizon}; "
                "horizon will be overridden to 5 by indicators.make_target."
            )

        if config.initial_train_days < config.min_train_days:
            raise ValueError(
                f"initial_train_days ({config.initial_train_days}) < "
                f"min_train_days ({config.min_train_days})"
            )

        if config.step_days < 5:
            warnings.append(f"step_days={config.step_days} is very small; expect many folds.")

        resolved_lgbm = {**lgbm_defaults(), **config.lgbm_params}

        plan = ExecutionPlan(
            experiment_id=experiment_id,
            ticker=config.ticker,
            target_kind=config.target_kind,
            horizon=config.horizon,
            steps=["DataAgent", "ModelingAgent", "EvaluationAgent", "ReportAgent"],
            feature_config={
                "return_windows": config.return_windows,
                "ma_windows": config.ma_windows,
                "rsi_window": config.rsi_window,
                "atr_window": config.atr_window,
                "volume_ma_window": config.volume_ma_window,
            },
            backtest_config={
                "initial_train_days": config.initial_train_days,
                "step_days": config.step_days,
                "min_train_days": config.min_train_days,
            },
            lgbm_config=resolved_lgbm,
            warnings=warnings,
            created_at=datetime.utcnow().isoformat(),
        )

        for w in warnings:
            logger.warning("[PlannerAgent] %s", w)
        logger.info("[PlannerAgent] plan ready: %d steps, %d warnings", len(plan.steps), len(plan.warnings))
        return plan

    # ── Pass 2: optional LLM feature suggestion ───────────────────────────────

    def _llm_suggest(self, plan: ExecutionPlan, config: ExperimentConfig) -> ExecutionPlan:
        """Ask the LLM to propose better feature windows based on past runs.

        On any failure, returns the deterministic plan unchanged.
        """
        try:
            past = _load_past_summaries(self.reports_dir, limit=_MAX_PAST_REPORTS)
            if not past:
                logger.info("[PlannerAgent] No past reports found; skipping LLM suggestion.")
                return plan

            user = _build_planner_prompt(plan, past)
            raw = self.llm_client.chat_json(_SYSTEM_PROMPT, user)  # type: ignore[union-attr]
            return _apply_llm_suggestions(plan, raw)

        except Exception as exc:
            logger.warning(
                "[PlannerAgent] LLM suggestion failed (%s: %s); using deterministic plan.",
                type(exc).__name__, exc,
            )
            return plan


# ── Prompt and response helpers ────────────────────────────────────────────────

def _load_past_summaries(reports_dir: Path, limit: int) -> list[dict]:
    """Load and compact the most recent ExperimentReport JSON files."""
    if not reports_dir.exists():
        return []

    files = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ev = data.get("evaluation", {})
            mr = data.get("modeling_result", {})
            summaries.append({
                "experiment_id": data.get("experiment_id", "?"),
                "generated_at": data.get("generated_at", "?")[:10],
                "target_kind": data.get("config", {}).get("target_kind", "?"),
                "feature_config": data.get("plan", {}).get("feature_config", {}),
                "n_folds": mr.get("n_folds", 0),
                "n_features": mr.get("n_features", 0),
                "top_features": mr.get("top_features", []),
                "directional_accuracy": round(ev.get("directional_accuracy_mean", 0), 4),
                "sharpe": round(ev.get("sharpe_mean", 0), 3),
                "ic": round(ev.get("aggregate_metrics", {}).get("ic", 0), 4),
                "verdict": ev.get("verdict", "?"),
            })
        except Exception as exc:
            logger.debug("[PlannerAgent] skipping report %s: %s", f.name, exc)

    return summaries


def _build_planner_prompt(plan: ExecutionPlan, past: list[dict]) -> str:
    past_text = "\n".join(
        f"  Exp {s['experiment_id']} ({s['generated_at']}): "
        f"verdict={s['verdict']} DA={s['directional_accuracy']} "
        f"Sharpe={s['sharpe']} IC={s['ic']} "
        f"top_features={s['top_features'][:3]} "
        f"feature_config={s['feature_config']}"
        for s in past
    )
    return (
        f"Ticker: {plan.ticker}  Target: {plan.target_kind}  Horizon: {plan.horizon}d\n"
        f"\nCurrent feature config:\n  {plan.feature_config}\n"
        f"\nPast experiments (most recent first):\n{past_text}\n"
        f"\nSuggest improved feature windows for the next experiment."
    )


def _apply_llm_suggestions(plan: ExecutionPlan, raw: dict) -> ExecutionPlan:
    """Merge LLM-suggested feature config into the plan with bound-clamping."""
    fc = dict(plan.feature_config)  # start from deterministic plan
    lc = dict(plan.lgbm_config)

    def _clamp_list(key: str, lo: int, hi: int) -> list[int] | None:
        val = raw.get(key)
        if not isinstance(val, list) or not val:
            return None
        clamped = sorted({max(lo, min(hi, int(v))) for v in val if isinstance(v, (int, float))})
        return clamped or None

    def _clamp_int(key: str, lo: int, hi: int) -> int | None:
        val = raw.get(key)
        if not isinstance(val, (int, float)):
            return None
        return max(lo, min(hi, int(val)))

    rw = _clamp_list("return_windows", _BOUNDS["return_windows_min"], _BOUNDS["return_windows_max"])
    maw = _clamp_list("ma_windows", _BOUNDS["ma_windows_min"], _BOUNDS["ma_windows_max"])
    rsi = _clamp_int("rsi_window", _BOUNDS["rsi_window_min"], _BOUNDS["rsi_window_max"])
    atr = _clamp_int("atr_window", _BOUNDS["atr_window_min"], _BOUNDS["atr_window_max"])
    vol = _clamp_int("volume_ma_window", _BOUNDS["volume_ma_window_min"], _BOUNDS["volume_ma_window_max"])

    if rw:  fc["return_windows"] = rw
    if maw: fc["ma_windows"] = maw
    if rsi: fc["rsi_window"] = rsi
    if atr: fc["atr_window"] = atr
    if vol: fc["volume_ma_window"] = vol

    lgbm_overrides = raw.get("lgbm_overrides")
    if isinstance(lgbm_overrides, dict):
        lc.update(lgbm_overrides)

    reasoning = str(raw.get("reasoning", ""))
    logger.info("[PlannerAgent] LLM updated feature_config=%s reasoning=%r", fc, reasoning[:80])

    return plan.model_copy(update={
        "feature_config": fc,
        "lgbm_config": lc,
        "llm_reasoning": reasoning,
    })
