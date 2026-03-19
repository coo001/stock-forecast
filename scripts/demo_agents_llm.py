"""Demo: multi-agent pipeline with optional LLM reasoning.

Shows three modes:
  --mock       Use a MockLLMClient (no API key, always works)
  --openai     Use OpenAI (requires OPENAI_API_KEY env var)
  --anthropic  Use Anthropic (requires ANTHROPIC_API_KEY env var)
  (default)    No LLM — same as original demo_agents.py

Usage::

    # Mock LLM (offline, shows LLM output fields populated with dummy data):
    python scripts/demo_agents_llm.py --mock

    # Real OpenAI:
    python scripts/demo_agents_llm.py --openai --model gpt-4o-mini

    # Real Anthropic:
    python scripts/demo_agents_llm.py --anthropic --model claude-haiku-4-5-20251001

    # Second run: PlannerAgent will find the first report and ask LLM to improve features:
    python scripts/demo_agents_llm.py --mock
    python scripts/demo_agents_llm.py --mock   # LLM sees history now

    # With real Samsung data:
    python scripts/demo_agents_llm.py --mock --real-data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.llm_client import LLMClient
from src.agents.orchestrator import run_experiment
from src.agents.schemas import ExperimentConfig
from src.data.synthetic import make_samsung_ohlcv, recommended_n


# ── Mock responses ─────────────────────────────────────────────────────────────
# Two responses: first for EvaluationAgent, second for PlannerAgent.
# (PlannerAgent calls LLM only when past reports exist, so on first run
#  only EvaluationAgent calls the LLM.)

_MOCK_EVALUATION_RESPONSE = {
    "interpretation": (
        "The model achieves near-random directional accuracy (~50%) across most folds, "
        "suggesting that the current technical indicator set lacks sufficient predictive "
        "signal for Samsung Electronics' next-day returns. The positive mean Sharpe "
        "is driven by a few high-performing folds and is not consistent."
    ),
    "verdict": "marginal",
    "llm_recommendations": [
        "Add KRX market-wide return as a feature to capture systematic momentum.",
        "Include a 5-day realised volatility feature; ATR alone may miss volatility regimes.",
        "Try a 120-day moving average ratio to capture long-term mean reversion.",
        "Consider adding day-of-week and month-of-year calendar features.",
        "Evaluate whether a classification objective (direction) outperforms regression.",
    ],
}

_MOCK_PLANNER_RESPONSE = {
    "reasoning": (
        "Prior experiments show ATR and short-term MA ratios as top features. "
        "Extending return windows to include 60d and 120d may capture mean reversion. "
        "Adding a longer RSI period (21d) may reduce noise."
    ),
    "return_windows": [1, 5, 10, 20, 60, 120],
    "ma_windows": [5, 20, 60, 120],
    "rsi_window": 21,
    "atr_window": 14,
    "volume_ma_window": 20,
    "lgbm_overrides": {"num_leaves": 15, "min_child_samples": 30},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Samsung forecasting agent demo with LLM")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Use MockLLMClient (no API key)")
    mode.add_argument("--openai", action="store_true", help="Use OpenAI (OPENAI_API_KEY)")
    mode.add_argument("--anthropic", action="store_true", help="Use Anthropic (ANTHROPIC_API_KEY)")
    p.add_argument("--model", default=None, help="Override model name")
    p.add_argument("--real-data", action="store_true", help="Download live data")
    p.add_argument("--fast", action="store_true", help="Smaller windows for quick run")
    p.add_argument("--output-dir", default="reports", help="Report directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)



def build_llm_client(args: argparse.Namespace) -> LLMClient | None:
    """Build the LLM client based on CLI flags."""
    if args.mock:
        # Two responses: evaluation first (always), planner second (only when history exists)
        client = LLMClient.mock([_MOCK_EVALUATION_RESPONSE, _MOCK_PLANNER_RESPONSE])
        logging.getLogger("demo").info("LLM mode: MOCK (no API calls)")
        return client

    if args.openai:
        model = args.model or "gpt-4o-mini"
        logging.getLogger("demo").info("LLM mode: OpenAI  model=%s", model)
        return LLMClient(
            provider="openai",
            model=model,
            api_key_env="OPENAI_API_KEY",
        )

    if args.anthropic:
        model = args.model or "claude-haiku-4-5-20251001"
        logging.getLogger("demo").info("LLM mode: Anthropic  model=%s", model)
        return LLMClient(
            provider="anthropic",
            model=model,
            api_key_env="ANTHROPIC_API_KEY",
        )

    logging.getLogger("demo").info("LLM mode: NONE (deterministic only)")
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("demo")

    llm_client = build_llm_client(args)

    if args.fast:
        initial_train_days, step_days, min_train_days = 252, 42, 126
    else:
        initial_train_days, step_days, min_train_days = 504, 63, 252

    config = ExperimentConfig(
        ticker="005930.KS",
        target_kind="next_day_log_return",
        horizon=1,
        initial_train_days=initial_train_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )

    if args.real_data:
        logger.info("Mode: REAL DATA")
        ohlcv_df = None
    else:
        n_rows = recommended_n(initial_train_days, step_days)
        logger.info("Mode: SYNTHETIC DATA  (%d rows)", n_rows)
        ohlcv_df = make_samsung_ohlcv(n=n_rows)

    try:
        report = run_experiment(
            config,
            ohlcv_df=ohlcv_df,
            reports_dir=args.output_dir,
            llm_client=llm_client,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc)
        return 1

    print()
    print(report.to_text())

    report_path = Path(args.output_dir) / f"{report.experiment_id}.json"
    if report_path.exists():
        print(f"\nFull JSON report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
