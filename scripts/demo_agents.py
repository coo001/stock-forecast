"""Demo: end-to-end multi-agent forecasting pipeline.

This script shows the full agent workflow without requiring a live internet
connection.  By default it generates synthetic Samsung-like price data so
the demo is reproducible and self-contained.

Usage::

    # Synthetic data (always works, no internet required):
    python scripts/demo_agents.py

    # Real data from Yahoo Finance (requires internet + yfinance):
    python scripts/demo_agents.py --real-data

    # Custom date range with real data:
    python scripts/demo_agents.py --real-data --start 2018-01-01 --end 2023-12-31

    # Fast run with a smaller dataset (fewer folds):
    python scripts/demo_agents.py --fast

Options:
    --real-data       Download live data from Yahoo Finance instead of synthetic
    --start DATE      Start date for real-data download (default: 2015-01-01)
    --end DATE        End date for real-data download (default: today)
    --fast            Reduce initial_train_days and step_days for a quick test
    --output-dir DIR  Directory for JSON reports (default: reports/)
    --log-level LVL   Logging level: DEBUG | INFO | WARNING (default: INFO)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Add project root to path so src/ is importable when run as a script ───────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.orchestrator import run_experiment
from src.agents.schemas import ExperimentConfig
from src.data.synthetic import make_samsung_ohlcv, recommended_n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Samsung forecasting agent demo")
    p.add_argument("--real-data", action="store_true", help="Use Yahoo Finance data")
    p.add_argument("--start", default="2015-01-01", help="Start date (real-data mode)")
    p.add_argument("--end", default=None, help="End date (real-data mode, default=today)")
    p.add_argument("--fast", action="store_true", help="Smaller windows for a quick run")
    p.add_argument("--output-dir", default="reports", help="Report output directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("demo")

    # ── Build ExperimentConfig ─────────────────────────────────────────────────
    if args.fast:
        initial_train_days, step_days, min_train_days = 252, 42, 126
    else:
        initial_train_days, step_days, min_train_days = 504, 63, 252

    config = ExperimentConfig(
        ticker="005930.KS",
        start_date=args.start,
        end_date=args.end,
        target_kind="next_day_log_return",
        horizon=1,
        initial_train_days=initial_train_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )

    # ── Prepare data ───────────────────────────────────────────────────────────
    if args.real_data:
        logger.info("Mode: REAL DATA  (ticker=%s, start=%s)", config.ticker, config.start_date)
        ohlcv_df = None   # DataAgent will download it
    else:
        n_rows = recommended_n(initial_train_days, step_days)
        logger.info("Mode: SYNTHETIC DATA  (%d rows)", n_rows)
        ohlcv_df = make_samsung_ohlcv(n=n_rows)

    # ── Run the agent pipeline ─────────────────────────────────────────────────
    logger.info("Starting multi-agent pipeline …")
    try:
        report = run_experiment(
            config,
            ohlcv_df=ohlcv_df,
            reports_dir=args.output_dir,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc)
        return 1

    # ── Print human-readable summary ───────────────────────────────────────────
    print()
    print(report.to_text())

    # ── Print path to JSON report ──────────────────────────────────────────────
    report_path = Path(args.output_dir) / f"{report.experiment_id}.json"
    if report_path.exists():
        print(f"\nFull JSON report: {report_path}")
    else:
        # Fallback: print abbreviated JSON to stdout
        print("\nReport JSON (abbreviated):")
        data = json.loads(report.model_dump_json())
        data.pop("plan", None)  # remove verbose plan from console output
        print(json.dumps(data, indent=2)[:2000] + "\n…")

    return 0


if __name__ == "__main__":
    sys.exit(main())
