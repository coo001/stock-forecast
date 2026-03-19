"""Entry point for the Samsung Electronics forecasting pipeline.

Wires config/default.yaml through the full agent pipeline
(PlannerAgent → DataAgent → ModelingAgent → EvaluationAgent → ReportAgent)
and optionally activates LLM reasoning in Planner and Evaluation.

Usage::

    # Deterministic baseline (no LLM):
    python run_pipeline.py

    # Synthetic data — no internet or yfinance required:
    python run_pipeline.py --synthetic

    # With OpenAI (set OPENAI_API_KEY first):
    python run_pipeline.py --llm-provider openai

    # With Anthropic:
    python run_pipeline.py --llm-provider anthropic --llm-model claude-haiku-4-5-20251001

    # Force-refresh data (re-download even if cache exists):
    python run_pipeline.py --force-refresh

    # Require cached data (raise if not present):
    python run_pipeline.py --no-download

    # Custom config file:
    python run_pipeline.py --config config/dev.yaml

    # Custom reports directory:
    python run_pipeline.py --reports-dir experiments/

    # Fixed experiment id (for reproducibility):
    python run_pipeline.py --experiment-id exp_baseline_v1

Experiment contract (CLAUDE.md § Forecasting rules):
    Target        : configurable via config → target.kind
    Horizon       : configurable via config → target.horizon
    Validation    : walk-forward, no future leakage
    Model         : LightGBM regressor (baseline)
    Report        : JSON saved to --reports-dir; text summary to stdout
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.agents.llm_client import build_llm_client
from src.agents.orchestrator import run_experiment
from src.startup import check_required, require_yfinance


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO),
        format=log_cfg.get("format", "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Samsung Electronics stock forecasting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config/default.yaml", metavar="PATH",
                   help="YAML config file (default: config/default.yaml)")
    p.add_argument("--reports-dir", default="reports", metavar="DIR",
                   help="Directory for JSON experiment reports (default: reports/)")
    p.add_argument("--experiment-id", default="", metavar="ID",
                   help="Fixed experiment ID for reproducibility (auto-generated if empty)")

    # Data
    data_g = p.add_mutually_exclusive_group()
    data_g.add_argument("--no-download", action="store_true",
                        help="Require cached data; raise if cache is absent")
    data_g.add_argument("--force-refresh", action="store_true",
                        help="Re-download data even if cache exists")
    data_g.add_argument("--synthetic", action="store_true",
                        help="Use synthetic Samsung-like data (no network or yfinance required)")

    # LLM overrides (take precedence over config/default.yaml → llm section)
    llm_g = p.add_argument_group("LLM options (override config/default.yaml → llm)")
    llm_g.add_argument("--llm-provider", default=None,
                       choices=["openai", "anthropic", "none"],
                       help="LLM provider (default: read from config)")
    llm_g.add_argument("--llm-model", default=None, metavar="MODEL",
                       help="Override model name (e.g. gpt-4o-mini)")

    return p.parse_args(argv)


# ── Config → ExperimentConfig mapping ─────────────────────────────────────────

def _build_experiment_config(cfg: dict, args: argparse.Namespace):
    """Translate YAML sections into a typed ExperimentConfig."""
    from src.agents.schemas import ExperimentConfig

    data_cfg = cfg.get("data", {})
    target_cfg = cfg.get("target", {})
    feat_cfg = cfg.get("features", {})
    bt_cfg = cfg.get("backtest", {})
    lgbm_cfg = cfg.get("lgbm", {})

    return ExperimentConfig(
        # data
        ticker=data_cfg.get("ticker", "005930.KS"),
        start_date=data_cfg.get("start_date", "2015-01-01"),
        end_date=data_cfg.get("end_date"),
        interval=data_cfg.get("interval", "1d"),
        cache_dir=data_cfg.get("cache_dir", "data/raw"),
        # target
        target_kind=target_cfg.get("kind", "next_day_log_return"),
        horizon=target_cfg.get("horizon", 1),
        # features
        return_windows=feat_cfg.get("return_windows", [1, 5, 10, 20]),
        ma_windows=feat_cfg.get("ma_windows", [5, 20, 60]),
        rsi_window=feat_cfg.get("rsi_window", 14),
        atr_window=feat_cfg.get("atr_window", 14),
        volume_ma_window=feat_cfg.get("volume_ma_window", 20),
        # backtest
        initial_train_days=bt_cfg.get("initial_train_days", 504),
        step_days=bt_cfg.get("step_days", 63),
        min_train_days=bt_cfg.get("min_train_days", 252),
        # model
        lgbm_params=lgbm_cfg,
        # runtime
        experiment_id=args.experiment_id,
    )


# ── Data pre-loading (handles --no-download / --force-refresh) ─────────────────

def _load_data(cfg: dict, args: argparse.Namespace):
    """Return (ohlcv_df | None).

    - Default: return None → DataAgent will download/cache automatically.
    - --no-download: load from cache only; raise if cache file is absent.
    - --force-refresh: download, then return None so DataAgent re-downloads.
      (Setting force_download=True in DataAgent is not yet exposed via config,
       so we pre-load here and pass the DataFrame directly instead.)
    - --synthetic: generate Samsung-like GBM data; no network or yfinance needed.
    """
    from src.data.loader import load_ohlcv, load_ohlcv_from_csv, _cache_path
    from src.data.schema import DataConfig
    from src.data.synthetic import make_samsung_ohlcv, recommended_n

    data_cfg_section = cfg.get("data", {})
    data_config = DataConfig(**data_cfg_section)

    if getattr(args, "synthetic", False):
        bt_cfg = cfg.get("backtest", {})
        n = recommended_n(
            initial_train_days=bt_cfg.get("initial_train_days", 504),
            step_days=bt_cfg.get("step_days", 63),
        )
        logging.getLogger(__name__).info("--synthetic: generating %d rows of synthetic data", n)
        return make_samsung_ohlcv(n=n)

    if args.no_download:
        cache = _cache_path(data_config)
        if not cache.exists():
            raise FileNotFoundError(
                f"--no-download specified but cache not found: {cache}\n"
                "Run without --no-download first to download and cache the data."
            )
        logging.getLogger(__name__).info("--no-download: loading from %s", cache)
        return load_ohlcv_from_csv(cache)

    if args.force_refresh:
        logging.getLogger(__name__).info("--force-refresh: re-downloading %s", data_config.ticker)
        return load_ohlcv(data_config, force_download=True)

    # Default: let DataAgent handle it (download or use cache automatically)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Preflight: fail fast with clear instructions if core deps are missing
    check_required()

    args = _parse_args(argv)

    # Require yfinance only when we will actually download data
    if not getattr(args, "synthetic", False) and not args.no_download:
        require_yfinance()

    cfg = _load_config(args.config)
    _setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Config: %s", args.config)

    # ── LLM client ────────────────────────────────────────────────────────────
    llm_client = build_llm_client(
        cfg.get("llm", {}),
        provider_override=args.llm_provider,
        model_override=args.llm_model,
    )

    # ── Experiment config ──────────────────────────────────────────────────────
    experiment_config = _build_experiment_config(cfg, args)

    # ── Optional pre-load (--no-download / --force-refresh) ───────────────────
    try:
        ohlcv_df = _load_data(cfg, args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # ── Run agent pipeline ─────────────────────────────────────────────────────
    try:
        report = run_experiment(
            experiment_config,
            ohlcv_df=ohlcv_df,
            reports_dir=args.reports_dir,
            llm_client=llm_client,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc)
        return 1

    # ── Print report ───────────────────────────────────────────────────────────
    print()
    print(report.to_text())

    report_path = Path(args.reports_dir) / f"{report.experiment_id}.json"
    logger.info("Report saved: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
