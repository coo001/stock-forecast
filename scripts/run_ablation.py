"""Ablation study comparing external data variants and naive baselines.

Runs walk-forward evaluation for four external-data configurations and three
naive baselines, then prints a comparison table and saves a compact JSON
summary to ``reports/ablation_summary_{timestamp}.json``.

No FRED / ECOS API keys are required — only market-source series (yfinance).

Usage::

    python scripts/run_ablation.py                   # live yfinance data
    python scripts/run_ablation.py --synthetic       # no network required
    python scripts/run_ablation.py --no-download     # use existing cache

Ablation variants:
    internal_only           — technical indicators only
    internal_kospi          — + KOSPI log return (ext_kospi)
    internal_kospi_usdkrw   — + USD/KRW log return (ext_usdkrw)
    internal_all_market     — + VIX level from yfinance (ext_vix)

Naive baselines:
    zero_predictor          — always predicts 0
    prev_return             — predicts log_ret_1d
    ridge                   — Ridge regression (alpha=1)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── make repo root importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.backtest.walk_forward import walk_forward
from src.data.loader import load_ohlcv, load_ohlcv_from_csv, _cache_path
from src.data.schema import DataConfig
from src.data.synthetic import make_samsung_ohlcv, recommended_n
from src.features.external_merge import ExternalSeriesConfig, merge_external_features
from src.features.pipeline import build_feature_matrix, feature_columns
from src.models.baselines import PrevReturnPredictor, RidgeForecaster, ZeroPredictor
from src.models.lgbm_model import LGBMForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"

_BACKTEST_DEFAULTS = dict(initial_train_days=504, step_days=63, min_train_days=252)

_LGBM_DEFAULTS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
)

_EXTERNAL_VARIANTS: dict[str, list[ExternalSeriesConfig]] = {
    "internal_only": [],
    "internal_kospi": [
        ExternalSeriesConfig(name="kospi", source="market", symbol="^KS11",
                             lag_days=1, feature_type="log_return"),
    ],
    "internal_kospi_usdkrw": [
        ExternalSeriesConfig(name="kospi", source="market", symbol="^KS11",
                             lag_days=1, feature_type="log_return"),
        ExternalSeriesConfig(name="usdkrw", source="market", symbol="USDKRW=X",
                             lag_days=1, feature_type="log_return"),
    ],
    "internal_all_market": [
        ExternalSeriesConfig(name="kospi", source="market", symbol="^KS11",
                             lag_days=1, feature_type="log_return"),
        ExternalSeriesConfig(name="usdkrw", source="market", symbol="USDKRW=X",
                             lag_days=1, feature_type="log_return"),
        ExternalSeriesConfig(name="vix", source="market", symbol="^VIX",
                             lag_days=1, feature_type="level"),
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_ohlcv(args: argparse.Namespace, cfg: dict) -> pd.DataFrame:
    data_cfg_section = cfg.get("data", {})
    bt_cfg = cfg.get("backtest", {})

    if args.synthetic:
        n = recommended_n(
            initial_train_days=bt_cfg.get("initial_train_days", 504),
            step_days=bt_cfg.get("step_days", 63),
        )
        logger.info("--synthetic: generating %d rows", n)
        return make_samsung_ohlcv(n=n)

    data_config = DataConfig(**data_cfg_section)

    if args.no_download:
        cache = _cache_path(data_config)
        if not cache.exists():
            raise FileNotFoundError(
                f"--no-download specified but cache not found: {cache}"
            )
        logger.info("--no-download: loading from %s", cache)
        return load_ohlcv_from_csv(cache)

    logger.info("Downloading / loading cached data for %s", data_config.ticker)
    return load_ohlcv(data_config)


def _build_feat_df(
    ohlcv: pd.DataFrame,
    ext_series: list[ExternalSeriesConfig],
    *,
    target_kind: str = "next_day_log_return",
    horizon: int = 1,
    cache_dir: str = "data/external",
    cache_ttl_hours: float = 24.0,
) -> pd.DataFrame:
    base = ohlcv.copy()
    ext_cols: list[str] = []

    if ext_series:
        fetch_start = base.index[0].strftime("%Y-%m-%d")
        fetch_end = base.index[-1].strftime("%Y-%m-%d")
        base = merge_external_features(
            base, ext_series,
            start=fetch_start, end=fetch_end,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )
        ext_cols = [c for c in base.columns if c.startswith("ext_")]
        if not ext_cols:
            logger.warning("No ext_* columns were merged — check symbols / network")

    feat_df = build_feature_matrix(
        base,
        return_windows=[1, 5, 10, 20],
        ma_windows=[5, 20, 60],
        rsi_window=14,
        atr_window=14,
        volume_ma_window=20,
        target_kind=target_kind,
        horizon=horizon,
    )

    if ext_cols:
        feat_df = feat_df.join(base[ext_cols], how="left")
        feat_df.dropna(inplace=True)

    return feat_df


def _run_lgbm(feat_df: pd.DataFrame, lgbm_params: dict, bt: dict) -> dict:
    def factory() -> LGBMForecaster:
        return LGBMForecaster(params={**lgbm_params, "verbose": -1})

    result = walk_forward(feat_df, factory, **bt)
    return result.aggregate_metrics()


def _run_baseline(feat_df: pd.DataFrame, factory, bt: dict) -> dict:
    result = walk_forward(feat_df, factory, **bt)
    return result.aggregate_metrics()


def _fmt(metrics: dict) -> str:
    da = metrics.get("directional_accuracy", float("nan"))
    sh = metrics.get("sharpe", float("nan"))
    mae = metrics.get("mae", float("nan"))
    ic = metrics.get("ic", float("nan"))
    return f"DA={da:.3f}  Sharpe={sh:.3f}  MAE={mae:.5f}  IC={ic:.4f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation study: external data variants")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--no-download", action="store_true")
    p.add_argument("--reports-dir", default="reports", metavar="DIR")
    args = p.parse_args(argv)

    cfg = _load_config()
    bt = {
        "initial_train_days": cfg.get("backtest", {}).get("initial_train_days", 504),
        "step_days": cfg.get("backtest", {}).get("step_days", 63),
        "min_train_days": cfg.get("backtest", {}).get("min_train_days", 252),
    }
    lgbm_params = {**_LGBM_DEFAULTS, **cfg.get("lgbm", {})}

    # Load base OHLCV
    try:
        ohlcv = _get_ohlcv(args, cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("OHLCV loaded: %d rows  (%s … %s)",
                len(ohlcv), ohlcv.index[0].date(), ohlcv.index[-1].date())

    results: dict[str, dict] = {}

    # ── LightGBM ablation variants ────────────────────────────────────────────
    for variant, ext_series in _EXTERNAL_VARIANTS.items():
        logger.info("=== LightGBM variant: %s ===", variant)
        try:
            feat_df = _build_feat_df(
                ohlcv, ext_series,
                cache_dir=cfg.get("external_data", {}).get("cache_dir", "data/external"),
                cache_ttl_hours=float(
                    cfg.get("external_data", {}).get("cache_ttl_hours", 24.0)
                ),
            )
            n_feat = len(feature_columns(feat_df))
            logger.info("  features: %d  (rows: %d)", n_feat, len(feat_df))
            metrics = _run_lgbm(feat_df, lgbm_params, bt)
            results[f"lgbm_{variant}"] = {
                "model": "LGBMForecaster",
                "variant": variant,
                "n_features": n_feat,
                "n_ext_features": len(ext_series),
                "metrics": metrics,
            }
            logger.info("  %s", _fmt(metrics))
        except Exception as exc:
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            results[f"lgbm_{variant}"] = {"error": str(exc)}

    # ── Naive baselines (use internal_only feature set) ────────────────────────
    logger.info("=== Building internal-only feature set for baselines ===")
    try:
        feat_df_base = _build_feat_df(ohlcv, [])
    except Exception as exc:
        logger.error("Failed to build baseline features: %s", exc)
        return 1

    baselines: list[tuple[str, object]] = [
        ("zero_predictor", ZeroPredictor),
        ("prev_return", PrevReturnPredictor),
        ("ridge", RidgeForecaster),
    ]
    for name, factory in baselines:
        logger.info("=== Baseline: %s ===", name)
        try:
            metrics = _run_baseline(feat_df_base, factory, bt)
            results[f"baseline_{name}"] = {
                "model": factory.__name__,
                "variant": "internal_only",
                "metrics": metrics,
            }
            logger.info("  %s", _fmt(metrics))
        except Exception as exc:
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            results[f"baseline_{name}"] = {"error": str(exc)}

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"{'Experiment':<35} {'DA':>7} {'Sharpe':>8} {'MAE':>10} {'IC':>8}")
    print("-" * 75)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<35}  ERROR: {r['error'][:30]}")
            continue
        m = r["metrics"]
        da = m.get("directional_accuracy", float("nan"))
        sh = m.get("sharpe", float("nan"))
        mae = m.get("mae", float("nan"))
        ic = m.get("ic", float("nan"))
        print(f"{key:<35} {da:>7.4f} {sh:>8.4f} {mae:>10.6f} {ic:>8.4f}")
    print("=" * 75)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"ablation_summary_{ts}.json"
    payload = {
        "generated_at": ts,
        "data_rows": len(ohlcv),
        "data_start": str(ohlcv.index[0].date()),
        "data_end": str(ohlcv.index[-1].date()),
        "backtest_config": bt,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Ablation summary saved: %s", out_path)
    print(f"\nReport saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
