"""Ablation study: external data contribution to model performance.

Compares four cumulative external-data configurations:

    internal_only           Technical indicators only (10 features)
    internal_market         + 6 market-price series (KOSPI, USD/KRW, SOX,
                              HYnix, DXY, S&P 500) — yfinance, no API key
    internal_market_dart    + DART official disclosures (earnings + major
                              events) — requires DART_API_KEY
    internal_market_dart_news + GDELT news sentiment + article volume
                              — no API key, internet required

Each variant also runs three naive baselines (zero, prev_return, ridge)
to provide a lower-bound reference.

Anti-leakage: every external series is shifted by lag_days=1 Samsung
trading day before merging.  No same-day data enters the feature matrix.

API key requirements
--------------------
market series     — none (yfinance)
dart series       — DART_API_KEY  (free: opendart.fss.or.kr)
news series       — none (GDELT 2.0 public API)

If an API key is missing the corresponding series fails silently inside
merge_external_features; the variant continues with fewer features.
The n_merged_ext column in the output table shows the actual count.

News feature caveat
-------------------
GDELT tone is based on CAMEO geopolitical sentiment, not a financial NLP
model.  The news features are heuristic/baseline quality.  Interpret any
DA/Sharpe uplift cautiously.

Usage::

    python scripts/run_ablation.py                   # live market data
    python scripts/run_ablation.py --synthetic       # offline, no network
    python scripts/run_ablation.py --no-download     # cached data only
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

_LGBM_DEFAULTS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
)

# ── Ablation variants (cumulative) ────────────────────────────────────────────
# Each variant adds on top of the previous tier.
# "note" is shown in the output table to flag data-quality caveats.

_MARKET_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(name="kospi",  source="market", symbol="^KS11",      lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="usdkrw", source="market", symbol="USDKRW=X",   lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="sox",    source="market", symbol="^SOX",        lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="hynix",  source="market", symbol="000660.KS",   lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="dxy",    source="market", symbol="DX-Y.NYB",    lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="sp500",  source="market", symbol="^GSPC",       lag_days=1, feature_type="log_return"),
]

_DART_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(name="dart_earnings", source="dart", symbol="00126380",
                         lag_days=1, feature_type="level", api_key_env="DART_API_KEY",
                         extra={"pblntf_ty": "A"}),
    ExternalSeriesConfig(name="dart_major",    source="dart", symbol="00126380",
                         lag_days=1, feature_type="level", api_key_env="DART_API_KEY",
                         extra={"pblntf_ty": "B"}),
]

_NEWS_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(name="news_tone",  source="gdelt",
                         symbol="samsung electronics semiconductor",
                         lag_days=1, feature_type="level",
                         extra={"metric": "tone"}),
    ExternalSeriesConfig(name="news_count", source="gdelt",
                         symbol="samsung electronics",
                         lag_days=1, feature_type="level",
                         extra={"metric": "count"}),
]

_VARIANTS: dict[str, dict] = {
    "internal_only": {
        "series": [],
        "note": "",
        "keys_needed": "none",
    },
    "internal_market": {
        "series": _MARKET_SERIES,
        "note": "",
        "keys_needed": "none",
    },
    "internal_market_dart": {
        "series": _MARKET_SERIES + _DART_SERIES,
        "note": "requires DART_API_KEY",
        "keys_needed": "DART_API_KEY",
    },
    "internal_market_dart_news": {
        "series": _MARKET_SERIES + _DART_SERIES + _NEWS_SERIES,
        "note": "news=baseline quality (GDELT CAMEO tone)",
        "keys_needed": "DART_API_KEY + internet",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_ohlcv(args: argparse.Namespace, cfg: dict) -> pd.DataFrame:
    bt_cfg = cfg.get("backtest", {})
    if args.synthetic:
        n = recommended_n(
            initial_train_days=bt_cfg.get("initial_train_days", 504),
            step_days=bt_cfg.get("step_days", 63),
        )
        logger.info("--synthetic: generating %d rows", n)
        return make_samsung_ohlcv(n=n)

    data_config = DataConfig(**cfg.get("data", {}))
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
) -> tuple[pd.DataFrame, list[str]]:
    """Return (feat_df, merged_ext_cols)."""
    base = ohlcv.copy()
    ext_cols: list[str] = []

    if ext_series:
        fetch_start = base.index[0].strftime("%Y-%m-%d")
        fetch_end   = base.index[-1].strftime("%Y-%m-%d")
        base = merge_external_features(
            base, ext_series,
            start=fetch_start, end=fetch_end,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )
        ext_cols = [c for c in base.columns if c.startswith("ext_")]
        if not ext_cols:
            logger.warning(
                "[ablation] 0 ext_* columns merged — check API keys / symbols"
            )

    feat_df = build_feature_matrix(
        base,
        return_windows=[1, 5, 10, 20],
        ma_windows=[5, 20, 60],
        rsi_window=14, atr_window=14, volume_ma_window=20,
        target_kind=target_kind, horizon=horizon,
    )

    if ext_cols:
        feat_df = feat_df.join(base[ext_cols], how="left")
        # Remove ext_* columns that are entirely NaN (failed fetches / no coverage).
        dead_cols = [c for c in ext_cols if feat_df[c].isna().all()]
        if dead_cols:
            logger.warning(
                "[ablation] dropping %d all-NaN ext columns (fetch failed or no coverage): %s",
                len(dead_cols), dead_cols,
            )
            feat_df.drop(columns=dead_cols, inplace=True)
            ext_cols = [c for c in ext_cols if c not in dead_cols]

        # Drop rows where non-external columns are NaN (target / technical warmup).
        # LightGBM handles NaN natively, so partial ext-column NaN is acceptable.
        non_ext = [c for c in feat_df.columns if not c.startswith("ext_")]
        feat_df.dropna(subset=non_ext, inplace=True)

        for col in ext_cols:
            nan_pct = feat_df[col].isna().mean()
            if nan_pct > 0.3:
                logger.warning(
                    "[ablation] ext col '%s' is %.0f%% NaN after merge", col, nan_pct * 100
                )

    return feat_df, ext_cols


def _run_lgbm(feat_df: pd.DataFrame, lgbm_params: dict, bt: dict) -> dict:
    def factory() -> LGBMForecaster:
        return LGBMForecaster(params={**lgbm_params, "verbose": -1})
    return walk_forward(feat_df, factory, **bt).aggregate_metrics()


def _run_baseline(feat_df: pd.DataFrame, factory, bt: dict) -> dict:
    return walk_forward(feat_df, factory, **bt).aggregate_metrics()


def _fmt_row(
    key: str,
    metrics: dict,
    n_feat: int,
    n_merged: int,
    note: str,
) -> str:
    da   = metrics.get("directional_accuracy", float("nan"))
    sh   = metrics.get("sharpe", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    ic   = metrics.get("ic", float("nan"))
    note_str = f"  [{note}]" if note else ""
    return (
        f"{key:<35} {n_feat:>5} {n_merged:>6}  "
        f"{da:>7.4f} {sh:>8.4f} {rmse:>9.6f} {ic:>8.4f}"
        f"{note_str}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation: external data contribution")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--synthetic",    action="store_true",
                   help="Generate synthetic OHLCV — no network required")
    g.add_argument("--no-download",  action="store_true",
                   help="Use existing cache only")
    p.add_argument("--reports-dir",  default="reports", metavar="DIR")
    args = p.parse_args(argv)

    cfg = _load_config()
    bt = {
        "initial_train_days": cfg.get("backtest", {}).get("initial_train_days", 504),
        "step_days":          cfg.get("backtest", {}).get("step_days",           63),
        "min_train_days":     cfg.get("backtest", {}).get("min_train_days",      252),
    }
    lgbm_params = {**_LGBM_DEFAULTS, **cfg.get("lgbm", {})}
    ext_cfg = cfg.get("external_data", {})
    cache_dir        = ext_cfg.get("cache_dir",        "data/external")
    cache_ttl_hours  = float(ext_cfg.get("cache_ttl_hours", 24.0))

    try:
        ohlcv = _get_ohlcv(args, cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "OHLCV loaded: %d rows  (%s … %s)",
        len(ohlcv), ohlcv.index[0].date(), ohlcv.index[-1].date(),
    )

    results: dict[str, dict] = {}

    # ── LightGBM ablation ─────────────────────────────────────────────────────
    for variant_key, variant_def in _VARIANTS.items():
        logger.info("=== LightGBM variant: %s ===", variant_key)
        try:
            feat_df, merged_ext = _build_feat_df(
                ohlcv, variant_def["series"],
                cache_dir=cache_dir, cache_ttl_hours=cache_ttl_hours,
            )
            n_feat = len(feature_columns(feat_df))
            n_merged = len(merged_ext)
            logger.info(
                "  n_features=%d  n_merged_ext=%d/%d  rows=%d",
                n_feat, n_merged, len(variant_def["series"]), len(feat_df),
            )

            # Missing-ratio per merged column
            miss_ratios: dict[str, float] = {}
            for col in merged_ext:
                if col in feat_df.columns:
                    miss_ratios[col] = float(feat_df[col].isna().mean())

            metrics = _run_lgbm(feat_df, lgbm_params, bt)
            results[f"lgbm_{variant_key}"] = {
                "model":             "LGBMForecaster",
                "variant":           variant_key,
                "n_features":        n_feat,
                "n_configured_ext":  len(variant_def["series"]),
                "n_merged_ext":      n_merged,
                "merged_ext_cols":   merged_ext,
                "missing_ratios":    miss_ratios,
                "note":              variant_def["note"],
                "metrics":           metrics,
            }
            logger.info(
                "  DA=%.4f  Sharpe=%.4f  RMSE=%.6f  IC=%.4f",
                metrics.get("directional_accuracy", float("nan")),
                metrics.get("sharpe", float("nan")),
                metrics.get("rmse", float("nan")),
                metrics.get("ic", float("nan")),
            )
        except Exception as exc:
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            results[f"lgbm_{variant_key}"] = {
                "error": str(exc),
                "variant": variant_key,
                "note": variant_def["note"],
            }

    # ── Naive baselines (internal_only feature set) ────────────────────────────
    logger.info("=== Building internal-only features for baselines ===")
    try:
        feat_df_base, _ = _build_feat_df(ohlcv, [])
    except Exception as exc:
        logger.error("Failed to build baseline features: %s", exc)
        return 1

    n_base_feat = len(feature_columns(feat_df_base))
    for bl_name, bl_factory in [
        ("zero_predictor", ZeroPredictor),
        ("prev_return",    PrevReturnPredictor),
        ("ridge",          RidgeForecaster),
    ]:
        logger.info("=== Baseline: %s ===", bl_name)
        try:
            metrics = _run_baseline(feat_df_base, bl_factory, bt)
            results[f"baseline_{bl_name}"] = {
                "model":      bl_factory.__name__,
                "variant":    "internal_only",
                "n_features": n_base_feat,
                "n_merged_ext": 0,
                "metrics":    metrics,
            }
            logger.info(
                "  DA=%.4f  Sharpe=%.4f  RMSE=%.6f  IC=%.4f",
                metrics.get("directional_accuracy", float("nan")),
                metrics.get("sharpe", float("nan")),
                metrics.get("rmse", float("nan")),
                metrics.get("ic", float("nan")),
            )
        except Exception as exc:
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            results[f"baseline_{bl_name}"] = {"error": str(exc)}

    # ── Summary table ──────────────────────────────────────────────────────────
    _print_summary(results)

    # ── Analysis ──────────────────────────────────────────────────────────────
    _print_analysis(results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"ablation_summary_{ts}.json"
    payload = {
        "generated_at":   ts,
        "data_rows":      len(ohlcv),
        "data_start":     str(ohlcv.index[0].date()),
        "data_end":       str(ohlcv.index[-1].date()),
        "backtest_config": bt,
        "news_caveat": (
            "GDELT tone uses CAMEO geopolitical sentiment, not a financial NLP model. "
            "news_tone / news_count features are baseline-quality proxies. "
            "Replace with FinBERT for production use."
        ),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Ablation summary saved: %s", out_path)
    print(f"\nReport saved → {out_path}")
    return 0


def _print_summary(results: dict) -> None:
    hdr = f"{'Experiment':<35} {'N_feat':>6} {'N_ext':>5}  {'DA':>7} {'Sharpe':>8} {'RMSE':>9} {'IC':>8}"
    sep = "─" * len(hdr)
    print()
    print("=" * len(hdr))
    print(hdr)
    print(sep)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<35}  ERROR: {r['error'][:40]}")
            continue
        m    = r.get("metrics", {})
        note = r.get("note", "")
        print(_fmt_row(
            key,
            m,
            r.get("n_features", 0),
            r.get("n_merged_ext", 0),
            note,
        ))
    print("=" * len(hdr))


def _print_analysis(results: dict) -> None:
    """Print a human-readable verdict comparing the four LGBM variants."""
    variants = ["lgbm_internal_only", "lgbm_internal_market",
                "lgbm_internal_market_dart", "lgbm_internal_market_dart_news"]

    def _da(key: str) -> float | None:
        r = results.get(key, {})
        if "error" in r:
            return None
        return r.get("metrics", {}).get("directional_accuracy")

    print()
    print("─" * 70)
    print("ANALYSIS")
    print("─" * 70)

    base_da = _da("lgbm_internal_only")
    if base_da is None:
        print("  internal_only variant failed — no analysis available.")
        return

    for v in variants[1:]:
        da = _da(v)
        if da is None:
            r = results.get(v, {})
            n_merged = r.get("n_merged_ext", 0)
            n_conf   = r.get("n_configured_ext", 0)
            if n_merged < n_conf:
                print(
                    f"  {v}: only {n_merged}/{n_conf} series merged "
                    "(check API keys / network)"
                )
            else:
                print(f"  {v}: FAILED or not available")
            continue
        delta = da - base_da
        direction = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "~")
        r = results.get(v, {})
        n_merged = r.get("n_merged_ext", 0)
        n_conf   = r.get("n_configured_ext", 0)
        merge_note = (
            f"  [{n_merged}/{n_conf} series merged]"
            if n_merged < n_conf else ""
        )
        print(
            f"  {v:<35}  DA={da:.4f}  Δ={delta:+.4f} {direction}{merge_note}"
        )

    print()
    print("  NOTE: News features (news_tone, news_count) use GDELT CAMEO sentiment.")
    print("        This is a geopolitical tone metric, NOT a financial NLP model.")
    print("        Any uplift from news features should be validated carefully.")
    print("        Replace with FinBERT/KoFinBERT for production use.")
    print("─" * 70)


if __name__ == "__main__":
    sys.exit(main())
