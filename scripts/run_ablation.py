"""Ablation study: external data contribution to model performance.

Compares four cumulative external-data configurations (DART excluded until
DART_API_KEY is available):

    internal_only           Technical indicators only (10 features)
    internal_market         + 6 market-price series (KOSPI, USD/KRW, SOX,
                              Hynix, DXY, S&P 500) -- yfinance, no API key
    internal_market_news    + 4 GDELT news series (Samsung-specific tone/count
                              + semiconductor-sector tone/count)
                              -- no API key, internet required
    internal_market_news_dart  (structure preserved for DART_API_KEY)
                            + DART earnings + major-event disclosures
                              -- requires DART_API_KEY

Each variant also runs three naive baselines (zero, prev_return, ridge)
as a performance lower bound.

Leakage safety
--------------
Every external series is shifted by lag_days=1 Samsung trading day before
merging.  No same-day data enters the feature matrix.

Synthetic vs. real-data environment
------------------------------------
``--synthetic`` generates synthetic Samsung OHLCV price data (no network
needed for the price series).  However, GDELT and yfinance calls STILL
reach the internet for real external data.
- Without internet: GDELT/yfinance fail silently (all-NaN → auto-dropped).
  Variants that rely on them will run with N_merged_ext < N_configured.
- With internet: all non-DART series fetch normally even in --synthetic mode.

DART always requires DART_API_KEY regardless of --synthetic flag.

News feature quality caveat
----------------------------
GDELT tone is based on CAMEO geopolitical sentiment, not a financial NLP
model.  The news features are heuristic/baseline quality:
  news_tone_*   : GDELT tone proxy (geopolitical event framing, noisy)
  news_count_*  : raw article volume (attention proxy, valence-neutral)
Interpret any DA/Sharpe uplift from news features cautiously.
Replace with FinBERT / KoFinBERT for production use.

Usage::

    python scripts/run_ablation.py                   # live market data
    python scripts/run_ablation.py --synthetic       # synthetic OHLCV
    python scripts/run_ablation.py --no-download     # cached data only
    python scripts/run_ablation.py --include-dart    # add DART variants
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# -- make repo root importable -------------------------------------------------
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

# -- Config --------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"

_LGBM_DEFAULTS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    num_leaves=31, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
)

# -- External series definitions -----------------------------------------------

_MARKET_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(name="kospi",  source="market", symbol="^KS11",      lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="usdkrw", source="market", symbol="USDKRW=X",   lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="sox",    source="market", symbol="^SOX",        lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="hynix",  source="market", symbol="000660.KS",   lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="dxy",    source="market", symbol="DX-Y.NYB",    lag_days=1, feature_type="log_return"),
    ExternalSeriesConfig(name="sp500",  source="market", symbol="^GSPC",       lag_days=1, feature_type="log_return"),
]

# Improved GDELT queries: Samsung-specific + semiconductor sector
# Two separate queries capture different signal dimensions:
#   Samsung query   → company-specific attention / tone
#   Semicon query   → sector-level macro attention / tone
_NEWS_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(
        name="news_tone_samsung", source="gdelt",
        symbol="Samsung Electronics semiconductor chip memory",
        lag_days=1, feature_type="level",
        extra={"metric": "tone"},
    ),
    ExternalSeriesConfig(
        name="news_count_samsung", source="gdelt",
        symbol="Samsung Electronics semiconductor chip memory",
        lag_days=1, feature_type="level",
        extra={"metric": "count"},
    ),
    ExternalSeriesConfig(
        name="news_tone_semicon", source="gdelt",
        symbol="semiconductor DRAM NAND HBM memory chip foundry",
        lag_days=1, feature_type="level",
        extra={"metric": "tone"},
    ),
    ExternalSeriesConfig(
        name="news_count_semicon", source="gdelt",
        symbol="semiconductor DRAM NAND HBM memory chip foundry",
        lag_days=1, feature_type="level",
        extra={"metric": "count"},
    ),
]

_DART_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(
        name="dart_earnings", source="dart", symbol="00126380",
        lag_days=1, feature_type="level", api_key_env="DART_API_KEY",
        extra={"pblntf_ty": "A"},
    ),
    ExternalSeriesConfig(
        name="dart_major", source="dart", symbol="00126380",
        lag_days=1, feature_type="level", api_key_env="DART_API_KEY",
        extra={"pblntf_ty": "B"},
    ),
]

# -- Ablation variants ---------------------------------------------------------

def _build_variants(include_dart: bool) -> dict[str, dict]:
    variants = {
        "internal_only": {
            "series": [],
            "note": "baseline",
            "needs_internet": False,
            "needs_dart": False,
        },
        "internal_market": {
            "series": _MARKET_SERIES,
            "note": "yfinance (needs internet)",
            "needs_internet": True,
            "needs_dart": False,
        },
        "internal_market_news": {
            "series": _MARKET_SERIES + _NEWS_SERIES,
            "note": "GDELT (needs internet; news=CAMEO proxy)",
            "needs_internet": True,
            "needs_dart": False,
        },
    }
    if include_dart:
        variants["internal_market_news_dart"] = {
            "series": _MARKET_SERIES + _NEWS_SERIES + _DART_SERIES,
            "note": "requires DART_API_KEY",
            "needs_internet": True,
            "needs_dart": True,
        }
    return variants


# -- Helpers -------------------------------------------------------------------

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
        logger.info("--synthetic: generating %d rows (price data only)", n)
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
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Return (feat_df, merged_ext_cols, drop_log)."""
    base = ohlcv.copy()
    ext_cols: list[str] = []
    drop_log: dict[str, str] = {}

    if ext_series:
        configured_names = {f"ext_{s.name}" for s in ext_series}
        fetch_start = base.index[0].strftime("%Y-%m-%d")
        fetch_end   = base.index[-1].strftime("%Y-%m-%d")
        base = merge_external_features(
            base, ext_series,
            start=fetch_start, end=fetch_end,
            cache_dir=cache_dir, cache_ttl_hours=cache_ttl_hours,
        )
        ext_cols = [c for c in base.columns if c.startswith("ext_")]

        # Track series that failed entirely (never added to ohlcv)
        for name in configured_names:
            if name not in ext_cols:
                drop_log[name] = "fetch failed (API error / key missing / network)"

        if not ext_cols:
            logger.warning("[ablation] 0 ext_* columns merged after all attempts")

    feat_df = build_feature_matrix(
        base,
        return_windows=[1, 5, 10, 20],
        ma_windows=[5, 20, 60],
        rsi_window=14, atr_window=14, volume_ma_window=20,
        target_kind=target_kind, horizon=horizon,
    )

    if ext_cols:
        feat_df = feat_df.join(base[ext_cols], how="left")

        # Drop all-NaN ext columns (no data coverage or fetch failed silently)
        dead = [c for c in ext_cols if feat_df[c].isna().all()]
        if dead:
            logger.warning(
                "[ablation] dropping %d all-NaN ext columns: %s", len(dead), dead
            )
            for c in dead:
                drop_log[c] = "all-NaN (no coverage / no internet)"
            feat_df.drop(columns=dead, inplace=True)
            ext_cols = [c for c in ext_cols if c not in dead]

        # Drop rows only on non-ext columns (LightGBM handles NaN natively)
        non_ext = [c for c in feat_df.columns if not c.startswith("ext_")]
        feat_df.dropna(subset=non_ext, inplace=True)

        for col in ext_cols:
            nan_pct = feat_df[col].isna().mean()
            if nan_pct > 0.5:
                logger.warning(
                    "[ablation] '%s' is %.0f%% NaN -- low coverage", col, nan_pct * 100
                )

    return feat_df, ext_cols, drop_log


def _run_lgbm(feat_df: pd.DataFrame, lgbm_params: dict, bt: dict) -> dict:
    def factory() -> LGBMForecaster:
        return LGBMForecaster(params={**lgbm_params, "verbose": -1})
    return walk_forward(feat_df, factory, **bt).aggregate_metrics()


def _run_baseline(feat_df: pd.DataFrame, factory, bt: dict) -> dict:
    return walk_forward(feat_df, factory, **bt).aggregate_metrics()


# -- Main ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation: external data contribution")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--synthetic",   action="store_true",
                   help="Synthetic OHLCV; market/GDELT still use real internet")
    g.add_argument("--no-download", action="store_true",
                   help="Cached OHLCV only")
    p.add_argument("--include-dart", action="store_true",
                   help="Add DART variant (requires DART_API_KEY env var)")
    p.add_argument("--reports-dir", default="reports", metavar="DIR")
    args = p.parse_args(argv)

    cfg = _load_config()
    bt = {
        "initial_train_days": cfg.get("backtest", {}).get("initial_train_days", 504),
        "step_days":          cfg.get("backtest", {}).get("step_days",           63),
        "min_train_days":     cfg.get("backtest", {}).get("min_train_days",      252),
    }
    lgbm_params  = {**_LGBM_DEFAULTS, **cfg.get("lgbm", {})}
    ext_cfg      = cfg.get("external_data", {})
    cache_dir    = ext_cfg.get("cache_dir",       "data/external")
    cache_ttl    = float(ext_cfg.get("cache_ttl_hours", 24.0))
    variants     = _build_variants(include_dart=args.include_dart)

    has_internet = _check_internet()
    has_dart_key = bool(os.environ.get("DART_API_KEY", ""))

    _print_env_summary(args, has_internet, has_dart_key)

    try:
        ohlcv = _get_ohlcv(args, cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "OHLCV loaded: %d rows  (%s ... %s)",
        len(ohlcv), ohlcv.index[0].date(), ohlcv.index[-1].date(),
    )

    results: dict[str, dict] = {}

    # -- LightGBM ablation -----------------------------------------------------
    for variant_key, vdef in variants.items():
        logger.info("=== LightGBM variant: %s ===", variant_key)

        # Warn up front if required resources are missing
        if vdef["needs_dart"] and not has_dart_key:
            logger.warning("  DART_API_KEY not set -- DART series will fail silently")
        if vdef["needs_internet"] and not has_internet:
            logger.warning("  No internet detected -- market/GDELT series may fail")

        try:
            feat_df, merged_ext, drop_log = _build_feat_df(
                ohlcv, vdef["series"],
                cache_dir=cache_dir, cache_ttl_hours=cache_ttl,
            )
            n_feat        = len(feature_columns(feat_df))
            n_configured  = len(vdef["series"])
            n_merged      = len(merged_ext)
            n_dropped     = len(drop_log)

            logger.info(
                "  features=%d  ext configured=%d / merged=%d / dropped=%d  rows=%d",
                n_feat, n_configured, n_merged, n_dropped, len(feat_df),
            )

            # Per-column NaN coverage
            coverage: dict[str, float] = {}
            for col in merged_ext:
                if col in feat_df.columns:
                    coverage[col] = 1.0 - float(feat_df[col].isna().mean())

            metrics = _run_lgbm(feat_df, lgbm_params, bt)
            results[f"lgbm_{variant_key}"] = {
                "model":            "LGBMForecaster",
                "variant":          variant_key,
                "n_features":       n_feat,
                "n_configured_ext": n_configured,
                "n_merged_ext":     n_merged,
                "n_dropped_ext":    n_dropped,
                "merged_ext_cols":  merged_ext,
                "coverage":         coverage,
                "drop_log":         drop_log,
                "note":             vdef["note"],
                "metrics":          metrics,
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
                "error": str(exc), "variant": variant_key, "note": vdef["note"],
            }

    # -- Naive baselines -------------------------------------------------------
    logger.info("=== Building internal-only features for baselines ===")
    try:
        feat_df_base, _, _ = _build_feat_df(ohlcv, [])
    except Exception as exc:
        logger.error("Failed to build baseline features: %s", exc)
        return 1

    n_base = len(feature_columns(feat_df_base))
    for bl_name, bl_factory in [
        ("zero_predictor", ZeroPredictor),
        ("prev_return",    PrevReturnPredictor),
        ("ridge",          RidgeForecaster),
    ]:
        logger.info("=== Baseline: %s ===", bl_name)
        try:
            metrics = _run_baseline(feat_df_base, bl_factory, bt)
            results[f"baseline_{bl_name}"] = {
                "model": bl_factory.__name__, "variant": "internal_only",
                "n_features": n_base, "n_merged_ext": 0, "metrics": metrics,
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

    # -- Output ----------------------------------------------------------------
    _print_feature_coverage(results)
    _print_summary(results)
    _print_analysis(results, args)

    # -- Save JSON --------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"ablation_summary_{ts}.json"
    payload = {
        "generated_at":   ts,
        "environment": {
            "synthetic_ohlcv": args.synthetic,
            "has_internet":    has_internet,
            "has_dart_key":    has_dart_key,
        },
        "data_rows":      len(ohlcv),
        "data_start":     str(ohlcv.index[0].date()),
        "data_end":       str(ohlcv.index[-1].date()),
        "backtest_config": bt,
        "news_quality_caveat": (
            "GDELT tone uses CAMEO geopolitical sentiment, not financial NLP. "
            "news_tone_* and news_count_* are heuristic/baseline quality proxies. "
            "Replace with FinBERT/KoFinBERT for production use."
        ),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Ablation summary saved: %s", out_path)
    print(f"\nReport saved → {out_path}")
    return 0


# -- Print helpers -------------------------------------------------------------

def _check_internet() -> bool:
    """Quick probe: try reaching GDELT API host (no data fetched)."""
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("api.gdeltproject.org", 443)
        )
        return True
    except Exception:
        return False


def _print_env_summary(args, has_internet: bool, has_dart_key: bool) -> None:
    print()
    print("=" * 70)
    print("ABLATION ENVIRONMENT")
    print("-" * 70)
    print(f"  OHLCV source      : {'synthetic (price data only)' if args.synthetic else 'real download'}")
    print(f"  Internet access   : {'YES' if has_internet else 'NO -- market/GDELT series will fail silently'}")
    print(f"  DART_API_KEY      : {'SET' if has_dart_key else 'NOT SET -- DART series will fail silently'}")
    print()
    if args.synthetic and not has_internet:
        print("  NOTE: --synthetic with no internet means GDELT and yfinance will")
        print("        fail silently.  Only internal_only variant will have full data.")
    if args.synthetic and has_internet:
        print("  NOTE: --synthetic uses synthetic OHLCV prices but GDELT/yfinance")
        print("        fetch REAL external data for the synthetic date range.")
    print("=" * 70)


def _print_feature_coverage(results: dict) -> None:
    """Show per-series coverage for each variant."""
    print()
    print("-" * 70)
    print("EXTERNAL FEATURE COVERAGE")
    print("-" * 70)
    for key, r in results.items():
        if "lgbm_" not in key:
            continue
        if "error" in r:
            continue
        n_conf  = r.get("n_configured_ext", 0)
        n_merge = r.get("n_merged_ext",     0)
        n_drop  = r.get("n_dropped_ext",    0)
        if n_conf == 0:
            continue
        print(f"  {key}:")
        print(f"    configured={n_conf}  merged={n_merge}  dropped={n_drop}")
        for col, cov in (r.get("coverage") or {}).items():
            print(f"      [ok] {col:<30}  coverage={cov*100:.1f}%")
        for col, reason in (r.get("drop_log") or {}).items():
            print(f"      [--] {col:<30}  [{reason}]")
    print("-" * 70)


def _print_summary(results: dict) -> None:
    header = f"{'Experiment':<38} {'N_feat':>6} {'N_ext':>5}  {'DA':>7} {'Sharpe':>8} {'RMSE':>9} {'IC':>8}"
    sep    = "-" * len(header)
    print()
    print("=" * len(header))
    print(header)
    print(sep)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<38}  ERROR: {r['error'][:38]}")
            continue
        m    = r.get("metrics", {})
        note = r.get("note", "")
        da   = m.get("directional_accuracy", float("nan"))
        sh   = m.get("sharpe", float("nan"))
        rmse = m.get("rmse", float("nan"))
        ic   = m.get("ic", float("nan"))
        n_feat  = r.get("n_features", 0)
        n_merge = r.get("n_merged_ext", 0)
        note_str = f"  [{note}]" if note else ""
        print(
            f"{key:<38} {n_feat:>6} {n_merge:>5}  "
            f"{da:>7.4f} {sh:>8.4f} {rmse:>9.6f} {ic:>8.4f}"
            f"{note_str}"
        )
    print("=" * len(header))


def _print_analysis(results: dict, args) -> None:
    """Compare DA across LGBM variants; honest about data quality."""
    def _da(key: str) -> float | None:
        r = results.get(key, {})
        if "error" in r:
            return None
        return r.get("metrics", {}).get("directional_accuracy")

    base_da = _da("lgbm_internal_only")
    print()
    print("-" * 70)
    print("ANALYSIS  (LGBM variants vs. internal_only)")
    print("-" * 70)

    if base_da is None:
        print("  internal_only variant failed -- no analysis available.")
    else:
        for key in [k for k in results if k.startswith("lgbm_") and k != "lgbm_internal_only"]:
            r  = results[key]
            da = _da(key)
            if da is None:
                err = r.get("error", "")
                print(f"  {key:<38}  FAILED: {err[:30]}")
                continue
            delta    = da - base_da
            arrow    = "^" if delta > 0.001 else ("v" if delta < -0.001 else "~")
            n_conf   = r.get("n_configured_ext", 0)
            n_merge  = r.get("n_merged_ext",     0)
            frac_str = f"[{n_merge}/{n_conf} series merged]" if n_conf > 0 else ""
            print(f"  {key:<38}  DA={da:.4f}  D={delta:+.4f} {arrow}  {frac_str}")

    # Dynamic market-feature assessment
    market_da    = _da("lgbm_internal_market")
    market_delta = (market_da - base_da) if (base_da is not None and market_da is not None) else None

    print()
    print("  HONEST ASSESSMENT")
    print("  -----------------")
    print("  * Market features (KOSPI/SOX/USD/KRW etc.): low-latency, reproducible.")
    if market_delta is not None and market_delta < -0.005:
        print(f"    [!] DA DECREASED by {abs(market_delta):.4f} vs internal_only.")
        print("    Possible causes:")
        print("      1) Multicollinear features (KOSPI/SOX/SP500/Hynix are correlated).")
        print("         Try colsample_bytree < 0.8 or feature selection.")
        print("      2) Default hyperparams tuned for 10 features, not 16.")
        print("         Consider re-running with tuned lgbm_params for this feature set.")
        print("      3) Noise: 2183 OOS obs over 11 years -- small changes in DA are")
        print("         within sampling noise (~1-2 pp) for 35 folds of 63 obs each.")
    elif market_delta is not None and market_delta > 0.005:
        print(f"    [ok] DA improved by {market_delta:.4f} vs internal_only.")
    else:
        print("    DA change is within noise range (+/-0.5 pp).")
    print("    Likely to contribute most on high-volatility macro event days.")
    print()
    print("  * GDELT news features: CAMEO geopolitical tone, NOT financial NLP.")
    print("    Signal quality: heuristic/baseline.  Treat any uplift with caution.")
    print("    Better alternative: FinBERT / KoFinBERT (requires model download).")
    print()
    print("  * DART features: official audited facts.  Awaiting DART_API_KEY.")
    if args.synthetic:
        print()
        print("  * --synthetic mode: OHLCV prices are random-walk data.")
        print("    External series with real data on synthetic dates may produce")
        print("    spurious correlations.  Run on real OHLCV for valid assessment.")
    print("-" * 70)


if __name__ == "__main__":
    sys.exit(main())
