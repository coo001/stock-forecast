"""Pre-populate the ExternalCache for all GDELT series.

Run this once before the main ablation to avoid hitting GDELT's 1-request/5s
rate limit during model training.  Series already in cache (within TTL) are
skipped automatically.  Failed series are reported but do not abort the script.

Usage::

    python scripts/build_gdelt_cache.py
    python scripts/build_gdelt_cache.py --start 2018-01-01 --end 2024-12-31
    python scripts/build_gdelt_cache.py --force   # ignore existing cache entries
    python scripts/build_gdelt_cache.py --cache-dir data/external

GDELT rate limit
----------------
The GDELT 2.0 Document API enforces 1 request / 5 seconds per IP.
This script respects that limit via the delay already set in news_client.py
(_POLITE_DELAY_S = 5.5 s).  With 44 quarterly chunks per series and 4 series,
expect a first-run wall time of ~40-50 minutes.  Subsequent runs use the cache
and complete in seconds.

Cache location
--------------
The cache key format mirrors ``load_external_series()`` exactly:

    {source}__{symbol}__{start}__{end}

Stored as CSV + JSON metadata sidecar in ``cache_dir``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external.cache import ExternalCache
from src.data.external.news_client import fetch_series
from src.features.external_merge import ExternalSeriesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Mirrors _NEWS_SERIES from run_ablation.py.
# NOTE: keep in sync if run_ablation.py changes the series definitions.
_NEWS_SERIES: list[ExternalSeriesConfig] = [
    ExternalSeriesConfig(
        name="news_tone_samsung",
        source="gdelt",
        symbol="Samsung Electronics semiconductor chip memory",
        lag_days=1,
        feature_type="level",
        extra={"metric": "tone"},
    ),
    ExternalSeriesConfig(
        name="news_count_samsung",
        source="gdelt",
        symbol="Samsung Electronics semiconductor chip memory",
        lag_days=1,
        feature_type="level",
        extra={"metric": "count"},
    ),
    ExternalSeriesConfig(
        name="news_tone_semicon",
        source="gdelt",
        symbol="semiconductor DRAM NAND HBM memory chip foundry",
        lag_days=1,
        feature_type="level",
        extra={"metric": "tone"},
    ),
    ExternalSeriesConfig(
        name="news_count_semicon",
        source="gdelt",
        symbol="semiconductor DRAM NAND HBM memory chip foundry",
        lag_days=1,
        feature_type="level",
        extra={"metric": "count"},
    ),
]


def _cache_key(cfg: ExternalSeriesConfig, start: str, end: str) -> str:
    """Exactly mirrors the key used in load_external_series() in external_merge.py."""
    return f"{cfg.source}__{cfg.symbol}__{start}__{end}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre-populate GDELT cache")
    p.add_argument(
        "--start", default="2015-01-01", metavar="DATE",
        help="Fetch start date (ISO, default: 2015-01-01)",
    )
    p.add_argument(
        "--end", default=date.today().isoformat(), metavar="DATE",
        help="Fetch end date (ISO, default: today)",
    )
    p.add_argument(
        "--cache-dir", default="data/external", metavar="DIR",
        help="Cache directory (default: data/external)",
    )
    p.add_argument(
        "--ttl-hours", type=float, default=24.0, metavar="H",
        help="Cache TTL hours for staleness check (default: 24)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if a valid cache entry exists",
    )
    args = p.parse_args(argv)

    cache = ExternalCache(args.cache_dir)
    start = args.start
    end = args.end

    n_cached = 0
    n_fetched = 0
    n_failed = 0

    print()
    print("=" * 70)
    print("GDELT CACHE BUILDER")
    print("-" * 70)
    print(f"  Date range : {start} -> {end}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  Series     : {len(_NEWS_SERIES)}")
    print(f"  Force      : {'yes (ignoring existing cache)' if args.force else 'no'}")
    print(f"  TTL        : {args.ttl_hours:.0f} h")
    print("=" * 70)

    for cfg in _NEWS_SERIES:
        key = _cache_key(cfg, start, end)
        metric = cfg.extra.get("metric", "tone")

        # Check existing cache unless --force
        if not args.force:
            cached = cache.get(key, ttl_hours=args.ttl_hours)
            if cached is not None:
                n_valid = int(cached.notna().sum())
                n_total = len(cached)
                pct = 100 * n_valid / max(n_total, 1)
                print(
                    f"  [CACHED]  {cfg.name:<32}  "
                    f"valid={n_valid}/{n_total} ({pct:.1f}%)"
                )
                n_cached += 1
                continue

        # Fetch from GDELT
        print(f"  [FETCH ]  {cfg.name:<32}  metric={metric}")
        print(f"            query={cfg.symbol[:55]!r}")
        try:
            series = fetch_series(
                cfg.symbol,
                start=start,
                end=end,
                metric=metric,
                source_backend="gdelt",
            )
            cache.set(key, series)
            n_valid = int(series.notna().sum())
            n_total = len(series)
            pct = 100 * n_valid / max(n_total, 1)
            status = "[ok]" if pct > 80 else ("[LOW]" if pct > 0 else "[ZERO]")
            print(
                f"            {status} done  "
                f"valid={n_valid}/{n_total} ({pct:.1f}% coverage)"
            )
            n_fetched += 1
        except Exception as exc:
            print(f"            [FAIL] {type(exc).__name__}: {exc}")
            logger.error(
                "[build_gdelt_cache] '%s' failed: %s: %s",
                cfg.name, type(exc).__name__, exc,
            )
            n_failed += 1
        print()

    print("-" * 70)
    print("SUMMARY")
    print(f"  Series total : {len(_NEWS_SERIES)}")
    print(f"  Cache hits   : {n_cached}")
    print(f"  Newly fetched: {n_fetched}")
    print(f"  Failed       : {n_failed}")
    print("-" * 70)

    if n_failed > 0:
        print()
        print("  [!] Some series failed. Possible causes:")
        print("      - No internet access")
        print("      - GDELT API outage or rate limit (wait 60s and retry)")
        print("      - Query returned zero articles for date range")
        print("  Run again to retry failed series (cache hits skip successful ones).")

    if n_fetched > 0 or (n_cached == len(_NEWS_SERIES)):
        print()
        print("Coverage report (from cache):")
        for cfg in _NEWS_SERIES:
            key = _cache_key(cfg, start, end)
            series = cache.get(key, ttl_hours=args.ttl_hours * 100)
            if series is None:
                print(f"  [miss] {cfg.name:<32}  not in cache")
                continue
            n_v = int(series.notna().sum())
            n_t = len(series)
            pct = 100 * n_v / max(n_t, 1)
            flag = "[ok]  " if pct > 80 else ("[LOW]  " if pct > 0 else "[ZERO] ")
            print(f"  {flag}{cfg.name:<32}  {n_v}/{n_t} days  ({pct:.1f}%)")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
