"""Merge external macro/market features into the Samsung OHLCV DataFrame.

Leakage-safe alignment pipeline
================================
For each external series the following steps are applied in order::

    raw series  (arbitrary calendar, any frequency)
        1. reindex to Samsung trading dates  (forward-fill gaps)
        2. apply feature_type transformation (level / diff / pct_change / log_return)
        3. .shift(lag_days)          ← the primary anti-leakage guard
        4. rename to "ext_{name}"

After step 3, the feature at Samsung date T uses only data from
T - lag_days (or earlier).

Leakage proof
-------------
Let ``ext[T]`` be the merged feature at Samsung trading date T, and let
``raw_reindexed`` be the source series after step 1.

For ``feature_type = "level"``::

    ext[T] = raw_reindexed[T - lag_days]     (last known value ≤ T - lag_days)

For ``feature_type = "log_return"``::

    log_ret_aligned[T]  = log(raw_reindexed[T] / raw_reindexed[T-1])
    ext[T]              = log_ret_aligned[T - lag_days]
                        = log(raw_reindexed[T - lag_days] /
                              raw_reindexed[T - lag_days - 1])

In both cases, no value from T or later enters the feature.  This holds
as long as ``lag_days >= 1``, which is enforced by ``ExternalSeriesConfig``.

Warmup NaN
----------
The first ``lag_days`` rows of every merged column are NaN.  For
``log_return`` / ``diff`` / ``pct_change`` the first row is also NaN due to
the differencing.  Callers should call ``dropna()`` after merging (or rely on
``build_feature_matrix``'s own ``dropna``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.external.cache import ExternalCache

logger = logging.getLogger(__name__)

_VALID_FEATURE_TYPES = frozenset({"level", "diff", "pct_change", "log_return"})
_VALID_SOURCES = frozenset({"market", "fred", "ecos", "alpha_vantage"})


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExternalSeriesConfig:
    """Configuration for one external time series.

    Attributes:
        name: Short identifier; the merged column will be ``"ext_{name}"``.
        source: Data source — ``"market"`` (yfinance), ``"fred"``, ``"ecos"``,
                or ``"alpha_vantage"`` (stub).
        symbol: Source-specific identifier:
                market → Yahoo Finance ticker (``"^KS11"``)
                fred   → FRED series ID (``"GS10"``)
                ecos   → ECOS stat_code (``"722Y001"``)
        lag_days: Samsung trading days to shift after alignment. **Must be >= 1.**
                  Default 1 prevents same-day leakage for daily series.
                  Use >= 5 for monthly data to account for announcement delays.
        feature_type: Transformation applied to the aligned daily series.
                      ``"level"``      — raw numeric value (e.g. yield %, VIX level)
                      ``"diff"``       — first difference
                      ``"pct_change"`` — percentage change  (value - prev) / prev
                      ``"log_return"`` — natural log of the ratio  log(val / prev)
        frequency: Native data frequency: ``"daily"``, ``"weekly"``, ``"monthly"``.
                   Forward-fill handles all frequencies uniformly; this field is
                   informational only (used for logging).
        api_key_env: Name of the env var holding the API key.
                     Empty string means no key is required (e.g. yfinance).
        extra: Source-specific extra parameters passed to the client function.
               ECOS: ``{"item_code": "0101000", "cycle": "MM"}``
    """

    name: str
    source: str
    symbol: str
    lag_days: int = 1
    feature_type: str = "level"
    frequency: str = "daily"
    api_key_env: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lag_days < 1:
            raise ValueError(
                f"ExternalSeriesConfig '{self.name}': lag_days must be >= 1 "
                f"(got {self.lag_days}).  lag_days=0 risks look-ahead leakage."
            )
        if self.feature_type not in _VALID_FEATURE_TYPES:
            raise ValueError(
                f"ExternalSeriesConfig '{self.name}': feature_type must be one of "
                f"{sorted(_VALID_FEATURE_TYPES)}, got '{self.feature_type}'."
            )
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"ExternalSeriesConfig '{self.name}': source must be one of "
                f"{sorted(_VALID_SOURCES)}, got '{self.source}'."
            )


# ── Public functions ──────────────────────────────────────────────────────────

def load_external_series(
    cfg: ExternalSeriesConfig,
    start: str,
    end: str,
    *,
    cache: ExternalCache,
    cache_ttl_hours: float = 24.0,
) -> pd.Series:
    """Fetch one external series, using the local cache when available.

    Args:
        cfg: Series configuration.
        start: Earliest fetch date (ISO string). Should precede ``ohlcv.index[0]``
               so forward-fill has a value available from the first Samsung date.
        end: Latest fetch date (ISO string).
        cache: ExternalCache instance for read/write.
        cache_ttl_hours: Max cache age before a fresh download is triggered.

    Returns:
        Raw ``pd.Series`` with DatetimeIndex (not yet aligned, lagged, or
        transformed).
    """
    cache_key = f"{cfg.source}__{cfg.symbol}__{start}__{end}"

    cached = cache.get(cache_key, ttl_hours=cache_ttl_hours)
    if cached is not None:
        return cached

    series = _fetch_from_source(cfg, start, end)
    cache.set(cache_key, series)
    return series


def merge_external_features(
    ohlcv: pd.DataFrame,
    series_configs: list[ExternalSeriesConfig],
    *,
    start: str,
    end: str,
    cache_dir: str = "data/external",
    cache_ttl_hours: float = 24.0,
) -> pd.DataFrame:
    """Merge external macro/market features into the OHLCV DataFrame.

    Each series is fetched (or loaded from cache), reindexed to Samsung trading
    dates, transformed, and shifted by ``lag_days`` before being added as a
    new column.  All operations on a series are independent, so partial
    failures do not abort the merge — a warning is logged instead.

    Args:
        ohlcv: Samsung OHLCV DataFrame with DatetimeIndex (trading days).
        series_configs: List of series to merge. Empty list → no-op.
        start: Fetch start date (ISO string).  Pass a date *before*
               ``ohlcv.index[0]`` to ensure forward-fill has data from day 1.
        end: Fetch end date (ISO string).
        cache_dir: Directory for cached CSVs.
        cache_ttl_hours: Max cache age before re-fetching.

    Returns:
        Copy of *ohlcv* with additional ``"ext_*"`` columns.  The index is
        unchanged.  Failed series produce no column (silent failure with log).

    Raises:
        No exceptions propagate — each series failure is caught and logged.
    """
    if not series_configs:
        return ohlcv.copy()

    cache = ExternalCache(cache_dir)
    target_index = ohlcv.index
    result = ohlcv.copy()

    loaded: list[str] = []
    failed: list[str] = []

    for cfg in series_configs:
        col_name = f"ext_{cfg.name}"
        try:
            raw = load_external_series(
                cfg, start=start, end=end,
                cache=cache, cache_ttl_hours=cache_ttl_hours,
            )
            aligned = _align_series(raw, target_index)
            transformed = _apply_feature_type(aligned, cfg.feature_type, cfg.name)
            lagged = transformed.shift(cfg.lag_days)

            result[col_name] = lagged
            loaded.append(col_name)
            logger.info(
                "[external_merge] '%s'  source=%s  freq=%s  lag=%d  type=%s"
                "  non-null=%d / %d",
                col_name, cfg.source, cfg.frequency, cfg.lag_days,
                cfg.feature_type, lagged.notna().sum(), len(lagged),
            )
        except Exception as exc:
            logger.warning(
                "[external_merge] failed to load '%s' (source=%s symbol=%s): "
                "%s: %s",
                cfg.name, cfg.source, cfg.symbol, type(exc).__name__, exc,
            )
            failed.append(cfg.name)

    if loaded:
        logger.info("[external_merge] merged %d series: %s", len(loaded), loaded)
    if failed:
        logger.warning("[external_merge] %d series failed: %s", len(failed), failed)

    return result


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_from_source(cfg: ExternalSeriesConfig, start: str, end: str) -> pd.Series:
    """Dispatch to the correct client based on ``cfg.source``."""
    if cfg.source == "market":
        from src.data.external.market_client import fetch_series
        return fetch_series(cfg.symbol, start=start, end=end)

    elif cfg.source == "fred":
        from src.data.external.fred_client import fetch_series
        return fetch_series(
            cfg.symbol, start, end,
            api_key_env=cfg.api_key_env or "FRED_API_KEY",
        )

    elif cfg.source == "ecos":
        from src.data.external.ecos_client import fetch_series
        return fetch_series(
            cfg.symbol,
            item_code=cfg.extra.get("item_code", ""),
            cycle=cfg.extra.get("cycle", "MM"),
            start=start,
            end=end,
            api_key_env=cfg.api_key_env or "ECOS_API_KEY",
        )

    elif cfg.source == "alpha_vantage":
        from src.data.external.alpha_vantage_client import fetch_series
        return fetch_series(
            cfg.extra.get("function", "TIME_SERIES_DAILY"),
            cfg.symbol,
            api_key_env=cfg.api_key_env or "ALPHA_VANTAGE_API_KEY",
        )

    raise ValueError(f"Unknown source: '{cfg.source}'")


def _align_series(raw: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex *raw* to *target_index* using forward-fill for gaps.

    Forward-fill propagates the last known value across weekends, Korean
    public holidays, and any other dates absent from the source.  This is
    the correct treatment for macro series (e.g. a monthly rate stays constant
    between announcements).

    The union of the raw and target indices is formed first so that forward-
    fill carries data from before the first Samsung trading date.
    """
    combined = raw.index.union(target_index).sort_values()
    reindexed = raw.reindex(combined).ffill()
    return reindexed.reindex(target_index)


def _apply_feature_type(series: pd.Series, feature_type: str, name: str) -> pd.Series:
    """Apply the specified transformation to the aligned series.

    All transformations produce NaN at the first row (no prior value to
    compare against), except ``"level"`` which returns the series unchanged.
    """
    if feature_type == "level":
        return series

    elif feature_type == "diff":
        return series.diff()

    elif feature_type == "pct_change":
        return series.pct_change()

    elif feature_type == "log_return":
        prev = series.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(series / prev)
        return pd.Series(log_ret, index=series.index, name=series.name)

    raise ValueError(f"Unknown feature_type '{feature_type}' for series '{name}'.")
