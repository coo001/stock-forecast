"""Market data loader for Samsung Electronics.

Downloads OHLCV data via yfinance and caches it locally as CSV.
The loader is the only place that touches external APIs; everything
downstream consumes a plain ``pandas.DataFrame``.

Columns returned (after normalisation):
    date (index, DatetimeIndex), open, high, low, close, volume
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.data.schema import DataConfig

logger = logging.getLogger(__name__)


def load_ohlcv(cfg: DataConfig, *, force_download: bool = False) -> pd.DataFrame:
    """Return a cleaned OHLCV DataFrame for the configured ticker.

    Strategy:
        1. If a local cache CSV exists (and ``force_download`` is False), load it.
        2. Otherwise, download from Yahoo Finance via ``yfinance`` and cache.

    Args:
        cfg: DataConfig with ticker, dates, interval, cache_dir.
        force_download: Ignore cache and always re-download.

    Returns:
        DataFrame with DatetimeIndex and columns
        [open, high, low, close, volume], sorted ascending.

    Raises:
        RuntimeError: if yfinance is unavailable and no cache exists.
        ValueError: if the downloaded frame is empty.
    """
    cache_path = _cache_path(cfg)

    if cache_path.exists() and not force_download:
        logger.info("Loading from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _normalise(df)

    logger.info("Downloading %s from %s to %s", cfg.ticker, cfg.start_date, cfg.end_date or "today")
    df = _download(cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info("Cached to %s", cache_path)
    return df


def load_ohlcv_from_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from an arbitrary CSV file (for offline / test use).

    The CSV must have a date-parseable first column and at least the columns
    open, high, low, close, volume (case-insensitive).
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalise(df)


# ── Internals ─────────────────────────────────────────────────────────────────

def _download(cfg: DataConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed.\n\n"
            "Fix options:\n"
            "  1. Install yfinance:  pip install yfinance\n"
            "  2. Use synthetic data (no internet required):  python run_pipeline.py --synthetic\n"
        ) from exc

    raw = yf.download(
        cfg.ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{cfg.ticker}'")

    return _normalise(raw)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and index; drop rows with NaN close."""
    # yfinance returns MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)
    return df


def _cache_path(cfg: DataConfig) -> Path:
    safe_ticker = cfg.ticker.replace(".", "_").replace("/", "_")
    filename = f"{safe_ticker}_{cfg.start_date}_{cfg.interval}.csv"
    return Path(cfg.cache_dir) / filename
