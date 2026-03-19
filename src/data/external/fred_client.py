"""FRED (Federal Reserve Economic Data) REST client.

Fetches macro time-series directly from the FRED API using only the
standard ``requests`` package — no ``fredapi`` dependency required.

API reference: https://fred.stlouisfed.org/docs/api/fred/

Commonly used series IDs
------------------------
GS10     US 10-Year Treasury Constant Maturity Rate (%, daily)
VIXCLS   CBOE Volatility Index: VIX (daily)
DFF      Federal Funds Effective Rate (%, daily)
CPIAUCSL US CPI All Items (monthly, index)

Free API key registration: https://fred.stlouisfed.org/docs/api/api_key.html
Set env var:  FRED_API_KEY=your_key_here
"""
from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(
    series_id: str,
    start: str,
    end: str | None = None,
    *,
    api_key_env: str = "FRED_API_KEY",
) -> pd.Series:
    """Fetch a FRED series and return as a ``pd.Series``.

    Missing observations (reported as ``"."`` by FRED) are silently dropped.

    Args:
        series_id: FRED series identifier (e.g. ``"GS10"``).
        start: Observation start date (ISO string, inclusive).
        end: Observation end date (ISO string, inclusive). ``None`` → today.
        api_key_env: Name of the environment variable holding the FRED API key.

    Returns:
        ``pd.Series`` with DatetimeIndex and float values.
        Index name is ``"date"``.

    Raises:
        RuntimeError: if the API key env var is empty, or the HTTP request fails.
        ValueError: if the series contains no numeric observations.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests package is required for FRED data.\n"
            "Install with: pip install requests"
        ) from exc

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"FRED API key not set. Set the {api_key_env!r} environment variable.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    end_str = end or date.today().isoformat()
    params = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end_str,
        "api_key": api_key,
        "file_type": "json",
    }

    resp = requests.get(_BASE_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"FRED API request failed: status={resp.status_code}  "
            f"body={resp.text[:300]}"
        )

    observations = resp.json().get("observations", [])
    if not observations:
        raise ValueError(f"FRED returned no observations for series '{series_id}'.")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        val_str = obs.get("value", ".")
        if val_str == ".":
            continue
        try:
            values.append(float(val_str))
            dates.append(pd.Timestamp(obs["date"]))
        except (ValueError, KeyError):
            continue

    if not values:
        raise ValueError(
            f"FRED series '{series_id}' has no numeric observations "
            f"between {start} and {end_str}."
        )

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
    series.index.name = "date"
    series = series.sort_index()

    logger.info(
        "[fred_client] %s: %d rows  (%s → %s)",
        series_id, len(series), series.index[0].date(), series.index[-1].date(),
    )
    return series
