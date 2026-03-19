"""Bank of Korea ECOS (Economic Statistics System) REST client.

Fetches Korean macro statistics from the ECOS Open API.
API reference: https://ecos.bok.or.kr/api/

Commonly used codes
-------------------
Stat code : 722Y001   Bank of Korea Base Rate (기준금리)
  item_code : 0101000
  cycle     : MM  (monthly)

Stat code : 901Y009   Consumer Price Index (소비자물가지수)
  item_code : 0   (all items)
  cycle     : MM

Free API key registration: https://ecos.bok.or.kr/#/
Set env var:  ECOS_API_KEY=your_key_here
"""
from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"


def fetch_series(
    stat_code: str,
    *,
    item_code: str,
    cycle: str = "MM",
    start: str,
    end: str | None = None,
    api_key_env: str = "ECOS_API_KEY",
) -> pd.Series:
    """Fetch a Bank of Korea ECOS macro series.

    Args:
        stat_code: ECOS statistical table code (e.g. ``"722Y001"``).
        item_code: Indicator code within the table (e.g. ``"0101000"``).
        cycle: Frequency: ``"DD"`` daily, ``"MM"`` monthly, ``"QQ"`` quarterly.
        start: Period start — ISO date string; the YYYYMM portion is extracted
               for monthly requests (e.g. ``"2015-01-01"`` → ``"201501"``).
        end: Period end. ``None`` → current month.
        api_key_env: Env var holding the ECOS API key.

    Returns:
        ``pd.Series`` with DatetimeIndex (first day of each period) and float
        values.  Index name is ``"date"``.

    Raises:
        RuntimeError: if the API key env var is empty, or the request fails.
        ValueError: if the series is empty.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests package is required for ECOS data.\n"
            "Install with: pip install requests"
        ) from exc

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"ECOS API key not set. Set the {api_key_env!r} environment variable.\n"
            "Get a free key at: https://ecos.bok.or.kr/#/"
        )

    start_period = _to_ecos_period(start, cycle)
    end_period = _to_ecos_period(end or date.today().isoformat(), cycle)

    url = (
        f"{_BASE_URL}/{api_key}/json/kr/1/1000"
        f"/{stat_code}/{cycle}/{start_period}/{end_period}/{item_code}"
    )

    try:
        resp = requests.get(url, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"ECOS HTTP request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"ECOS API error: status={resp.status_code}  body={resp.text[:300]}"
        )

    data = resp.json()
    if "StatisticSearch" not in data:
        err_info = data.get("RESULT", data)
        raise RuntimeError(f"ECOS API returned an error response: {err_info}")

    rows = data["StatisticSearch"].get("row", [])
    if not rows:
        raise ValueError(
            f"ECOS series {stat_code}/{item_code} returned no rows "
            f"(start={start_period}, end={end_period})."
        )

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows:
        period = row.get("TIME", "")
        val_str = row.get("DATA_VALUE", "")
        if not val_str or val_str in ("-", ""):
            continue
        try:
            ts = _parse_period(period, cycle)
            values.append(float(val_str))
            dates.append(ts)
        except (ValueError, IndexError):
            continue

    if not values:
        raise ValueError(
            f"ECOS series {stat_code}/{item_code} has no numeric values in range."
        )

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=f"{stat_code}_{item_code}")
    series.index.name = "date"
    series = series.sort_index()

    logger.info(
        "[ecos_client] %s/%s: %d rows  (%s → %s)",
        stat_code, item_code, len(series),
        series.index[0].date(), series.index[-1].date(),
    )
    return series


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_ecos_period(iso_date: str, cycle: str) -> str:
    """Convert an ISO date string to the ECOS period format.

    Examples:
        "2015-01-15", "MM" → "201501"
        "2015-01-15", "DD" → "20150115"
        "2015-01-15", "QQ" → "20151"
    """
    clean = iso_date[:10].replace("-", "")  # "20150115"
    if cycle == "MM":
        return clean[:6]   # "201501"
    elif cycle == "QQ":
        month = int(clean[4:6])
        quarter = (month - 1) // 3 + 1
        return f"{clean[:4]}{quarter}"
    else:  # DD or annual
        return clean


def _parse_period(period: str, cycle: str) -> pd.Timestamp:
    """Parse an ECOS period string to the first day of that period."""
    if cycle == "MM" and len(period) == 6:
        return pd.Timestamp(f"{period[:4]}-{period[4:6]}-01")
    elif cycle == "QQ" and len(period) == 5:
        quarter = int(period[4])
        month = (quarter - 1) * 3 + 1
        return pd.Timestamp(f"{period[:4]}-{month:02d}-01")
    elif cycle == "DD" and len(period) == 8:
        return pd.Timestamp(f"{period[:4]}-{period[4:6]}-{period[6:8]}")
    else:
        return pd.Timestamp(period)
