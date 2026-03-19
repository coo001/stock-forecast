"""News sentiment and article-volume time series.

Two source backends are supported:

``gdelt`` (default — no API key, historical coverage from 2013)
    Queries the GDELT 2.0 Document API:
    https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    No registration required.  Rate-limited; requests are chunked by quarter.
    Returns daily tone scores or article counts.
    Coverage: English-language global media; Korean financial news may be
    underrepresented for Samsung-specific sentiment.

``newsapi`` (recent data only — requires NEWSAPI_KEY)
    Queries https://newsapi.org for article counts and headline sentiment.
    Free tier: last 30 days only. NOT suitable for historical backtesting.
    Useful for live / real-time pipeline operation.

Anti-leakage
------------
News published on day T is available to market participants on T.
Using ``lag_days=1`` (default) ensures that the feature at model row T
reflects only articles published on T−1 or earlier.

Fact vs. opinion policy
-----------------------
- News TONE is treated as a "market perception signal", not ground truth.
- It is stored in a separate feature group from fundamental data (DART).
- High tone variance or conflicting signals are not explicitly modelled
  at this stage; the count feature serves as a complementary uncertainty proxy.

Known limitations
-----------------
1. GDELT API is unofficial; structure may change without notice.
2. GDELT tone is computed from article-level CAMEO sentiment, which is
   designed for geopolitical events, not financial news.  It is a noisy
   but directionally useful proxy.
3. The keyword-based headline sentiment used for ``newsapi`` is a crude
   approximation.  For production, replace with a fine-tuned financial NLP
   model (e.g. FinBERT).
4. NewsAPI free tier: 30-day history limit.  All older dates → NaN.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_POLITE_DELAY_S = 0.6    # between GDELT chunk requests
_CHUNK_MONTHS = 3        # GDELT query window size

# ── Keyword-based headline sentiment (dependency-free) ───────────────────────
# Source: financial/semiconductor domain vocabulary
# These should be treated as heuristic approximations, not precise scores.
_POS_WORDS = frozenset({
    "surge", "jump", "beat", "record", "growth", "strong", "upgrade",
    "bullish", "gain", "rise", "win", "boost", "positive", "profit",
    "recovery", "expansion", "outperform", "rebound", "milestone",
    "contract", "order", "exceed", "robust", "rally", "advance",
    "breakout", "demand", "upside", "raised", "increases", "improves",
})
_NEG_WORDS = frozenset({
    "fall", "drop", "miss", "weak", "loss", "downgrade", "bearish",
    "decline", "concern", "risk", "negative", "cut", "warning",
    "slowdown", "oversupply", "inventory", "glut", "recall",
    "investigation", "penalty", "layoff", "restructure", "deficit",
    "disappointing", "headwind", "pressure", "competition", "weaker",
    "shortfall", "lower", "reduces", "slump", "plunge",
})


def fetch_series(
    query: str,
    start: str,
    end: str | None = None,
    *,
    api_key_env: str = "NEWSAPI_KEY",
    metric: str = "tone",
    source_backend: str = "gdelt",
) -> pd.Series:
    """Fetch daily news metric as a ``pd.Series``.

    Args:
        query: Search query string
               (e.g. ``"samsung electronics semiconductor"``).
        start: ISO fetch-start date.
        end: ISO fetch-end date. ``None`` → today.
        api_key_env: Env var for NewsAPI key (only used when
                     ``source_backend="newsapi"``).
        metric: Metric to return.
                ``"tone"``  — average daily tone/sentiment score.
                ``"count"`` — number of articles per day.
        source_backend: ``"gdelt"`` (historical, no key) or
                        ``"newsapi"`` (recent only, key required).

    Returns:
        Daily ``pd.Series`` (DatetimeIndex, start→end).
        NaN where no data is available.  Float dtype.

    Raises:
        RuntimeError: Network failure, missing API key (newsapi only),
                      or unsupported ``source_backend``.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc

    end_str = end or date.today().isoformat()

    if source_backend == "gdelt":
        return _fetch_gdelt(requests, query, start, end_str, metric)
    elif source_backend == "newsapi":
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"NewsAPI key not set. Set {api_key_env!r} environment variable.\n"
                "Free registration: https://newsapi.org/register\n"
                "NOTE: Free tier covers only the last 30 days of history. "
                "Use source_backend='gdelt' for historical backtesting."
            )
        return _fetch_newsapi(requests, query, start, end_str, api_key, metric)
    else:
        raise ValueError(
            f"Unsupported source_backend: {source_backend!r}. "
            "Use 'gdelt' or 'newsapi'."
        )


# ── GDELT backend ─────────────────────────────────────────────────────────────

def _fetch_gdelt(
    requests_module,
    query: str,
    start: str,
    end: str,
    metric: str,
) -> pd.Series:
    """Fetch from GDELT in quarterly chunks to avoid API timeouts."""
    date_range = pd.date_range(start=start, end=end, freq="D")
    result = pd.Series(float("nan"), index=date_range, dtype=float)
    result.index.name = "date"

    mode = "TimelineTone" if metric == "tone" else "TimelineVol"
    chunks = _date_chunks(start, end, months=_CHUNK_MONTHS)

    failed_chunks = 0
    for chunk_start, chunk_end in chunks:
        try:
            chunk_data = _gdelt_chunk(
                requests_module, query, chunk_start, chunk_end, mode
            )
            for ts, val in chunk_data.items():
                if ts in result.index:
                    result[ts] = val
            time.sleep(_POLITE_DELAY_S)
        except Exception as exc:
            failed_chunks += 1
            logger.warning(
                "[news_client] GDELT chunk %s–%s failed (%s): %s",
                chunk_start, chunk_end, type(exc).__name__, exc,
            )

    n_valid = int(result.notna().sum())
    logger.info(
        "[news_client] GDELT query=%r metric=%s: %d/%d valid days "
        "(%d chunks, %d failed)  (%s→%s)",
        query[:50], metric, n_valid, len(result),
        len(chunks), failed_chunks, start, end,
    )
    return result


def _gdelt_chunk(
    requests_module,
    query: str,
    start: str,
    end: str,
    mode: str,
) -> dict[pd.Timestamp, float]:
    """Query GDELT Timeline API for one date range chunk.

    Returns a dict mapping Timestamps → float values.
    """
    start_dt = datetime.strptime(start[:10], "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end[:10], "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "timeresolution": "day",
    }

    resp = requests_module.get(_GDELT_DOC_URL, params=params, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GDELT API HTTP {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    timeline = data.get("timeline", [])
    if not timeline:
        return {}

    # GDELT structure: [{"series": [{"date": "YYYYMMDDHHMMSS", "value": float}]}]
    series_data = (
        timeline[0].get("series", [])
        if isinstance(timeline[0], dict)
        else timeline
    )

    out: dict[pd.Timestamp, float] = {}
    for point in series_data:
        dt_str = point.get("date", "")
        val = point.get("value")
        if dt_str and val is not None:
            try:
                ts = pd.Timestamp(
                    f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
                )
                out[ts] = float(val)
            except (ValueError, TypeError):
                pass
    return out


# ── NewsAPI backend ───────────────────────────────────────────────────────────

def _fetch_newsapi(
    requests_module,
    query: str,
    start: str,
    end: str,
    api_key: str,
    metric: str,
) -> pd.Series:
    """Fetch from NewsAPI.org.

    Limitation: free tier provides only the last 30 days of history.
    Dates outside that window will have NaN values.
    """
    date_range = pd.date_range(start=start, end=end, freq="D")
    daily_scores: dict[pd.Timestamp, list[float]] = {d: [] for d in date_range}

    page = 1
    while True:
        params = {
            "q": query,
            "from": start[:10],
            "to": end[:10],
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "page": page,
            "apiKey": api_key,
        }

        try:
            resp = requests_module.get(_NEWSAPI_URL, params=params, timeout=30)
        except Exception as exc:
            raise RuntimeError(f"NewsAPI request failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"NewsAPI error: status={resp.status_code}  body={resp.text[:200]}"
            )

        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(
                f"NewsAPI error: {data.get('message', data)}"
            )

        articles = data.get("articles", [])
        if not articles:
            break

        for article in articles:
            pub_at = article.get("publishedAt", "")
            title = (article.get("title") or "")
            description = (article.get("description") or "")
            if pub_at:
                ts = pd.Timestamp(pub_at[:10])
                if ts in daily_scores:
                    score = _headline_sentiment(f"{title} {description}")
                    daily_scores[ts].append(score)

        total = data.get("totalResults", 0)
        # NewsAPI free tier caps retrievable articles at ~100 pages
        if page * 100 >= min(total, 1000):
            break
        page += 1
        time.sleep(0.3)

    # Aggregate to daily series
    result = pd.Series(float("nan"), index=date_range, dtype=float)
    result.index.name = "date"

    for ts, scores in daily_scores.items():
        if not scores:
            continue
        if metric == "count":
            result[ts] = float(len(scores))
        else:  # tone / sentiment
            result[ts] = sum(scores) / len(scores)

    n_valid = int(result.notna().sum())
    logger.info(
        "[news_client] NewsAPI query=%r metric=%s: %d days with data  (%s→%s)",
        query[:50], metric, n_valid, start, end,
    )
    return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def _headline_sentiment(text: str) -> float:
    """Keyword-based financial sentiment score in [-1.0, 1.0].

    This is a fast, zero-dependency heuristic intended for feature engineering.
    It counts positive and negative financial/semiconductor keywords in the text.

    Limitations:
        - Ignores negation ("not strong" scores positive)
        - Domain vocabulary may miss nuanced financial language
        - Should be replaced with a fine-tuned financial NLP model in production
          (e.g. FinBERT, KoFinBERT for Korean text)
    """
    if not text:
        return 0.0
    words = set(text.lower().split())
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _date_chunks(
    start: str, end: str, months: int = 3
) -> list[tuple[str, str]]:
    """Split a date range into non-overlapping chunks of *months* months."""
    chunks: list[tuple[str, str]] = []
    cur = datetime.strptime(start[:10], "%Y-%m-%d")
    end_dt = datetime.strptime(end[:10], "%Y-%m-%d")

    while cur <= end_dt:
        next_month = cur.month + months
        next_year = cur.year + (next_month - 1) // 12
        next_month = (next_month - 1) % 12 + 1
        chunk_end = min(
            datetime(next_year, next_month, 1) - timedelta(days=1),
            end_dt,
        )
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)

    return chunks
