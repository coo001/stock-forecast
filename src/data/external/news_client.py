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
3. The keyword-based headline metrics used for ``newsapi`` are heuristic
   approximations.  For production, replace with a fine-tuned financial NLP
   model (e.g. FinBERT).
4. NewsAPI free tier: 30-day history limit.  All older dates → NaN.
5. Uncertainty / event scores are measured by keyword presence, not
   syntactic understanding.  Negation is NOT modelled.

Why these metrics, and their limitations
-----------------------------------------
``tone``        GDELT CAMEO tone: noisy geopolitical proxy.  Use with caution
                for financial prediction.  Baseline feature only.
``count``       Raw article volume. Captures attention spikes but not valence.
``pos_score``   Fraction of positive-sentiment keywords in headlines.
                Ignores negation ("not strong" → positive).
``neg_score``   Fraction of negative-sentiment keywords. Same negation caveat.
``uncertainty`` Fraction of uncertainty/hedging keywords.  Correlates with
                event risk and analyst disagreement.
``event_earnings`` Binary flag: 1.0 if any article mentions earnings-event
                   keywords on that day. Useful as a regime indicator.
``event_supply``   Binary flag: supply-chain / capacity event indicator.
``source_diversity`` Count of distinct media outlets covering the query.
                     Higher diversity → more consensus signal.

All NewsAPI metrics are computed from headline + description text only;
full article body is not available on the free tier.
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
_POLITE_DELAY_S = 5.5    # GDELT enforces ≤ 1 req/5 s per IP
_POLITE_DELAY_429 = 15.0 # extra back-off on HTTP 429
_CHUNK_MONTHS = 3        # GDELT query window size

# ── Financial/semiconductor vocabulary ────────────────────────────────────────
# Source: financial press + semiconductor industry vocabulary
# All scores ignore negation — heuristic quality only.

_POS_WORDS = frozenset({
    # Price / earnings positives
    "surge", "jump", "beat", "record", "growth", "strong", "upgrade",
    "bullish", "gain", "rise", "win", "boost", "positive", "profit",
    "recovery", "expansion", "outperform", "rebound", "milestone",
    "contract", "order", "exceed", "robust", "rally", "advance",
    "breakout", "demand", "upside", "raised", "increases", "improves",
    # Semiconductor / tech positives
    "ramp", "launch", "breakthrough", "innovation", "adoption",
    "shipment", "design-win", "backlog", "inventory-lean", "recovery",
    "upgrade", "qualification", "mass-production", "yield-improvement",
    "hbm", "ai-chip", "server", "datacenter",
})

_NEG_WORDS = frozenset({
    # Price / earnings negatives
    "fall", "drop", "miss", "weak", "loss", "downgrade", "bearish",
    "decline", "concern", "risk", "negative", "cut", "warning",
    "slowdown", "oversupply", "inventory", "glut", "recall",
    "investigation", "penalty", "layoff", "restructure", "deficit",
    "disappointing", "headwind", "pressure", "competition", "weaker",
    "shortfall", "lower", "reduces", "slump", "plunge",
    # Semiconductor / tech negatives
    "oversupply", "price-decline", "margin-squeeze", "capex-cut",
    "fab-closure", "inventory-buildup", "demand-weakness", "tariff",
    "sanction", "export-control", "ban", "restriction", "delay",
})

_UNCERTAINTY_WORDS = frozenset({
    # Hedging / ambiguity language in analyst / news reports
    "uncertain", "unclear", "volatile", "risk", "concern", "worry",
    "ambiguous", "mixed", "diverge", "conflicting", "unpredictable",
    "cautious", "watchful", "monitor", "wait-and-see", "depends",
    "conditional", "subject-to", "pending", "await", "assess",
    "variable", "fluctuating", "unstable", "unresolved", "dispute",
})

_EVENT_EARNINGS_WORDS = frozenset({
    # Earnings / guidance / analyst coverage events
    "earnings", "revenue", "profit", "eps", "quarterly", "annual",
    "guidance", "forecast", "outlook", "beat", "miss", "report",
    "results", "financial", "income", "operating", "margin",
    "dividend", "buyback", "analyst", "estimate", "consensus",
    "target", "rating", "upgrade", "downgrade", "preview", "release",
})

_EVENT_SUPPLY_WORDS = frozenset({
    # Supply chain, capacity, production events
    "supply", "demand", "inventory", "shortage", "oversupply", "glut",
    "production", "capacity", "capex", "expansion", "cutback",
    "fab", "wafer", "yield", "chip", "dram", "nand", "hbm",
    "foundry", "packaging", "node", "nm", "process", "ramping",
    "qualification", "allocation", "lead-time", "order-cut",
    "tariff", "sanction", "export-control", "restriction",
})

# Supported metric types
_GDELT_METRICS = frozenset({"tone", "count"})
_NEWSAPI_METRICS = frozenset({
    "tone",            # average (pos - neg) / total, range [-1, 1]
    "count",           # article count
    "pos_score",       # mean fraction of positive keywords
    "neg_score",       # mean fraction of negative keywords
    "uncertainty",     # mean fraction of uncertainty keywords
    "event_earnings",  # fraction of articles with earnings keywords
    "event_supply",    # fraction of articles with supply-chain keywords
    "source_diversity",# count of distinct media outlets per day
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
               For GDELT, this is passed directly to the API.
               Use quoted phrases for exact matching: ``'"Samsung Electronics"'``.
        start: ISO fetch-start date.
        end: ISO fetch-end date. ``None`` → today.
        api_key_env: Env var for NewsAPI key (only used when
                     ``source_backend="newsapi"``).
        metric: Metric to return.
                GDELT backend: ``"tone"`` or ``"count"``.
                NewsAPI backend: ``"tone"``, ``"count"``, ``"pos_score"``,
                ``"neg_score"``, ``"uncertainty"``, ``"event_earnings"``,
                ``"event_supply"``, ``"source_diversity"``.
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
        if metric not in _GDELT_METRICS:
            raise ValueError(
                f"GDELT backend supports metrics: {sorted(_GDELT_METRICS)}.  "
                f"Got '{metric}'.  For pos_score/neg_score/uncertainty/event_* "
                f"use source_backend='newsapi'."
            )
        return _fetch_gdelt(requests, query, start, end_str, metric)

    elif source_backend == "newsapi":
        if metric not in _NEWSAPI_METRICS:
            raise ValueError(
                f"NewsAPI backend supports metrics: {sorted(_NEWSAPI_METRICS)}.  "
                f"Got '{metric}'."
            )
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
    """Fetch from GDELT in quarterly chunks to avoid API timeouts.

    GDELT tone note
    ---------------
    GDELT CAMEO tone is computed on geopolitical event articles.  For
    financial news the signal is noisy.  Typical range: roughly -10 to +10;
    positive = more favourable framing, negative = more conflict framing.
    This is a directional proxy, not a precise financial sentiment score.

    GDELT requires internet access.  In --synthetic mode (synthetic price data
    but real dates), GDELT calls still reach out to the real API.  Without
    internet, all chunks fail silently and the series is all-NaN → the column
    is dropped by the pipeline.
    """
    date_range = pd.date_range(start=start, end=end, freq="D")
    result = pd.Series(float("nan"), index=date_range, dtype=float)
    result.index.name = "date"

    mode = "TimelineTone" if metric == "tone" else "TimelineVol"
    chunks = _date_chunks(start, end, months=_CHUNK_MONTHS)

    failed_chunks = 0
    for chunk_start, chunk_end in chunks:
        # One retry on 429; otherwise attempt once.
        for attempt in range(2):
            try:
                chunk_data = _gdelt_chunk(
                    requests_module, query, chunk_start, chunk_end, mode
                )
                for ts, val in chunk_data.items():
                    if ts in result.index:
                        result[ts] = val
                time.sleep(_POLITE_DELAY_S)
                break  # success
            except RuntimeError as exc:
                if "429" in str(exc) and attempt == 0:
                    # Already slept _POLITE_DELAY_429 inside _gdelt_chunk; retry.
                    logger.info(
                        "[news_client] GDELT 429 on chunk %s–%s; retrying …",
                        chunk_start, chunk_end,
                    )
                    continue
                failed_chunks += 1
                logger.warning(
                    "[news_client] GDELT chunk %s–%s failed (%s): %s",
                    chunk_start, chunk_end, type(exc).__name__, exc,
                )
                break
            except Exception as exc:
                failed_chunks += 1
                logger.warning(
                    "[news_client] GDELT chunk %s–%s failed (%s): %s",
                    chunk_start, chunk_end, type(exc).__name__, exc,
                )
                break

    n_valid = int(result.notna().sum())
    logger.info(
        "[news_client] GDELT query=%r metric=%s: %d/%d valid days "
        "(%d chunks, %d failed)  (%s→%s)",
        query[:50], metric, n_valid, len(result),
        len(chunks), failed_chunks, start, end,
    )
    if n_valid == 0:
        logger.warning(
            "[news_client] GDELT returned 0 valid days for query=%r metric=%s. "
            "Possible causes: no internet access, query returns no results, "
            "or GDELT API outage.  Series will be all-NaN → column will be dropped.",
            query[:50], metric,
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

    if resp.status_code == 429:
        # Rate-limited: back off and raise so caller can retry or skip
        time.sleep(_POLITE_DELAY_429)
        raise RuntimeError(
            f"GDELT API HTTP 429: {resp.text[:200]}"
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"GDELT API HTTP {resp.status_code}: {resp.text[:200]}"
        )

    # Guard: GDELT occasionally returns empty body (no articles / transient issue)
    body = resp.text.strip()
    if not body:
        return {}

    try:
        data = resp.json()
    except Exception:
        return {}   # non-JSON response (HTML error page etc.)

    if not isinstance(data, dict):
        return {}

    timeline = data.get("timeline", [])
    if not timeline:
        return {}

    # GDELT structure: [{"series": [{"date": "YYYYMMDDHHMMSS", "value": float}]}]
    first = timeline[0]
    if isinstance(first, dict):
        # Normal: {"series": [...], "name": "All Articles"}
        series_data = first.get("series", [])
        # Fallback: point dicts live directly in timeline (flattened structure)
        if not series_data and "date" in first:
            series_data = timeline
    else:
        # Unexpected structure — skip this chunk
        return {}

    out: dict[pd.Timestamp, float] = {}
    for point in series_data:
        if not isinstance(point, dict):
            continue   # skip strings / non-dict items
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

    Available metrics
    -----------------
    tone            : mean (pos - neg) / (pos + neg) per day.  Range [-1, 1].
    count           : article count per day.
    pos_score       : mean fraction of pos-keyword words.  Range [0, 1].
    neg_score       : mean fraction of neg-keyword words.  Range [0, 1].
    uncertainty     : mean fraction of uncertainty words.  Range [0, 1].
    event_earnings  : fraction of articles containing earnings keywords.
    event_supply    : fraction of articles containing supply-chain keywords.
    source_diversity: count of distinct media sources (domain names).
    """
    date_range = pd.date_range(start=start, end=end, freq="D")
    # Per-day accumulator:  date → list of per-article scores
    daily: dict[pd.Timestamp, list[float]] = {d: [] for d in date_range}
    # For source_diversity: date → set of source names
    daily_sources: dict[pd.Timestamp, set[str]] = {d: set() for d in date_range}

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
            source_name = (
                (article.get("source") or {}).get("name") or
                (article.get("source") or {}).get("id") or
                "unknown"
            )

            if not pub_at:
                continue

            ts = pd.Timestamp(pub_at[:10])
            if ts not in daily:
                continue

            text = f"{title} {description}"

            # Accumulate per-article score for the requested metric
            score = _compute_article_score(text, metric)
            if score is not None:
                daily[ts].append(score)

            # Always track source diversity
            daily_sources[ts].add(source_name)

        total = data.get("totalResults", 0)
        if page * 100 >= min(total, 1000):
            break
        page += 1
        time.sleep(0.3)

    # Aggregate to daily series
    result = pd.Series(float("nan"), index=date_range, dtype=float)
    result.index.name = "date"

    for ts in date_range:
        scores = daily[ts]
        if metric == "source_diversity":
            # Special case: count of unique sources for this day
            n_sources = len(daily_sources[ts])
            if n_sources > 0:
                result[ts] = float(n_sources)
        elif scores:
            if metric == "count":
                result[ts] = float(len(scores))
            else:
                result[ts] = sum(scores) / len(scores)

    n_valid = int(result.notna().sum())
    logger.info(
        "[news_client] NewsAPI query=%r metric=%s: %d days with data  (%s→%s)",
        query[:50], metric, n_valid, start, end,
    )
    return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def _compute_article_score(text: str, metric: str) -> float | None:
    """Compute a single per-article score for the given metric.

    Returns None for metrics that are aggregated differently (e.g. count,
    source_diversity are handled at the day-level, not article-level).

    All text-based scores ignore negation — heuristic quality only.
    """
    if not text:
        return 0.0 if metric not in ("count", "source_diversity") else None

    if metric == "count":
        return 1.0  # each article contributes 1

    words = set(text.lower().split())

    if metric in ("tone", None):
        pos = len(words & _POS_WORDS)
        neg = len(words & _NEG_WORDS)
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0

    elif metric == "pos_score":
        total = len(words)
        return len(words & _POS_WORDS) / total if total > 0 else 0.0

    elif metric == "neg_score":
        total = len(words)
        return len(words & _NEG_WORDS) / total if total > 0 else 0.0

    elif metric == "uncertainty":
        total = len(words)
        return len(words & _UNCERTAINTY_WORDS) / total if total > 0 else 0.0

    elif metric == "event_earnings":
        # Binary: 1.0 if any earnings keyword present
        return 1.0 if (words & _EVENT_EARNINGS_WORDS) else 0.0

    elif metric == "event_supply":
        return 1.0 if (words & _EVENT_SUPPLY_WORDS) else 0.0

    elif metric == "source_diversity":
        return None  # handled at day-level

    return 0.0


def _headline_sentiment(text: str) -> float:
    """Keyword-based financial sentiment score in [-1.0, 1.0].

    Backward-compatible wrapper around _compute_article_score with metric="tone".

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
