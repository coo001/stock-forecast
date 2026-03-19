"""DART (Data Analysis, Retrieval and Transfer System) OpenAPI client.

Fetches Samsung Electronics official disclosure events from the Korean
Financial Supervisory Service (금융감독원) DART system and returns them
as a daily count series suitable for feature engineering.

API reference : https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001
Free key      : https://opendart.fss.or.kr/intro/main.do
Set env var   : DART_API_KEY=<your_key>

Samsung Electronics corp_code: 00126380

Disclosure type codes (pblntf_ty)
----------------------------------
A  정기공시   — quarterly (분기), semi-annual (반기), annual (사업) earnings filings
B  주요사항   — material event disclosures (capex, business transfers, management)
F  공정공시   — fair disclosure: guidance, earnings previews, forward-looking statements
""  (empty)  — all types combined

Anti-leakage design
-------------------
Each DART filing has a receipt date (``rcept_dt``).  A filing received on date T
becomes public on T.  The merge pipeline applies ``shift(lag_days >= 1)``, so:

    ext_dart_X[T] = count_of_filings_on(T − lag_days)

Using ``lag_days=1`` is conservative: we treat knowledge of a filing as
available from the next trading day, avoiding any intraday ambiguity.

Fact vs. opinion separation
----------------------------
``pblntf_ty="A"`` → quarterly/annual filings   : hard facts (audited/reviewed)
``pblntf_ty="F"`` → guidance / fair disclosures: forward-looking, treat as signal,
                    NOT as ground truth.  Stored separately from "A" features.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

SAMSUNG_CORP_CODE = "00126380"

_BASE_URL = "https://opendart.fss.or.kr/api/list.json"
_PAGE_SIZE = 100
_POLITE_DELAY_S = 0.25   # between paginated requests (DART rate limit)


def fetch_series(
    corp_code: str,
    start: str,
    end: str | None = None,
    *,
    api_key_env: str = "DART_API_KEY",
    pblntf_ty: str = "",
) -> pd.Series:
    """Return daily disclosure-count series as ``pd.Series``.

    Args:
        corp_code: DART corporation code (Samsung = ``"00126380"``).
        start: ISO fetch-start date (e.g. ``"2015-01-01"``).
        end: ISO fetch-end date. ``None`` → today.
        api_key_env: Env var name that holds the DART API key.
        pblntf_ty: Disclosure type filter.
                   ``"A"`` → 정기공시 (earnings).
                   ``"B"`` → 주요사항보고서 (material events).
                   ``"F"`` → 공정공시 (guidance/fair disclosures).
                   ``""``  → all types.

    Returns:
        ``pd.Series`` with daily ``DatetimeIndex`` (calendar days, start→end).
        Values are float counts: 0.0 on days with no filings, ≥1.0 otherwise.
        Index name is ``"date"``.

    Raises:
        RuntimeError: API key missing, HTTP error, or DART API error.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"DART API key not set.  Set the {api_key_env!r} environment variable.\n"
            "Free registration at: https://opendart.fss.or.kr/intro/main.do\n"
            "Approval usually takes a few hours."
        )

    end_str = end or date.today().isoformat()
    bgn_de = start.replace("-", "")[:8]
    end_de = end_str.replace("-", "")[:8]

    filings = _fetch_all_pages(requests, api_key, corp_code, bgn_de, end_de, pblntf_ty)

    # Build daily count series over the full calendar range
    date_range = pd.date_range(start=start, end=end_str, freq="D")
    counts = pd.Series(0.0, index=date_range, dtype=float)
    counts.index.name = "date"
    counts.name = f"dart_{pblntf_ty or 'all'}_{corp_code}"

    for filing in filings:
        rcept_dt = filing.get("rcept_dt", "")
        if len(rcept_dt) == 8:
            try:
                ts = pd.Timestamp(
                    f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
                )
                if ts in counts.index:
                    counts[ts] += 1.0
            except ValueError:
                pass

    n_filing_days = int((counts > 0).sum())
    total = int(counts.sum())
    logger.info(
        "[dart_client] corp=%s type=%r: %d filings on %d days  (%s → %s)",
        corp_code, pblntf_ty or "all", total, n_filing_days, start, end_str,
    )
    return counts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_all_pages(
    requests_module,
    api_key: str,
    corp_code: str,
    bgn_de: str,
    end_de: str,
    pblntf_ty: str,
) -> list[dict]:
    """Paginate through the DART list endpoint, collecting all items."""
    all_items: list[dict] = []
    page_no = 1

    while True:
        params: dict[str, object] = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": _PAGE_SIZE,
        }
        if pblntf_ty:
            params["pblntf_ty"] = pblntf_ty

        try:
            resp = requests_module.get(_BASE_URL, params=params, timeout=30)
        except Exception as exc:
            raise RuntimeError(f"DART HTTP request failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"DART API HTTP error: status={resp.status_code}  "
                f"body={resp.text[:200]}"
            )

        data = resp.json()
        status = data.get("status", "")

        # "013" = "조회된 데이터가 없습니다" — no results, not an error
        if status == "013":
            break
        if status != "000":
            raise RuntimeError(
                f"DART API returned status={status!r}: "
                f"{data.get('message', '')}"
            )

        page_items = data.get("list", [])
        all_items.extend(page_items)

        total_count = int(data.get("total_count", 0))
        if not page_items or len(all_items) >= total_count:
            break

        page_no += 1
        time.sleep(_POLITE_DELAY_S)

    return all_items
