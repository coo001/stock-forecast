"""Tests for src/data/external/dart_client.py.

All HTTP calls are mocked — no network access required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.external.dart_client import (
    SAMSUNG_CORP_CODE,
    _fetch_all_pages,
    fetch_series,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _page(items: list[dict], total: int, status: str = "000") -> dict:
    return {"status": status, "message": "", "total_count": total, "list": items}


def _filing(rcept_dt: str) -> dict:
    return {"rcept_dt": rcept_dt, "corp_name": "삼성전자", "report_nm": "test"}


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFetchAllPages:
    def test_single_page(self):
        items = [_filing("20230105"), _filing("20230112")]
        requests_mod = MagicMock()
        requests_mod.get.return_value = _mock_response(_page(items, total=2))

        result = _fetch_all_pages(
            requests_mod, "test_key", "00126380", "20230101", "20230131", ""
        )

        assert len(result) == 2
        assert requests_mod.get.call_count == 1

    def test_no_results_status_013(self):
        """DART status '013' means no data — should return empty list, not raise."""
        requests_mod = MagicMock()
        requests_mod.get.return_value = _mock_response(
            {"status": "013", "message": "조회된 데이터가 없습니다", "list": []}
        )

        result = _fetch_all_pages(
            requests_mod, "test_key", "00126380", "20230101", "20230131", ""
        )

        assert result == []

    def test_multiple_pages(self):
        page1_items = [_filing(f"202301{d:02d}") for d in range(1, 6)]  # 5 items
        page2_items = [_filing(f"202301{d:02d}") for d in range(6, 9)]  # 3 items

        requests_mod = MagicMock()
        requests_mod.get.side_effect = [
            _mock_response(_page(page1_items, total=8)),
            _mock_response(_page(page2_items, total=8)),
        ]

        with patch("src.data.external.dart_client.time.sleep"):
            result = _fetch_all_pages(
                requests_mod, "test_key", "00126380", "20230101", "20230131", ""
            )

        assert len(result) == 8
        assert requests_mod.get.call_count == 2

    def test_http_error_raises(self):
        requests_mod = MagicMock()
        requests_mod.get.return_value = _mock_response({}, status=500)

        with pytest.raises(RuntimeError, match="HTTP error"):
            _fetch_all_pages(
                requests_mod, "test_key", "00126380", "20230101", "20230131", ""
            )

    def test_api_error_status_raises(self):
        requests_mod = MagicMock()
        requests_mod.get.return_value = _mock_response(
            {"status": "010", "message": "인증키가 유효하지 않습니다."}
        )

        with pytest.raises(RuntimeError, match="status="):
            _fetch_all_pages(
                requests_mod, "bad_key", "00126380", "20230101", "20230131", ""
            )


class TestFetchSeries:
    def test_returns_daily_series(self, monkeypatch):
        items = [_filing("20230105"), _filing("20230105"), _filing("20230112")]
        monkeypatch.setenv("DART_API_KEY", "test_key")

        import src.data.external.dart_client as dart_mod

        def fake_fetch_all(req_mod, api_key, corp, bgn, end, ty):
            return items

        monkeypatch.setattr(dart_mod, "_fetch_all_pages", fake_fetch_all)

        series = fetch_series(
            SAMSUNG_CORP_CODE, "2023-01-01", "2023-01-31",
            api_key_env="DART_API_KEY",
        )

        assert isinstance(series, pd.Series)
        assert series.index.name == "date"
        assert len(series) == 31  # Jan 2023
        # 2023-01-05 has 2 filings
        assert series[pd.Timestamp("2023-01-05")] == 2.0
        # 2023-01-12 has 1 filing
        assert series[pd.Timestamp("2023-01-12")] == 1.0
        # Non-filing days are 0.0
        assert series[pd.Timestamp("2023-01-02")] == 0.0

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove DART_API_KEY from env if present
            import os
            os.environ.pop("DART_API_KEY", None)
            with pytest.raises(RuntimeError, match="DART API key not set"):
                fetch_series(SAMSUNG_CORP_CODE, "2023-01-01", "2023-01-31")

    def test_samsung_corp_code_constant(self):
        assert SAMSUNG_CORP_CODE == "00126380"

    def test_date_range_includes_all_calendar_days(self):
        """Series should cover every calendar day, not just trading days."""
        items: list[dict] = []

        import src.data.external.dart_client as dart_mod
        original = dart_mod._fetch_all_pages

        def fake_fetch_all(*args, **kwargs):
            return items

        dart_mod._fetch_all_pages = fake_fetch_all
        try:
            with patch.dict("os.environ", {"DART_API_KEY": "test_key"}):
                series = fetch_series(
                    SAMSUNG_CORP_CODE, "2023-01-01", "2023-01-07"
                )
        finally:
            dart_mod._fetch_all_pages = original

        # 7 calendar days
        assert len(series) == 7
        assert (series == 0.0).all()

    def test_pblntf_ty_passed_through(self, monkeypatch):
        """pblntf_ty parameter should be forwarded to _fetch_all_pages."""
        captured = {}

        import src.data.external.dart_client as dart_mod

        def fake_fetch_all(req_mod, api_key, corp, bgn, end, ty):
            captured["pblntf_ty"] = ty
            return []

        monkeypatch.setattr(dart_mod, "_fetch_all_pages", fake_fetch_all)
        monkeypatch.setenv("DART_API_KEY", "test_key")

        fetch_series(SAMSUNG_CORP_CODE, "2023-01-01", "2023-01-31", pblntf_ty="A")

        assert captured["pblntf_ty"] == "A"
