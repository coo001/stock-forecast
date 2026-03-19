"""Tests for src/data/external/news_client.py.

All HTTP calls are mocked — no network access required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.external.news_client import (
    _date_chunks,
    _gdelt_chunk,
    _headline_sentiment,
    fetch_series,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _gdelt_response(points: list[tuple[str, float]]) -> dict:
    """Build a minimal GDELT Timeline API response."""
    series = [{"date": dt, "value": val} for dt, val in points]
    return {"timeline": [{"series": series}]}


# ── _headline_sentiment ────────────────────────────────────────────────────────

class TestHeadlineSentiment:
    def test_empty_string(self):
        assert _headline_sentiment("") == 0.0

    def test_no_keywords(self):
        assert _headline_sentiment("the weather is nice today") == 0.0

    def test_positive_only(self):
        score = _headline_sentiment("Samsung earnings surge record growth")
        assert score > 0.0
        assert score <= 1.0

    def test_negative_only(self):
        score = _headline_sentiment("Samsung stock drop decline loss warning")
        assert score < 0.0
        assert score >= -1.0

    def test_mixed_balanced(self):
        score = _headline_sentiment("surge drop")
        assert score == 0.0

    def test_case_insensitive(self):
        assert _headline_sentiment("SURGE") == _headline_sentiment("surge")

    def test_score_range(self):
        for text in [
            "Samsung beats record profit growth milestone",
            "Samsung drops loss decline weak warning bearish",
            "Apple reports mixed results with moderate growth",
        ]:
            score = _headline_sentiment(text)
            assert -1.0 <= score <= 1.0, f"Out of range for: {text!r}"


# ── _date_chunks ───────────────────────────────────────────────────────────────

class TestDateChunks:
    def test_single_chunk(self):
        chunks = _date_chunks("2023-01-01", "2023-02-15", months=3)
        assert len(chunks) == 1
        assert chunks[0] == ("2023-01-01", "2023-02-15")

    def test_two_chunks(self):
        chunks = _date_chunks("2023-01-01", "2023-07-31", months=3)
        assert len(chunks) == 3  # Jan–Mar, Apr–Jun, Jul
        assert chunks[0][0] == "2023-01-01"
        assert chunks[-1][1] == "2023-07-31"

    def test_chunks_are_non_overlapping(self):
        chunks = _date_chunks("2022-01-01", "2023-12-31", months=3)
        for i in range(len(chunks) - 1):
            end_i = pd.Timestamp(chunks[i][1])
            start_next = pd.Timestamp(chunks[i + 1][0])
            assert start_next == end_i + pd.Timedelta(days=1)

    def test_single_day_range(self):
        chunks = _date_chunks("2023-06-15", "2023-06-15", months=3)
        assert len(chunks) == 1
        assert chunks[0] == ("2023-06-15", "2023-06-15")

    def test_exact_quarter_boundary(self):
        chunks = _date_chunks("2023-01-01", "2023-03-31", months=3)
        assert len(chunks) == 1
        assert chunks[0] == ("2023-01-01", "2023-03-31")


# ── _gdelt_chunk ───────────────────────────────────────────────────────────────

class TestGdeltChunk:
    def test_parses_valid_response(self):
        points = [
            ("20230105000000", 1.5),
            ("20230106000000", -0.3),
            ("20230107000000", 2.1),
        ]
        mock_req = MagicMock()
        mock_req.get.return_value = _mock_response(_gdelt_response(points))

        result = _gdelt_chunk(mock_req, "samsung", "2023-01-05", "2023-01-07", "TimelineTone")

        assert len(result) == 3
        assert result[pd.Timestamp("2023-01-05")] == pytest.approx(1.5)
        assert result[pd.Timestamp("2023-01-06")] == pytest.approx(-0.3)

    def test_empty_timeline_returns_empty_dict(self):
        mock_req = MagicMock()
        mock_req.get.return_value = _mock_response({"timeline": []})

        result = _gdelt_chunk(mock_req, "samsung", "2023-01-01", "2023-01-07", "TimelineTone")
        assert result == {}

    def test_http_error_raises(self):
        mock_req = MagicMock()
        mock_req.get.return_value = _mock_response({}, status=429)

        with pytest.raises(RuntimeError, match="GDELT API HTTP 429"):
            _gdelt_chunk(mock_req, "samsung", "2023-01-01", "2023-01-07", "TimelineTone")

    def test_missing_value_skipped(self):
        data = {"timeline": [{"series": [{"date": "20230105000000"}]}]}  # no "value"
        mock_req = MagicMock()
        mock_req.get.return_value = _mock_response(data)

        result = _gdelt_chunk(mock_req, "samsung", "2023-01-05", "2023-01-05", "TimelineTone")
        assert result == {}


# ── fetch_series (gdelt backend) ───────────────────────────────────────────────

class TestFetchSeriesGdelt:
    def test_returns_daily_series(self):
        points_jan = [("20230105000000", 1.5), ("20230110000000", -0.5)]
        mock_req = MagicMock()
        mock_req.get.return_value = _mock_response(_gdelt_response(points_jan))

        import src.data.external.news_client as news_mod

        def fake_gdelt_chunk(req, query, start, end, mode):
            return {pd.Timestamp("2023-01-05"): 1.5, pd.Timestamp("2023-01-10"): -0.5}

        original = news_mod._gdelt_chunk
        news_mod._gdelt_chunk = fake_gdelt_chunk
        try:
            with patch("src.data.external.news_client.time.sleep"):
                series = fetch_series(
                    "samsung", "2023-01-01", "2023-01-31", source_backend="gdelt"
                )
        finally:
            news_mod._gdelt_chunk = original

        assert isinstance(series, pd.Series)
        assert series.index.name == "date"
        # Known data points
        assert series[pd.Timestamp("2023-01-05")] == pytest.approx(1.5)
        assert series[pd.Timestamp("2023-01-10")] == pytest.approx(-0.5)
        # Sparse dates are NaN (not filled here because GDELT doesn't report them)

    def test_failed_chunk_does_not_abort(self):
        """A chunk failure should log a warning and continue, not raise."""
        import src.data.external.news_client as news_mod

        call_count = {"n": 0}

        def flaky_chunk(req, query, start, end, mode):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated network timeout")
            return {pd.Timestamp("2023-04-10"): 0.8}

        original = news_mod._gdelt_chunk
        news_mod._gdelt_chunk = flaky_chunk
        try:
            with patch("src.data.external.news_client.time.sleep"):
                series = fetch_series(
                    "samsung", "2023-01-01", "2023-06-30", source_backend="gdelt"
                )
        finally:
            news_mod._gdelt_chunk = original

        # Should not raise; first chunk failed, second succeeded
        assert isinstance(series, pd.Series)
        assert series[pd.Timestamp("2023-04-10")] == pytest.approx(0.8)

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported source_backend"):
            fetch_series("samsung", "2023-01-01", "2023-01-31", source_backend="unknown_api")


# ── fetch_series (newsapi backend) ─────────────────────────────────────────────

class TestFetchSeriesNewsAPI:
    def _make_article(self, date_str: str, title: str) -> dict:
        return {
            "publishedAt": f"{date_str}T10:00:00Z",
            "title": title,
            "description": "",
        }

    def test_returns_sentiment_series(self):
        articles = [
            self._make_article("2023-01-05", "Samsung reports record profit surge"),
            self._make_article("2023-01-05", "Samsung semiconductor growth beats"),
            self._make_article("2023-01-06", "Samsung stock drop loss warning"),
        ]
        api_response = {"status": "ok", "totalResults": 3, "articles": articles}

        mock_requests = MagicMock()
        mock_requests.get.return_value = _mock_response(api_response)

        with patch("src.data.external.news_client.time.sleep"):
            from src.data.external.news_client import _fetch_newsapi
            series = _fetch_newsapi(
                mock_requests, "samsung", "2023-01-05", "2023-01-07", "test_key", "tone"
            )

        assert isinstance(series, pd.Series)
        assert series.index.name == "date"
        # Jan 5 should have positive sentiment (surge, profit, growth, beats)
        assert series[pd.Timestamp("2023-01-05")] > 0
        # Jan 6 should have negative sentiment (drop, loss, warning)
        assert series[pd.Timestamp("2023-01-06")] < 0

    def test_missing_newsapi_key_raises(self):
        import os
        os.environ.pop("NEWSAPI_KEY", None)
        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("NEWSAPI_KEY", None)
            with pytest.raises(RuntimeError, match="NewsAPI key not set"):
                fetch_series(
                    "samsung", "2023-01-01", "2023-01-31",
                    source_backend="newsapi", api_key_env="NEWSAPI_KEY",
                )

    def test_count_metric(self, monkeypatch):
        """metric='count' should return article count, not sentiment."""
        articles = [
            self._make_article("2023-01-05", "Samsung news one"),
            self._make_article("2023-01-05", "Samsung news two"),
            self._make_article("2023-01-05", "Samsung news three"),
        ]
        api_response = {"status": "ok", "totalResults": 3, "articles": articles}

        mock_requests = MagicMock()
        mock_requests.get.return_value = _mock_response(api_response)

        with patch("src.data.external.news_client.time.sleep"):
            from src.data.external.news_client import _fetch_newsapi
            series = _fetch_newsapi(
                mock_requests, "samsung", "2023-01-05", "2023-01-07", "test_key", "count"
            )

        assert series[pd.Timestamp("2023-01-05")] == 3.0
