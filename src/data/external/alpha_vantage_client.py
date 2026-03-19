"""Alpha Vantage data client — stub, reserved for future use.

Alpha Vantage provides fundamental data (earnings, EPS, P/E), alternative
market data, and economic indicators not covered by yfinance or FRED.

API reference: https://www.alphavantage.co/documentation/
Free key registration: https://www.alphavantage.co/support/#api-key
Set env var:  ALPHA_VANTAGE_API_KEY=your_key_here

Candidate features for future implementation
---------------------------------------------
- Earnings surprise (EPS actual vs estimate)
- Quarterly revenue growth
- Analyst sentiment scores
- Sector rotation signals (e.g. semiconductor vs broader market)

This module raises ``NotImplementedError`` until a concrete feature
requirement is identified — keeping the external data layer clean and
replaceable.
"""
from __future__ import annotations

import pandas as pd


def fetch_series(
    function: str,
    symbol: str,
    *,
    api_key_env: str = "ALPHA_VANTAGE_API_KEY",
    **kwargs,
) -> pd.Series:
    """Fetch a time series from Alpha Vantage (not yet implemented).

    Args:
        function: Alpha Vantage function name (e.g. ``"TIME_SERIES_DAILY"``).
        symbol: Equity ticker (e.g. ``"005930.KS"``).
        api_key_env: Env var name holding the Alpha Vantage key.
        **kwargs: Additional API parameters (passed through when implemented).

    Raises:
        NotImplementedError: always — this client is a placeholder.
    """
    raise NotImplementedError(
        "AlphaVantage client is not yet implemented.\n"
        "Use market_client for equity/FX data or fred_client for macro data.\n"
        f"Requested: function={function!r}, symbol={symbol!r}"
    )
