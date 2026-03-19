"""Market data client — daily close-price series via yfinance.

Supported symbols (examples):
    ^KS11       KOSPI Composite Index
    ^GSPC       S&P 500 Index
    ^VIX        CBOE Volatility Index (alternative to FRED VIXCLS)
    USDKRW=X    USD / KRW exchange rate
    005930.KS   Samsung Electronics (same as primary pipeline ticker)

Raises ``RuntimeError`` if yfinance is not installed, matching the behaviour
of ``src.data.loader._download()``.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_series(
    symbol: str,
    start: str,
    end: str | None = None,
) -> pd.Series:
    """Download daily close-price series for *symbol* via yfinance.

    Args:
        symbol: Yahoo Finance ticker (e.g. ``"^KS11"``, ``"USDKRW=X"``).
        start: ISO date string (inclusive), e.g. ``"2015-01-01"``.
        end: ISO date string (exclusive). ``None`` → today.

    Returns:
        ``pd.Series`` with timezone-naive ``DatetimeIndex`` (business days
        present in yfinance data) and float close-price values.
        Index name is ``"date"``.

    Raises:
        RuntimeError: if yfinance is not installed.
        ValueError: if the returned series is empty or all NaN.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for market data.\n"
            "Install with: pip install yfinance\n"
            "Or disable external_data in config/default.yaml."
        ) from exc

    ticker_obj = yf.Ticker(symbol)
    raw = ticker_obj.history(start=start, end=end, auto_adjust=True)

    if raw.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}' (start={start}, end={end})."
        )

    close = raw["Close"].rename(symbol)
    # Remove timezone info so the index is comparable with Samsung's naive index
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    close = close.dropna()

    if close.empty:
        raise ValueError(f"All close values are NaN for symbol '{symbol}'.")

    logger.info(
        "[market_client] %s: %d rows  (%s → %s)",
        symbol, len(close), close.index[0].date(), close.index[-1].date(),
    )
    return close
