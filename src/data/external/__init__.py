"""External data clients for macro and market features.

Public API
----------
``ExternalCache``      — file-based CSV cache with TTL
``market_client``      — yfinance: KOSPI, USD/KRW, VIX, S&P 500
``fred_client``        — FRED REST API: US 10Y yield, VIX, DFF
``ecos_client``        — Bank of Korea ECOS: KR base rate, CPI
``alpha_vantage_client`` — Alpha Vantage stub (reserved for future use)

All clients are stateless functions so they are easy to mock in tests.
API keys are always read from environment variables — never hardcoded.
"""
from src.data.external.cache import ExternalCache

__all__ = ["ExternalCache"]
