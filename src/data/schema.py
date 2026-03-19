"""Pydantic schemas for raw market data."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class OHLCVRow(BaseModel):
    """Single OHLCV observation for one trading day."""

    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def high_gte_low(self) -> "OHLCVRow":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) on {self.date}")
        return self


class DataConfig(BaseModel):
    """Config subset for data loading (mirrors config/default.yaml → data)."""

    ticker: str
    start_date: str
    end_date: str | None = None
    interval: str = "1d"
    cache_dir: str = "data/raw"
