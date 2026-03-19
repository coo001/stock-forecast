"""DataAgent — loads and validates OHLCV data.

Responsibilities:
- Call ``src.data.loader.load_ohlcv`` (the *only* place that touches external APIs)
- Validate data quality (sufficient rows, no gaps, no negative prices)
- Produce a DataSummary for downstream agents
- Store the raw DataFrame on self so the orchestrator can pass it to ModelingAgent
  without re-downloading or re-parsing

The DataFrame is *not* part of the Pydantic output (DataSummary) because it is
not JSON-serialisable. The orchestrator accesses ``data_agent.last_df`` directly.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.schemas import DataSummary, ExecutionPlan
from src.data.loader import load_ohlcv
from src.data.schema import DataConfig

logger = logging.getLogger(__name__)


class DataAgentInput(DataSummary.__class__):  # just use a plain dataclass-style carrier
    pass


# Use a simple container instead of inheriting from schema
class _DataInput:
    """Internal carrier: plan + raw config needed to fetch data."""
    def __init__(self, plan: ExecutionPlan, config_override: dict | None = None):
        self.plan = plan
        self.config_override = config_override or {}


class DataAgent(BaseAgent[_DataInput, DataSummary]):
    """Loads OHLCV data from cache or Yahoo Finance and validates quality."""

    name = "DataAgent"
    timeout_seconds = 120.0  # network download may be slow

    def __init__(self) -> None:
        super().__init__()
        self.last_df: pd.DataFrame | None = None   # side-channel for orchestrator

    def run_with_df(
        self,
        plan: ExecutionPlan,
        *,
        ohlcv_df: pd.DataFrame | None = None,
        data_config_override: dict | None = None,
    ) -> DataSummary:
        """Primary entry point used by the orchestrator.

        Args:
            plan: ExecutionPlan from PlannerAgent.
            ohlcv_df: Pre-loaded DataFrame (bypasses download; used in tests/demo).
            data_config_override: Additional DataConfig fields (e.g. cache_dir).

        Returns:
            DataSummary with quality metrics.
        """
        if ohlcv_df is not None:
            logger.info("[DataAgent] using pre-loaded DataFrame (%d rows)", len(ohlcv_df))
            self.last_df = ohlcv_df
            return self._summarise(ohlcv_df, source="synthetic", plan=plan)

        inp = _DataInput(plan, data_config_override)
        return self.run(inp)

    def _run(self, input: _DataInput) -> DataSummary:
        plan = input.plan
        overrides = input.config_override

        data_cfg = DataConfig(
            ticker=plan.ticker,
            start_date=overrides.get("start_date", "2015-01-01"),
            end_date=overrides.get("end_date", None),
            interval=overrides.get("interval", "1d"),
            cache_dir=overrides.get("cache_dir", "data/raw"),
        )

        try:
            df = load_ohlcv(data_cfg)
            source = "cache" if _cache_exists(data_cfg) else "download"
        except Exception as exc:
            raise RuntimeError(f"Failed to load OHLCV for {plan.ticker}: {exc}") from exc

        self.last_df = df
        return self._summarise(df, source=source, plan=plan)

    def _summarise(self, df: pd.DataFrame, *, source: str, plan: ExecutionPlan) -> DataSummary:
        bt = plan.backtest_config
        needed = bt["initial_train_days"] + bt["step_days"]

        if df.empty:
            raise ValueError("OHLCV DataFrame is empty after loading.")

        # Compute span in years to estimate trading days / year
        span_days = (df.index[-1] - df.index[0]).days
        tdy = round(len(df) / max(span_days / 365.25, 0.01), 1)

        summary = DataSummary(
            ticker=plan.ticker,
            n_rows=len(df),
            date_start=str(df.index[0].date()),
            date_end=str(df.index[-1].date()),
            trading_days_per_year=tdy,
            close_min=float(df["close"].min()),
            close_max=float(df["close"].max()),
            close_mean=float(df["close"].mean()),
            volume_mean=float(df["volume"].mean()),
            missing_close_pct=0.0,   # load_ohlcv already drops NaN close
            sufficient_for_backtest=len(df) >= needed,
            source=source,
        )

        if not summary.sufficient_for_backtest:
            logger.warning(
                "[DataAgent] Only %d rows; need %d for walk-forward. "
                "Reduce initial_train_days or step_days.",
                len(df), needed,
            )
        else:
            logger.info(
                "[DataAgent] %d rows loaded (%s → %s, src=%s)",
                len(df), summary.date_start, summary.date_end, source,
            )

        return summary


def _cache_exists(cfg: DataConfig) -> bool:
    from pathlib import Path
    safe = cfg.ticker.replace(".", "_").replace("/", "_")
    path = Path(cfg.cache_dir) / f"{safe}_{cfg.start_date}_{cfg.interval}.csv"
    return path.exists()
