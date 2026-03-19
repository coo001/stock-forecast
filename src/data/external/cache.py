"""Lightweight file-based cache for external time-series data.

Each series is stored as a CSV in ``data/external/`` alongside a small
``.meta.json`` sidecar that records the fetch timestamp.  On read, the
cache compares the file age against ``ttl_hours`` and returns ``None`` when
the entry is stale, prompting a fresh download.

Design constraints:
- No additional runtime dependencies (stdlib + pandas only)
- Stale / corrupt files fail silently (log + return None)
- Writing is best-effort (log warning on I/O errors, never raise)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ExternalCache:
    """CSV-based cache for ``pd.Series`` objects (date-indexed)."""

    def __init__(self, cache_dir: str | Path = "data/external") -> None:
        self.cache_dir = Path(cache_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, ttl_hours: float = 24.0) -> pd.Series | None:
        """Return the cached series, or ``None`` if absent / stale / corrupt.

        Args:
            key: Unique cache key (source + symbol + date range).
            ttl_hours: Maximum age in hours before the entry is considered stale.

        Returns:
            ``pd.Series`` with DatetimeIndex, or ``None``.
        """
        csv_path = self._csv_path(key)
        meta_path = self._meta_path(key)

        if not csv_path.exists() or not meta_path.exists():
            return None

        try:
            with meta_path.open(encoding="utf-8") as fh:
                info = json.load(fh)
            fetched_at = datetime.fromisoformat(info["fetched_at"])
            age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600

            if age_hours > ttl_hours:
                logger.debug("[cache] stale (%.1fh > %.0fh): %s", age_hours, ttl_hours, key)
                return None

            series = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze("columns")
            series.index = pd.to_datetime(series.index)
            series.index.name = "date"
            logger.debug("[cache] hit: %s  (%d rows, age %.1fh)", key, len(series), age_hours)
            return series

        except Exception as exc:
            logger.warning("[cache] read error for '%s': %s", key, exc)
            return None

    def set(self, key: str, series: pd.Series) -> None:
        """Write *series* to the cache under *key*.

        Silently logs on I/O error; never raises.

        Args:
            key: Unique cache key.
            series: Date-indexed ``pd.Series`` to persist.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path(key).parent.mkdir(parents=True, exist_ok=True)
            series.to_csv(self._csv_path(key), header=True)
            with self._meta_path(key).open("w", encoding="utf-8") as fh:
                json.dump(
                    {"fetched_at": datetime.now(timezone.utc).isoformat(), "key": key},
                    fh,
                )
            logger.debug("[cache] wrote: %s  (%d rows)", key, len(series))
        except Exception as exc:
            logger.warning("[cache] write error for '%s': %s", key, exc)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _csv_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("^", "").replace("=", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.csv"

    def _meta_path(self, key: str) -> Path:
        return self._csv_path(key).with_suffix("").with_suffix(".meta.json")
