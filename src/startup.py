"""Preflight dependency checks for the Samsung forecasting pipeline.

Call ``check_required()`` at the start of ``run_pipeline.py`` (and any other
entry point) to fail fast with clear install instructions rather than
encountering confusing ImportErrors deep inside the pipeline.

Design:
- ``check_required()``  — core scientific stack (pandas, numpy, lightgbm, …)
- ``check_optional()``  — optional extras; returns a dict of availability flags
- ``require_yfinance()`` — specifically validates yfinance, mentions --synthetic
"""
from __future__ import annotations

import importlib
import sys
from typing import NamedTuple


class _DepCheck(NamedTuple):
    module: str
    install: str          # pip install argument


_REQUIRED: list[_DepCheck] = [
    _DepCheck("pandas",      "pandas"),
    _DepCheck("numpy",       "numpy"),
    _DepCheck("lightgbm",    "lightgbm"),
    _DepCheck("sklearn",     "scikit-learn"),
    _DepCheck("pydantic",    "pydantic>=2"),
    _DepCheck("yaml",        "pyyaml"),
    _DepCheck("ta",          "ta"),
]

_OPTIONAL: dict[str, _DepCheck] = {
    "yfinance":  _DepCheck("yfinance",  "yfinance"),
    "openai":    _DepCheck("openai",    "openai>=1.0"),
    "anthropic": _DepCheck("anthropic", "anthropic>=0.25"),
}


def check_required() -> None:
    """Verify that all required dependencies are importable.

    Raises:
        SystemExit: with a formatted error message listing every missing
            package and the exact ``pip install`` command to fix it.
    """
    missing: list[_DepCheck] = []
    for dep in _REQUIRED:
        try:
            importlib.import_module(dep.module)
        except ImportError:
            missing.append(dep)

    if missing:
        packages = " ".join(d.install for d in missing)
        names = ", ".join(d.module for d in missing)
        _die(
            f"Missing required dependencies: {names}\n"
            f"Install them with:\n\n"
            f"    pip install {packages}\n\n"
            "Or install everything at once:\n\n"
            "    pip install -r requirements.txt\n"
        )


def check_optional() -> dict[str, bool]:
    """Check availability of optional dependencies.

    Returns:
        Dict mapping dependency name to True/False (available or not).
        Does NOT raise — callers decide what to do with missing optionals.
    """
    result: dict[str, bool] = {}
    for name, dep in _OPTIONAL.items():
        try:
            importlib.import_module(dep.module)
            result[name] = True
        except ImportError:
            result[name] = False
    return result


def require_yfinance() -> None:
    """Assert that yfinance is importable; raise SystemExit with helpful message.

    Called by run_pipeline.py when real market data is requested (i.e. not
    running with ``--synthetic``).

    Raises:
        SystemExit: if yfinance is not installed.
    """
    try:
        importlib.import_module("yfinance")
    except ImportError:
        _die(
            "yfinance is not installed, which is required to download market data.\n\n"
            "Fix options:\n\n"
            "  1. Install yfinance:\n"
            "         pip install yfinance\n\n"
            "  2. Use synthetic data instead (no internet required):\n"
            "         python run_pipeline.py --synthetic\n"
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _die(message: str) -> None:
    """Print a formatted error and exit with code 1."""
    border = "=" * 60
    print(f"\n{border}", file=sys.stderr)
    print("[Startup check failed]", file=sys.stderr)
    print(border, file=sys.stderr)
    print(message, file=sys.stderr)
    print(border, file=sys.stderr)
    sys.exit(1)
