"""Tests for src/startup.py preflight dependency checker."""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from src.startup import check_optional, check_required, require_yfinance


# ── check_required ─────────────────────────────────────────────────────────────

def test_check_required_passes_when_all_installed():
    """All project dependencies should be importable in the test environment."""
    # Should not raise or call sys.exit
    check_required()


def test_check_required_exits_when_module_missing(monkeypatch):
    """If a required module is missing, check_required() must call sys.exit(1)."""
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated missing lightgbm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("src.startup.importlib.import_module", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        check_required()

    assert exc_info.value.code == 1


def test_check_required_error_message_contains_package_name(monkeypatch, capsys):
    """The error output must name the missing package and a pip install command."""
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("src.startup.importlib.import_module", fake_import)

    with pytest.raises(SystemExit):
        check_required()

    captured = capsys.readouterr()
    assert "lightgbm" in captured.err
    assert "pip install" in captured.err


# ── check_optional ─────────────────────────────────────────────────────────────

def test_check_optional_returns_dict():
    result = check_optional()
    assert isinstance(result, dict)
    assert "yfinance" in result
    assert "openai" in result
    assert "anthropic" in result


def test_check_optional_does_not_raise_when_missing(monkeypatch):
    """check_optional must return False for a missing package, never raise."""
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("simulated missing openai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("src.startup.importlib.import_module", fake_import)

    result = check_optional()
    assert result["openai"] is False


def test_check_optional_true_when_available(monkeypatch):
    """If import succeeds, check_optional reports True."""
    import types

    fake_module = types.ModuleType("fake_module")

    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            return fake_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("src.startup.importlib.import_module", fake_import)

    result = check_optional()
    assert result["yfinance"] is True


# ── require_yfinance ──────────────────────────────────────────────────────────

def test_require_yfinance_passes_when_installed():
    """If yfinance is available, require_yfinance() must not raise."""
    try:
        importlib.import_module("yfinance")
    except ImportError:
        pytest.skip("yfinance not installed in this environment")

    require_yfinance()   # should not raise


def test_require_yfinance_exits_when_missing(monkeypatch, capsys):
    """require_yfinance() must call sys.exit(1) and mention --synthetic."""
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("simulated missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("src.startup.importlib.import_module", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        require_yfinance()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "--synthetic" in captured.err
    assert "pip install yfinance" in captured.err
