"""Thin LLM client for agent reasoning steps.

Supports OpenAI and Anthropic.  Designed to be injected into agents so
that tests can swap in a ``MockLLMClient`` without touching agent logic.

Usage::

    from src.agents.llm_client import LLMClient

    client = LLMClient.from_config(cfg["llm"])
    result = client.chat_json(system="...", user="...")  # -> dict

    # For tests:
    client = LLMClient.mock({"verdict": "marginal", ...})

Design rules (from CLAUDE.md § AutoGen rules):
- Retry on JSON parse failure (up to max_parse_retries)
- Timeout on every call
- Log each call's duration and failure reason
- Return structured JSON; never return raw text to agents
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = {"openai", "anthropic", "mock"}


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


class LLMClient:
    """Synchronous LLM client producing JSON-only responses.

    Args:
        provider: ``"openai"`` | ``"anthropic"`` | ``"mock"``
        model: Model identifier (e.g. ``"gpt-4o-mini"``, ``"claude-haiku-4-5-20251001"``).
        api_key: API key string. If None, read from the env var named in ``api_key_env``.
        api_key_env: Name of the environment variable holding the API key.
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the completion.
        timeout_seconds: Per-request wall-clock timeout.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        if provider not in _VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {_VALID_PROVIDERS}, got '{provider}'")

        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat_json(
        self,
        system: str,
        user: str,
        max_parse_retries: int = 2,
    ) -> dict[str, Any]:
        """Call the LLM and return a parsed JSON dict.

        If the response is not valid JSON, the client retries with an
        explicit correction prompt (up to ``max_parse_retries`` times).

        Args:
            system: System-role prompt (describes schema and task).
            user: User-role prompt (the actual data to analyse).
            max_parse_retries: How many extra attempts to make if JSON parsing fails.

        Returns:
            Parsed dict from the LLM response.

        Raises:
            LLMError: if all attempts fail or the provider is unavailable.
        """
        t0 = time.monotonic()
        last_exc: Exception | None = None
        raw = ""

        for attempt in range(max_parse_retries + 1):
            try:
                if attempt == 0:
                    raw = self._chat(system, user)
                else:
                    # Give the LLM explicit correction feedback
                    correction = (
                        f"Your previous response was not valid JSON. "
                        f"Error: {last_exc}. "
                        "Respond with ONLY a raw JSON object, no markdown, no explanation."
                    )
                    raw = self._chat(system, f"{user}\n\n{correction}")

                result = _parse_json(raw)
                elapsed = time.monotonic() - t0
                logger.debug(
                    "[LLMClient] %s/%s ok in %.2fs (attempt %d)",
                    self.provider, self.model, elapsed, attempt + 1,
                )
                return result

            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "[LLMClient] JSON parse failed (attempt %d/%d): %s",
                    attempt + 1, max_parse_retries + 1, exc,
                )

            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.error("[LLMClient] call failed after %.2fs: %s: %s", elapsed, type(exc).__name__, exc)
                raise LLMError(f"{self.provider} call failed: {exc}") from exc

        raise LLMError(
            f"Failed to obtain valid JSON from {self.provider} after "
            f"{max_parse_retries + 1} attempts. Last raw response: {raw[:200]!r}"
        )

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LLMClient":
        """Instantiate from the ``llm`` section of ``config/default.yaml``.

        If ``provider`` is ``"none"`` or the config is missing, returns ``None``
        so callers can check ``if llm_client:`` cleanly.
        """
        provider = cfg.get("provider", "none")
        if provider == "none":
            raise ValueError(
                "LLM provider is set to 'none'. "
                "Set llm.provider to 'openai' or 'anthropic' in config."
            )
        return cls(
            provider=provider,
            model=cfg.get("model", "gpt-4o-mini"),
            api_key=cfg.get("api_key"),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 1024),
            timeout_seconds=cfg.get("timeout_seconds", 30.0),
        )

    @classmethod
    def mock(cls, responses: dict[str, Any] | list[dict[str, Any]]) -> "_MockLLMClient":  # type: ignore[override]
        """Create a mock client for tests.

        Args:
            responses: A single dict (returned for every call) or a list of
                       dicts (returned in order; the last one is repeated if
                       the list is exhausted).
        """
        return _MockLLMClient(responses)

    # ── Provider implementations ───────────────────────────────────────────────

    def _chat(self, system: str, user: str) -> str:
        if self.provider == "openai":
            return self._chat_openai(system, user)
        elif self.provider == "anthropic":
            return self._chat_anthropic(system, user)
        elif self.provider == "mock":
            raise RuntimeError("Use LLMClient.mock() to create a mock client.")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _chat_openai(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai package not installed. Run: pip install openai") from exc

        if not self.api_key:
            raise LLMError("OpenAI API key not set. Set OPENAI_API_KEY env var.")

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},  # enforce JSON mode
        )
        return resp.choices[0].message.content or ""

    def _chat_anthropic(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("anthropic package not installed. Run: pip install anthropic") from exc

        if not self.api_key:
            raise LLMError("Anthropic API key not set. Set ANTHROPIC_API_KEY env var.")

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class _MockLLMClient(LLMClient):
    """LLMClient that returns predefined responses; no network calls."""

    def __init__(self, responses: dict[str, Any] | list[dict[str, Any]]) -> None:
        # Don't call super().__init__() — no API key needed
        self.provider = "mock"
        self.model = "mock"
        self._responses = responses if isinstance(responses, list) else [responses]
        self._call_count = 0

    def chat_json(self, system: str, user: str, max_parse_retries: int = 2) -> dict[str, Any]:
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        logger.debug("[MockLLMClient] returning response #%d", idx)
        return dict(response)  # return a copy


# ── JSON extraction helpers ────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response.

    Handles:
    - Pure JSON strings
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON embedded in surrounding prose
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Find the outermost { ... } span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


# ── Public factory ─────────────────────────────────────────────────────────────

def build_llm_client(
    cfg: dict[str, Any],
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> "LLMClient | None":
    """Build an LLMClient from a config dict, returning None when LLM is disabled.

    This is the preferred factory for ``run_pipeline.py`` and scripts that
    read the ``llm`` section of ``config/default.yaml``.

    Args:
        cfg: The ``llm`` section of the YAML config (may be empty or missing).
        provider_override: CLI-supplied provider string; takes precedence over config.
        model_override: CLI-supplied model string; takes precedence over config.

    Returns:
        A configured ``LLMClient``, or ``None`` when provider is ``"none"`` or absent.

    Example::

        llm_client = build_llm_client(cfg.get("llm", {}), provider_override=args.llm_provider)
        report = run_experiment(config, llm_client=llm_client)
    """
    provider = provider_override or cfg.get("provider", "none")
    if not provider or provider == "none":
        logger.info("[build_llm_client] LLM disabled (provider=none)")
        return None

    model = model_override or cfg.get("model", "gpt-4o-mini")
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")

    client = LLMClient(
        provider=provider,
        model=model,
        api_key=cfg.get("api_key"),
        api_key_env=api_key_env,
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 1024),
        timeout_seconds=cfg.get("timeout_seconds", 30.0),
    )
    logger.info(
        "[build_llm_client] provider=%s  model=%s  api_key_env=%s",
        provider, model, api_key_env,
    )
    return client
