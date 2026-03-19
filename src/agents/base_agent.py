"""Base class for all pipeline agents.

Provides:
- Structured logging (task start / success / failure with timing)
- Retry with exponential backoff
- Timeout enforcement (thread-based, cross-platform)
- ``as_autogen_tool()`` shim for future AutoGen ConversableAgent integration

Every concrete agent must implement ``_run(input)`` and nothing else.
The public ``run(input)`` method applies retry + logging automatically.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Generic, TypeVar

In = TypeVar("In")
Out = TypeVar("Out")

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Raised when an agent fails all retry attempts."""

    def __init__(self, agent_name: str, cause: Exception) -> None:
        self.agent_name = agent_name
        self.cause = cause
        super().__init__(f"[{agent_name}] failed: {cause}")


class BaseAgent(ABC, Generic[In, Out]):
    """Abstract pipeline agent with retry, timeout, and structured logging.

    Subclasses implement ``_run(input: In) -> Out`` and optionally
    override ``name``, ``max_retries``, and ``timeout_seconds``.
    """

    name: str = "BaseAgent"
    max_retries: int = 2
    timeout_seconds: float = 300.0

    def run(self, input: In) -> Out:
        """Execute the agent with automatic retry and logging.

        Args:
            input: Agent-specific input (a Pydantic model).

        Returns:
            Agent-specific output (a Pydantic model).

        Raises:
            AgentError: if all retry attempts fail.
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                logger.info("[%s] attempt %d/%d – starting", self.name, attempt + 1, self.max_retries + 1)
                result = self._run_with_timeout(input)
                elapsed = time.monotonic() - t0
                logger.info("[%s] completed in %.2fs", self.name, elapsed)
                return result

            except FuturesTimeoutError:
                elapsed = time.monotonic() - t0
                last_exc = TimeoutError(
                    f"[{self.name}] timed out after {elapsed:.1f}s (limit={self.timeout_seconds}s)"
                )
                logger.warning("[%s] timeout on attempt %d", self.name, attempt + 1)

            except Exception as exc:
                elapsed = time.monotonic() - t0
                last_exc = exc
                logger.warning(
                    "[%s] error on attempt %d (%.2fs): %s: %s",
                    self.name, attempt + 1, elapsed, type(exc).__name__, exc,
                )

            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.info("[%s] retrying in %ds …", self.name, backoff)
                time.sleep(backoff)

        logger.error("[%s] all %d attempts failed. Last error: %s", self.name, self.max_retries + 1, last_exc)
        raise AgentError(self.name, last_exc)  # type: ignore[arg-type]

    def _run_with_timeout(self, input: In) -> Out:
        """Run ``_run`` in a thread so we can enforce a wall-clock timeout."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._run, input)
            return future.result(timeout=self.timeout_seconds)

    @abstractmethod
    def _run(self, input: In) -> Out:
        """Core agent logic. Must be implemented by every subclass."""
        ...

    def as_autogen_tool(self) -> dict[str, Any]:
        """Return a description dict suitable for AutoGen tool registration.

        When AutoGen's ``ConversableAgent`` integration is added in v2,
        this method provides the metadata needed by ``register_for_llm``
        and ``register_for_execution``.

        Example::

            agent = DataAgent()
            tool_spec = agent.as_autogen_tool()
            # pass tool_spec to ConversableAgent.register_for_llm(...)

        Returns:
            Dict with keys ``name``, ``description``, ``callable``.
        """
        return {
            "name": self.name,
            "description": next(iter((self.__doc__ or "").strip().splitlines()), self.name),
            "callable": self.run,
        }
