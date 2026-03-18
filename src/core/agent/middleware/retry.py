"""Retry middleware — exponential backoff for transient SDK failures."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from core.agent.middleware.base import AgentMiddleware, SdkQueryFunc
from core.agent.schemas import AgentEvent, AgentEventType, ExecutionContext
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class RetryMiddleware(AgentMiddleware):
    """Retry failed SDK queries with exponential backoff.

    Only retries on transient errors (connection, timeout).  Permanent errors
    (invalid API key, model not found) are raised immediately.
    """

    # Exception substrings that indicate a *permanent* failure — never retry.
    _PERMANENT_MARKERS: tuple[str, ...] = (
        "invalid_api_key",
        "authentication",
        "permission",
        "not_found",
    )

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        collector: Any,
        ctx: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async for event in next_call(prompt, options, collector):
                    yield event
                return  # success — exit retry loop
            except AgentExecutionError as exc:
                last_exc = exc
                err_lower = str(exc).lower()

                # Don't retry permanent errors
                if any(marker in err_lower for marker in self._PERMANENT_MARKERS):
                    raise

                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(
                        "SDK query attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    yield AgentEvent(
                        event_type=AgentEventType.ERROR,
                        task_id=ctx.task_id,
                        text=f"Retry {attempt + 1}/{self.max_retries}: {exc}",
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise AgentExecutionError(
            f"SDK query failed after {self.max_retries} attempts",
            detail=str(last_exc),
        ) from last_exc
