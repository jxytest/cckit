"""Retry middleware — exponential backoff for transient SDK failures."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from cckit.exceptions import AgentExecutionError
from cckit.middleware.base import Middleware, SdkQueryFunc
from cckit.types import RunContext

logger = logging.getLogger(__name__)


class RetryMiddleware(Middleware):
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

    # Patterns that indicate a *rate-limit / overload* failure — transient but
    # benefiting from a longer backoff than a generic connection blip.
    # ``RateLimitEvent`` is a streamed message (observed in TracingMiddleware)
    # and never reaches this middleware as an exception; these patterns catch
    # the HTTP-layer 429 / overloaded errors that surface as
    # ``AgentExecutionError``. HTTP status codes are matched with word
    # boundaries so a port like ``:5290`` is not misread as status 529.
    _RATE_LIMIT_MARKERS: tuple[str, ...] = (
        "rate_limit",
        "rate limit",
        "overloaded",
        "too many requests",
    )
    _RATE_LIMIT_STATUS_RE = re.compile(r"(?:^|\D)(429|529)(?=\D|$)")

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        rate_limit_base_delay: float = 5.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        # Longer initial backoff for rate-limit errors (providers typically
        # need seconds, not sub-second, to clear a 429).
        self.rate_limit_base_delay = rate_limit_base_delay

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        state: Any,
        ctx: RunContext,
    ) -> AsyncIterator[Any]:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async for message in next_call(prompt, options, state):
                    yield message
                return  # success — exit retry loop
            except AgentExecutionError as exc:
                last_exc = exc
                err_lower = str(exc).lower()

                # Don't retry permanent errors
                if any(marker in err_lower for marker in self._PERMANENT_MARKERS):
                    raise

                if attempt < self.max_retries - 1:
                    is_rate_limited = any(
                        marker in err_lower for marker in self._RATE_LIMIT_MARKERS
                    ) or bool(self._RATE_LIMIT_STATUS_RE.search(err_lower))
                    # Rate-limit errors use a longer base delay; generic
                    # transient errors use the standard exponential backoff.
                    base = self.rate_limit_base_delay if is_rate_limited else self.base_delay
                    delay = min(base * (2**attempt), self.max_delay)
                    kind = "rate-limited" if is_rate_limited else "transient"
                    logger.warning(
                        "SDK query attempt %d/%d failed (%s), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries,
                        kind,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise AgentExecutionError(
            f"SDK query failed after {self.max_retries} attempts",
            detail=str(last_exc),
        ) from last_exc
