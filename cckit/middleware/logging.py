"""Logging middleware — record SDK query duration and cost."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from cckit.middleware.base import Middleware, SdkQueryFunc
from cckit.types import RunContext

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """Log SDK query start/end with duration and cost summary."""

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        state: Any,
        ctx: RunContext,
    ) -> AsyncIterator[Any]:
        start = time.monotonic()
        logger.info("SDK query started for task %s", ctx.task_id)

        message_count = 0
        try:
            async for message in next_call(prompt, options, state):
                message_count += 1
                yield message
        finally:
            elapsed = time.monotonic() - start
            final_message = state.final_message
            cost_usd = getattr(final_message, "total_cost_usd", 0.0) or 0.0
            logger.info(
                "SDK query ended for task %s: %.2fs, %d messages, cost=$%.4f",
                ctx.task_id,
                elapsed,
                message_count,
                cost_usd,
            )
