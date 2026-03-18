"""Logging middleware — record SDK query duration and cost."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from core.agent.middleware.base import AgentMiddleware, SdkQueryFunc
from core.agent.schemas import AgentEvent, ExecutionContext

logger = logging.getLogger(__name__)


class LoggingMiddleware(AgentMiddleware):
    """Log SDK query start/end with duration and cost summary."""

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        collector: Any,
        ctx: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        start = time.monotonic()
        logger.info("SDK query started for task %s", ctx.task_id)

        event_count = 0
        try:
            async for event in next_call(prompt, options, collector):
                event_count += 1
                yield event
        finally:
            elapsed = time.monotonic() - start
            logger.info(
                "SDK query ended for task %s: %.2fs, %d events, cost=$%.4f",
                ctx.task_id,
                elapsed,
                event_count,
                collector.cost_usd,
            )
