"""Concurrency middleware — limit parallel agent executions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from core.agent.middleware.base import AgentMiddleware, SdkQueryFunc
from core.agent.schemas import AgentEvent, ExecutionContext
from core.config import platform_settings

logger = logging.getLogger(__name__)


class ConcurrencyMiddleware(AgentMiddleware):
    """Limit the number of concurrent agent executions.

    Uses an ``asyncio.Semaphore`` to enforce
    ``PlatformSettings.max_concurrent_agents``.
    """

    def __init__(self, *, max_concurrent: int | None = None) -> None:
        limit = max_concurrent or platform_settings.max_concurrent_agents
        self._semaphore = asyncio.Semaphore(limit)
        self._limit = limit

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        collector: Any,
        ctx: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        logger.debug(
            "Acquiring concurrency slot (limit=%d) for task %s",
            self._limit,
            ctx.task_id,
        )
        async with self._semaphore:
            async for event in next_call(prompt, options, collector):
                yield event
