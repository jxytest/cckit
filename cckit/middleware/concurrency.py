"""Concurrency middleware — limit parallel agent executions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from cckit.middleware.base import Middleware, SdkQueryFunc
from cckit.types import RunContext

logger = logging.getLogger(__name__)


class ConcurrencyMiddleware(Middleware):
    """Limit the number of concurrent agent executions.

    Uses an ``asyncio.Semaphore`` to enforce the limit.
    No global configuration dependency — pass the limit explicitly.
    """

    def __init__(self, *, max_concurrent: int = 5) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._limit = max_concurrent

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        state: Any,
        ctx: RunContext,
    ) -> AsyncIterator[Any]:
        logger.debug(
            "Acquiring concurrency slot (limit=%d) for task %s",
            self._limit,
            ctx.task_id,
        )
        async with self._semaphore:
            async for message in next_call(prompt, options, state):
                yield message
