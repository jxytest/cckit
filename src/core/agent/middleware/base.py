"""Abstract base class for all agent middleware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from core.agent.schemas import AgentEvent, ExecutionContext

# Type alias for the inner async generator that middleware wraps.
# Signature: (prompt, options, collector) -> AsyncIterator[AgentEvent]
SdkQueryFunc = Any  # Callable[[str, Any, Any], AsyncIterator[AgentEvent]]


class AgentMiddleware(ABC):
    """Base class for agent execution middleware.

    Subclass and implement ``wrap()`` to intercept SDK query calls.
    The middleware chain executes in order::

        [Middleware_0] → [Middleware_1] → ... → [SDK query]

    Each middleware wraps the *next* callable, so the **first** middleware
    in the list is the outermost wrapper.
    """

    @abstractmethod
    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        collector: Any,
        ctx: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        """Wrap the next callable in the middleware chain.

        Parameters
        ----------
        next_call:
            The next middleware or the actual ``_run_sdk_query`` method.
        prompt:
            The prompt to send to the SDK.
        options:
            ``ClaudeAgentOptions`` instance.
        collector:
            ``StreamCollector`` instance.
        ctx:
            The current execution context.

        Yields
        ------
        AgentEvent
        """
        yield  # pragma: no cover — abstract
        ...
