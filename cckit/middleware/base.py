"""Abstract base class for all middleware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from cckit.types import RunContext

# Type alias for the inner async generator that middleware wraps.
# Signature: (prompt, options, state) -> AsyncIterator[SDK Message]
SdkQueryFunc = Any  # Callable[[str, Any, Any], AsyncIterator[Any]]


class Middleware(ABC):
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
        state: Any,
        ctx: RunContext,
    ) -> AsyncGenerator[Any, None]:
        """Wrap the next callable in the middleware chain.

        Parameters
        ----------
        next_call:
            The next middleware or the actual SDK bridge ``run_sdk_query``.
        prompt:
            The prompt to send to the SDK.
        options:
            ``ClaudeAgentOptions`` instance.
        state:
            Mutable run state shared across the execution.
        ctx:
            The current run context.

        Yields
        ------
        SDK message object
        """
        yield  # pragma: no cover — abstract
        ...
