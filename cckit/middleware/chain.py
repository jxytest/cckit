"""Middleware chain assembly.

Wraps the innermost SDK call with the configured middleware stack.  The
first middleware in the list becomes the outermost wrapper.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from cckit._engine.sdk_bridge import run_sdk_query

if TYPE_CHECKING:
    from cckit.middleware.base import Middleware
    from cckit.types import RunContext


def build_middleware_chain(
        middlewares: list[Middleware],
        ctx: RunContext,
) -> Any:
    """Wrap ``run_sdk_query`` with *middlewares*.

    Returns a callable with signature ``(prompt, options, state)`` yielding
    SDK messages.
    """

    # The innermost function — actual SDK call
    async def inner(
            prompt: str, options: Any, state: Any
    ) -> AsyncIterator[Any]:
        async for message in run_sdk_query(prompt, options, state):
            yield message

    current = inner

    # Wrap from inside out (last middleware wraps first)
    for mw in reversed(middlewares):

        def make_wrapper(middleware: Middleware, next_fn: Any) -> Any:
            async def wrapper(
                    prompt: str, options: Any, state: Any
            ) -> AsyncIterator[Any]:
                async for message in middleware.wrap(
                        next_fn, prompt, options, state, ctx
                ):
                    yield message

            return wrapper

        current = make_wrapper(mw, current)

    return current
