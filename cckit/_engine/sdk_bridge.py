"""SDK bridge — the only module that directly interacts with the claude-agent-sdk.

Extracted from the old ``executor.py``.  Provides a standalone async generator
that calls ``query()``, updates ``RunState``, and yields the SDK's native
message objects unchanged.

Session resume / fork is handled transparently via ``ClaudeAgentOptions.resume``
and ``ClaudeAgentOptions.fork_session`` — no special logic needed here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from cckit._engine.state import RunState
from cckit.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


async def run_sdk_query(
    prompt: str,
    options: Any,
    state: RunState,
) -> AsyncIterator[Any]:
    """Call ``query(prompt, options=options)`` and yield SDK messages.

    Uses the top-level ``query()`` async generator from ``claude_agent_sdk``
    instead of the stateful ``query`` context-manager approach.
    Session resume is handled transparently by ``ClaudeAgentOptions.resume``.
    """
    try:
        from claude_agent_sdk import query  # noqa: WPS433
    except ImportError as exc:
        raise AgentExecutionError(
            "claude-agent-sdk is not installed",
            detail=str(exc),
        ) from exc

    try:
        async for message in query(prompt=prompt, options=options):
            state.observe(message)
            yield message
    except Exception as exc:
        raise AgentExecutionError(
            f"SDK query failed: {exc}",
            detail=str(exc),
        ) from exc
