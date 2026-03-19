"""SDK bridge — the only module that directly interacts with the claude-agent-sdk.

Extracted from the old ``executor.py``.  Provides a standalone async generator
that calls ``query()``, streams messages through a ``StreamCollector``, and
yields ``AgentEvent`` objects.

Session resume / fork is handled transparently via ``ClaudeAgentOptions.resume``
and ``ClaudeAgentOptions.fork_session`` — no special logic needed here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from cckit._engine.collector import StreamCollector
from cckit.exceptions import AgentExecutionError
from cckit.types import AgentEvent

logger = logging.getLogger(__name__)


async def run_sdk_query(
    prompt: str,
    options: Any,
    collector: StreamCollector,
) -> AsyncIterator[AgentEvent]:
    """Call ``query(prompt, options=options)`` and yield events via collector.

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
            events = collector.process_message(message)
            for event in events:
                yield event
    except Exception as exc:
        raise AgentExecutionError(
            f"SDK query failed: {exc}",
            detail=str(exc),
        ) from exc
