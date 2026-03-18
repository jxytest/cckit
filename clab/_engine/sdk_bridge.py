"""SDK bridge — the only module that directly interacts with ClaudeSDKClient.

Extracted from the old ``executor.py``.  Provides a standalone async generator
that opens an SDK session, streams messages through a ``StreamCollector``, and
yields ``AgentEvent`` objects.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from clab._engine.collector import StreamCollector
from clab.exceptions import AgentExecutionError
from clab.types import AgentEvent

logger = logging.getLogger(__name__)


async def run_sdk_query(
    prompt: str,
    options: Any,
    collector: StreamCollector,
) -> AsyncIterator[AgentEvent]:
    """Open a ``ClaudeSDKClient`` session and yield events via collector.

    Uses one-shot semantics: connect → receive full response → disconnect.
    Resume is handled transparently by ``ClaudeAgentOptions.resume``.
    """
    try:
        from claude_agent_sdk import ClaudeSDKClient  # noqa: WPS433
    except ImportError as exc:
        raise AgentExecutionError(
            "claude-agent-sdk is not installed",
            detail=str(exc),
        ) from exc

    client: Any = None
    try:
        client = ClaudeSDKClient(options=options)
        await client.connect(prompt=prompt)

        async for message in client.receive_response():
            events = collector.process_message(message)
            for event in events:
                yield event
    except Exception as exc:
        raise AgentExecutionError(
            f"SDK query failed: {exc}",
            detail=str(exc),
        ) from exc
    finally:
        if client is not None:
            await client.disconnect()
