"""Stream collector — converts SDK messages into ``AgentEvent`` objects.

The SDK emits typed messages (``AssistantMessage``, ``ResultMessage``, etc.)
through its async iterator.  ``StreamCollector`` normalises these into our
domain ``AgentEvent`` type so the rest of the codebase never touches SDK
internals.
"""

from __future__ import annotations

import logging
from typing import Any

from core.agent.schemas import AgentEvent, AgentEventType

logger = logging.getLogger(__name__)


class StreamCollector:
    """Convert claude-agent-sdk stream messages into ``AgentEvent`` objects."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._text_parts: list[str] = []
        self._cost_usd: float = 0.0
        self._session_id: str = ""

    # ------------------------------------------------------------------
    # Public: call for every message from the SDK stream
    # ------------------------------------------------------------------

    def process_message(self, message: Any) -> list[AgentEvent]:
        """Process one SDK message and return zero or more ``AgentEvent``s.

        We import SDK types lazily so the rest of core/ doesn't depend on
        ``claude_agent_sdk`` at import time.
        """
        from claude_agent_sdk import (  # noqa: WPS433
            AssistantMessage,
            ResultMessage,
            SystemMessage,
        )

        events: list[AgentEvent] = []

        if isinstance(message, SystemMessage):
            events.extend(self._handle_system(message))
        elif isinstance(message, AssistantMessage):
            events.extend(self._handle_assistant(message))
        elif isinstance(message, ResultMessage):
            events.extend(self._handle_result(message))
        else:
            logger.debug("Unhandled SDK message type: %s", type(message).__name__)

        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_system(self, msg: Any) -> list[AgentEvent]:
        """Handle SystemMessage (e.g. init with session_id)."""
        events: list[AgentEvent] = []
        if hasattr(msg, "subtype") and msg.subtype == "init":
            data = getattr(msg, "data", {}) or {}
            self._session_id = data.get("session_id", "")
        return events

    def _handle_assistant(self, msg: Any) -> list[AgentEvent]:
        """Handle AssistantMessage — extract text and tool_use blocks."""
        events: list[AgentEvent] = []
        content_blocks = getattr(msg, "content", []) or []

        for block in content_blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", "")
                self._text_parts.append(text)
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.ASSISTANT_TEXT,
                        task_id=self.task_id,
                        text=text,
                    )
                )

            elif block_type == "tool_use":
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.TOOL_USE,
                        task_id=self.task_id,
                        data={
                            "tool_name": getattr(block, "name", ""),
                            "tool_input": getattr(block, "input", {}),
                            "tool_use_id": getattr(block, "id", ""),
                        },
                    )
                )

            elif block_type == "tool_result":
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.TOOL_RESULT,
                        task_id=self.task_id,
                        data={
                            "tool_use_id": getattr(block, "tool_use_id", ""),
                            "content": getattr(block, "content", ""),
                        },
                    )
                )

        return events

    def _handle_result(self, msg: Any) -> list[AgentEvent]:
        """Handle ResultMessage — the final message from the SDK."""
        result_text = getattr(msg, "result", "")
        cost = getattr(msg, "cost_usd", 0.0) or 0.0
        self._cost_usd = cost

        return [
            AgentEvent(
                event_type=AgentEventType.RESULT,
                task_id=self.task_id,
                text=result_text,
                data={
                    "cost_usd": cost,
                    "session_id": self._session_id,
                    "stop_reason": getattr(msg, "stop_reason", ""),
                    "duration_api_seconds": getattr(msg, "duration_api_seconds", 0),
                },
            )
        ]

    # ------------------------------------------------------------------
    # Summary accessors
    # ------------------------------------------------------------------

    @property
    def full_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    @property
    def session_id(self) -> str:
        return self._session_id
