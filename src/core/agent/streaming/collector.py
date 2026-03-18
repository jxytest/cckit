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

        # SystemMessage check must come first because TaskStartedMessage,
        # TaskProgressMessage, etc. are subclasses of SystemMessage.
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
        """Handle SystemMessage and its Task* subclasses."""
        from claude_agent_sdk import (  # noqa: WPS433
            TaskNotificationMessage,
            TaskProgressMessage,
            TaskStartedMessage,
        )

        events: list[AgentEvent] = []

        # Check subclasses *before* the generic init branch (they are also
        # SystemMessage instances with subtype set).
        if isinstance(msg, TaskStartedMessage):
            events.append(
                AgentEvent(
                    event_type=AgentEventType.SUB_AGENT_START,
                    task_id=self.task_id,
                    data={
                        "sub_task_id": msg.task_id,
                        "description": msg.description,
                        "session_id": msg.session_id,
                        "tool_use_id": msg.tool_use_id,
                    },
                )
            )
        elif isinstance(msg, TaskProgressMessage):
            events.append(
                AgentEvent(
                    event_type=AgentEventType.SUB_AGENT_PROGRESS,
                    task_id=self.task_id,
                    data={
                        "sub_task_id": msg.task_id,
                        "description": msg.description,
                        "usage": msg.usage,
                        "last_tool_name": msg.last_tool_name,
                    },
                )
            )
        elif isinstance(msg, TaskNotificationMessage):
            events.append(
                AgentEvent(
                    event_type=AgentEventType.SUB_AGENT_END,
                    task_id=self.task_id,
                    data={
                        "sub_task_id": msg.task_id,
                        "status": msg.status,
                        "summary": msg.summary,
                        "usage": msg.usage,
                        "output_file": msg.output_file,
                    },
                )
            )
        elif hasattr(msg, "subtype") and msg.subtype == "init":
            data = getattr(msg, "data", {}) or {}
            self._session_id = data.get("session_id", "")

        return events

    def _handle_assistant(self, msg: Any) -> list[AgentEvent]:
        """Handle AssistantMessage — extract text, thinking, and tool blocks."""
        from claude_agent_sdk import (  # noqa: WPS433
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        events: list[AgentEvent] = []
        content_blocks = getattr(msg, "content", []) or []

        for block in content_blocks:
            if isinstance(block, TextBlock):
                text = block.text
                self._text_parts.append(text)
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.ASSISTANT_TEXT,
                        task_id=self.task_id,
                        text=text,
                    )
                )

            elif isinstance(block, ThinkingBlock):
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.THINKING,
                        task_id=self.task_id,
                        text=block.thinking,
                        data={"signature": block.signature},
                    )
                )

            elif isinstance(block, ToolUseBlock):
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.TOOL_USE,
                        task_id=self.task_id,
                        data={
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_use_id": block.id,
                        },
                    )
                )

            elif isinstance(block, ToolResultBlock):
                events.append(
                    AgentEvent(
                        event_type=AgentEventType.TOOL_RESULT,
                        task_id=self.task_id,
                        data={
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        },
                    )
                )

        # Surface assistant-level errors (e.g. rate_limit, billing_error)
        error = getattr(msg, "error", None)
        if error:
            events.append(
                AgentEvent(
                    event_type=AgentEventType.ERROR,
                    task_id=self.task_id,
                    text=str(error),
                    data={"error_type": str(error)},
                )
            )

        return events

    def _handle_result(self, msg: Any) -> list[AgentEvent]:
        """Handle ResultMessage — the final message from the SDK."""
        cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
        self._cost_usd = cost

        session_id = getattr(msg, "session_id", "") or ""
        if session_id:
            self._session_id = session_id

        return [
            AgentEvent(
                event_type=AgentEventType.RESULT,
                task_id=self.task_id,
                text=getattr(msg, "result", "") or "",
                data={
                    "cost_usd": cost,
                    "session_id": self._session_id,
                    "stop_reason": getattr(msg, "stop_reason", ""),
                    "duration_ms": getattr(msg, "duration_ms", 0),
                    "duration_api_ms": getattr(msg, "duration_api_ms", 0),
                    "num_turns": getattr(msg, "num_turns", 0),
                    "usage": getattr(msg, "usage", None),
                    "is_error": getattr(msg, "is_error", False),
                    "structured_output": getattr(msg, "structured_output", None),
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
