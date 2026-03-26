"""Execution state for SDK-backed agent runs."""

from __future__ import annotations

from typing import Any


class RunState:
    """Track the minimal state needed to build the final AgentResult."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.session_id: str = ""
        self.final_message: Any | None = None

    def observe(self, message: Any) -> None:
        """Update run state from a single SDK message."""
        from claude_agent_sdk import ResultMessage, SystemMessage  # noqa: WPS433

        session_id = getattr(message, "session_id", "") or ""
        if session_id:
            self.session_id = session_id

        if isinstance(message, ResultMessage):
            self.final_message = message
            return

        if isinstance(message, SystemMessage) and message.subtype == "init":
            data = getattr(message, "data", {}) or {}
            init_session_id = data.get("session_id", "")
            if init_session_id:
                self.session_id = init_session_id
