"""Execution state for SDK-backed agent runs."""

from __future__ import annotations

from typing import Any


class RunState:
    """Track the minimal state needed to build the final AgentResult."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.session_id: str = ""
        self.final_message: Any | None = None
        self._stderr_lines: list[str] = []

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

    def observe_stderr(self, line: str) -> None:
        """Track recent Claude CLI stderr lines for error reporting."""
        line = line.rstrip()
        if not line:
            return

        self._stderr_lines.append(line)
        if len(self._stderr_lines) > 40:
            self._stderr_lines = self._stderr_lines[-40:]

    @property
    def stderr_text(self) -> str:
        """Return buffered stderr as a single string."""
        return "\n".join(self._stderr_lines).strip()
