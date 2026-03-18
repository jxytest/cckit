"""Data schemas for the agent subsystem.

These are pure data containers — no SDK dependency, no I/O.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from core.models import CustomModel

# ---------------------------------------------------------------------------
# Sub-agent declaration (passed to SDK as AgentDefinition)
# ---------------------------------------------------------------------------

class SubAgentConfig(CustomModel):
    """Declarative sub-agent that the parent agent *may* invoke via ``Task``."""

    name: str
    description: str
    tools: list[str] = Field(default_factory=list)
    system_prompt: str = ""


# ---------------------------------------------------------------------------
# Agent configuration (returned by BaseAgent.config())
# ---------------------------------------------------------------------------

class AgentConfig(CustomModel):
    """Static configuration for an agent type."""

    agent_type: str
    display_name: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    mcp_tool_names: list[str] = Field(default_factory=list)
    sub_agents: list[SubAgentConfig] = Field(default_factory=list)
    model: str = ""  # empty = inherit global AnthropicSettings.model
    max_tokens: int = 16384
    max_turns: int = 0  # 0 = use global AnthropicSettings.max_turns
    needs_workspace: bool = True
    needs_git_clone: bool = False
    skills: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Execution context (built at runtime, injected into agent)
# ---------------------------------------------------------------------------

class ExecutionContext(CustomModel):
    """Runtime context passed to an agent execution."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    workspace_dir: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    prompt: str = ""
    extra_params: dict[str, Any] = Field(default_factory=dict)
    git_repo_url: str = ""
    git_branch: str = ""
    resume_session_id: str = ""  # when set, resumes a previous SDK session
    fork_session: bool = False  # when resuming, create a new branch session


# ---------------------------------------------------------------------------
# Task status enum
# ---------------------------------------------------------------------------

class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

class AgentResult(CustomModel):
    """Final result returned after agent execution completes."""

    task_id: str
    agent_type: str
    status: TaskStatus = TaskStatus.COMPLETED
    result_text: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    is_error: bool = False
    error_message: str = ""
    session_id: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

class AgentEventType(StrEnum):
    ASSISTANT_TEXT = "assistant_text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    SUB_AGENT_START = "sub_agent_start"
    SUB_AGENT_PROGRESS = "sub_agent_progress"
    SUB_AGENT_END = "sub_agent_end"
    RESULT = "result"
    ERROR = "error"


class AgentEvent(CustomModel):
    """A single event emitted during agent execution (for streaming)."""

    event_type: AgentEventType
    task_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
