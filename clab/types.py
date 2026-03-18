"""Data types for clab.

Contains all data containers: ModelConfig, LiteLlm, WorkspaceConfig,
RunContext, AgentResult, AgentEvent, RunnerConfig, SandboxOptions, etc.

These are pure data — no SDK dependency, no I/O.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from clab._models import CustomModel

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class ModelConfig(CustomModel):
    """Direct Anthropic API model configuration.

    Usage::

        ModelConfig(model="claude-sonnet-4-6")
        ModelConfig(model="claude-sonnet-4-6", api_key="sk-...", base_url="https://...")
    """

    model: str = "claude-sonnet-4-6"
    api_key: str = ""  # empty = use ANTHROPIC_API_KEY env var
    base_url: str = ""  # empty = use ANTHROPIC_BASE_URL env var or default
    max_tokens: int = 16384
    max_turns: int = 50
    permission_mode: str = "bypassPermissions"
    timeout_seconds: int = 300


class LiteLlm(CustomModel):
    """Bridge to any model via LiteLLM proxy.

    LiteLLM runs as an OpenAI-compatible proxy server, accepting OpenAI-format
    requests and routing to 100+ model providers.

    Usage::

        # 1. Start LiteLLM proxy: litellm --model openai/gemini-3-pro
        # 2. Reference in Agent:
        Agent(model=LiteLlm(model="openai/gemini-3-pro", api_base="http://localhost:4000"))

    How it works:
    - LiteLlm is converted to ModelConfig, with base_url pointing to LiteLLM proxy
    - claude-agent-sdk connects to LiteLLM proxy via ANTHROPIC_BASE_URL
    - LiteLLM routes the request to the actual model provider
    """

    model: str  # LiteLLM model identifier, e.g. "openai/gemini-3-pro"
    api_base: str = "http://localhost:4000"  # LiteLLM proxy address
    api_key: str = ""  # LiteLLM proxy API key
    max_tokens: int = 16384
    max_turns: int = 50
    permission_mode: str = "bypassPermissions"
    timeout_seconds: int = 300

    def to_model_config(self) -> ModelConfig:
        """Convert to ModelConfig for internal use by Runner."""
        return ModelConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.api_base,
            max_tokens=self.max_tokens,
            max_turns=self.max_turns,
            permission_mode=self.permission_mode,
            timeout_seconds=self.timeout_seconds,
        )


# ---------------------------------------------------------------------------
# Workspace configuration
# ---------------------------------------------------------------------------


class WorkspaceConfig(CustomModel):
    """Workspace configuration — controls agent execution environment isolation.

    This was previously ``needs_workspace`` / ``needs_git_clone`` on the Agent;
    now it's part of RunContext so execution environment is controlled by the
    caller, not the agent definition.
    """

    enabled: bool = True  # whether to create an isolated workspace directory
    git_clone: bool = False  # whether to clone a git repo into the workspace


# ---------------------------------------------------------------------------
# Execution context (runtime)
# ---------------------------------------------------------------------------


class RunContext(CustomModel):
    """Runtime context — controls all runtime parameters for a single execution.

    Replaces the old ``ExecutionContext``. Key changes:
    - ``extra_params`` renamed to ``params`` (shorter, clearer)
    - workspace/git_clone moved here from Agent definition
    - ``prompt`` is the primary instruction field
    """

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    # Workspace
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    workspace_dir: Path | None = None  # injected by Runner, or manually specified

    # Git
    git_repo_url: str = ""
    git_branch: str = ""

    # Session resume
    resume_session_id: str = ""  # when set, resumes a previous SDK session
    fork_session: bool = False  # when resuming, create a new branch session


# ---------------------------------------------------------------------------
# Task status
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


# ---------------------------------------------------------------------------
# Sandbox options (for RunnerConfig)
# ---------------------------------------------------------------------------


class SandboxOptions(CustomModel):
    """Sandbox / workspace isolation configuration."""

    enabled: bool = False
    workspace_root: Path = Path("/tmp/clab_workspaces")
    network_allowed_hosts: list[str] = Field(default_factory=list)
    max_file_size_mb: int = 50


# ---------------------------------------------------------------------------
# Runner configuration
# ---------------------------------------------------------------------------


class RunnerConfig(CustomModel):
    """Runner execution configuration — replaces the old global singletons.

    Users can either construct explicitly or use ``from_env()`` to read
    environment variables (same vars as the old Settings classes).
    """

    default_model: ModelConfig = Field(default_factory=ModelConfig)
    sandbox: SandboxOptions = Field(default_factory=SandboxOptions)
    skills_dir: Path = Path("/opt/agent-platform/skills")
    max_concurrent_agents: int = 5
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> RunnerConfig:
        """Create a RunnerConfig by reading environment variables.

        Reads ANTHROPIC_*, SANDBOX_*, PLATFORM_* env vars the same way
        the old singleton Settings classes did.  This is a convenience for
        existing deployments — users who want full control should construct
        RunnerConfig directly.
        """
        from pydantic_settings import BaseSettings

        class _EnvModel(BaseSettings):
            model_config = {"env_prefix": "ANTHROPIC_", "case_sensitive": False}
            api_key: str = ""
            base_url: str = ""
            model: str = "claude-sonnet-4-6"
            max_tokens: int = 16384
            max_turns: int = 50
            permission_mode: str = "bypassPermissions"
            timeout_seconds: int = 300

        class _EnvSandbox(BaseSettings):
            model_config = {"env_prefix": "SANDBOX_", "case_sensitive": False}
            enabled: bool = False
            workspace_root: Path = Path("/tmp/clab_workspaces")
            network_allowed_hosts: list[str] = []
            max_file_size_mb: int = 50

        class _EnvPlatform(BaseSettings):
            model_config = {"env_prefix": "PLATFORM_", "case_sensitive": False}
            debug: bool = False
            log_level: str = "INFO"
            max_concurrent_agents: int = 5
            skills_dir: Path = Path("/opt/agent-platform/skills")

        env_m = _EnvModel()
        env_s = _EnvSandbox()
        env_p = _EnvPlatform()

        return cls(
            default_model=ModelConfig(
                model=env_m.model,
                api_key=env_m.api_key,
                base_url=env_m.base_url,
                max_tokens=env_m.max_tokens,
                max_turns=env_m.max_turns,
                permission_mode=env_m.permission_mode,
                timeout_seconds=env_m.timeout_seconds,
            ),
            sandbox=SandboxOptions(
                enabled=env_s.enabled,
                workspace_root=env_s.workspace_root,
                network_allowed_hosts=env_s.network_allowed_hosts,
                max_file_size_mb=env_s.max_file_size_mb,
            ),
            skills_dir=env_p.skills_dir,
            max_concurrent_agents=env_p.max_concurrent_agents,
            debug=env_p.debug,
            log_level=env_p.log_level,
        )
