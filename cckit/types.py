"""Data types for cckit.

Contains all data containers: ModelConfig, LiteLlm, GitConfig, WorkspaceConfig,
RunContext, AgentResult, AgentEvent, RunnerConfig, SandboxOptions, etc.

These are pure data — no SDK dependency, no I/O.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from cckit._models import CustomModel

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
# Git configuration
# ---------------------------------------------------------------------------


class GitConfig(CustomModel):
    """Git repository configuration — first-class credentials + clone options.

    Centralizes everything related to git operations (clone, push, etc.)
    so that credentials are explicitly managed and never leak into the
    Agent subprocess environment.

    Usage::

        # Token-based auth (most common — GitLab/GitHub PAT)
        GitConfig(
            repo_url="https://gitlab.com/team/project.git",
            token="glpat-xxxx",
            branch="main",
        )

        # Custom env-based auth (GIT_ASKPASS, SSH keys, etc.)
        GitConfig(
            repo_url="git@github.com:team/project.git",
            auth_env={"GIT_SSH_COMMAND": "ssh -i /path/to/key"},
        )

    How credentials are resolved (by Runner):
        1. If ``token`` is set, Runner injects a ``GIT_ASKPASS`` helper
           that echoes the token — works for any HTTPS remote.
        2. If ``auth_env`` is set, those vars are passed directly to
           the git subprocess as ``extra_env``.
        3. Both can coexist — ``token`` is applied first, then
           ``auth_env`` is merged on top (explicit wins).
    """

    repo_url: str = ""
    branch: str = ""
    token: str = ""  # PAT / deploy token — Runner converts to GIT_ASKPASS
    auth_env: dict[str, str] = Field(default_factory=dict)  # extra git-only env

    # Clone options
    clone: bool = False  # whether to clone during workspace setup
    depth: int = 1  # --depth for shallow clone, 0 = full history

    # Internal: cached path to the temporary askpass helper script.
    # Created on first call to build_git_env() and reused afterwards.
    _askpass_script: str | None = PrivateAttr(default=None)

    def build_git_env(self) -> dict[str, str]:
        """Build the environment dict for git subprocesses.

        This is the **single source of truth** for git credential
        injection — used by both clone and push.  The result is
        **never** merged into the Agent subprocess environment.

        The temporary helper script is created once and cached so that
        multiple calls (clone, push, …) reuse the same file.  Call
        :meth:`cleanup_askpass` when the script is no longer needed.
        """
        env: dict[str, str] = {}

        if self.token:
            # Lazily create the askpass helper script.
            if self._askpass_script is None:
                self._askpass_script = self._create_askpass_script(self.token)
            env["GIT_ASKPASS"] = self._askpass_script
            env["GIT_TERMINAL_PROMPT"] = "0"

        # auth_env overrides / supplements — user knows best
        env.update(self.auth_env)
        return env

    def cleanup_askpass(self) -> None:
        """Remove the temporary askpass helper script if it exists."""
        if self._askpass_script is not None:
            import os

            try:
                os.unlink(self._askpass_script)
            except OSError:
                pass
            self._askpass_script = None

    @staticmethod
    def _create_askpass_script(token: str) -> str:
        """Write a temporary script that prints *token* to stdout.

        ``GIT_ASKPASS`` is invoked by git as::

            $GIT_ASKPASS "Password for 'https://…': "

        The script ignores the prompt argument and simply echoes the
        token.  This is the standard, portable approach that works for
        any HTTPS remote without embedding the token in the URL or
        persisting it in ``.git/config``.
        """
        import os
        import stat
        import sys
        import tempfile

        if sys.platform == "win32":
            fd, path = tempfile.mkstemp(suffix=".bat", prefix="cckit_askpass_")
            with os.fdopen(fd, "w") as f:
                f.write(f"@echo off\necho {token}\n")
        else:
            fd, path = tempfile.mkstemp(suffix=".sh", prefix="cckit_askpass_")
            with os.fdopen(fd, "w") as f:
                f.write(f"#!/bin/sh\necho '{token}'\n")
            os.chmod(path, stat.S_IRWXU)  # 0o700

        return path


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


# ---------------------------------------------------------------------------
# Execution context (runtime)
# ---------------------------------------------------------------------------


class RunContext(CustomModel):
    """Runtime context — controls all runtime parameters for a single execution.

    Key design choices:
    - ``params`` — business data for instruction templates
    - ``env`` — environment variables for the **Agent subprocess only**
    - ``git`` — all git config (URL, branch, credentials, clone options)

    Credentials isolation:
        ``git.token`` / ``git.auth_env`` are **never** injected into the
        Agent subprocess.  They are only passed to git CLI subprocesses
        by Runner.  This prevents the Agent (which can run arbitrary
        Bash commands) from reading secrets via ``env``.
    """

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    # Workspace
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    workspace_dir: Path | None = None  # injected by Runner, or manually specified

    # Git — first-class configuration
    git: GitConfig = Field(default_factory=GitConfig)

    # Session resume
    resume_session_id: str = ""  # when set, resumes a previous SDK session
    fork_session: bool = False  # when resuming, create a new branch session

    # ------------------------------------------------------------------
    # Backward compatibility — deprecated, will be removed in v1.0
    # ------------------------------------------------------------------

    git_repo_url: str = ""
    git_branch: str = ""

    def _resolved_git(self) -> GitConfig:
        """Return the effective GitConfig, merging deprecated fields.

        Priority: ``git`` (new) > ``git_repo_url``/``git_branch`` (old).
        If the new ``git`` field has no ``repo_url`` set but the old
        ``git_repo_url`` is present, upgrade transparently.
        """
        if self.git.repo_url:
            return self.git

        if not self.git_repo_url:
            return self.git

        # Upgrade from deprecated fields
        return self.git.model_copy(update={
            "repo_url": self.git_repo_url,
            "branch": self.git_branch or self.git.branch,
            "clone": self.git.clone or bool(self.git_repo_url),
        })


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
# Stream result wrapper
# ---------------------------------------------------------------------------


class _ResultHolder:
    """Mutable container shared between the async generator and StreamResult.

    This is an internal implementation detail — not a data model, so it's
    a plain class (same pattern as Runner, StreamCollector, WorkspaceManager).
    """

    __slots__ = ("result", "workspace_dir")

    def __init__(self) -> None:
        self.result: AgentResult | None = None
        self.workspace_dir: Path | None = None


class StreamResult:
    """Async-iterable wrapper around an agent execution stream.

    Returned by :meth:`Runner.run_stream`.  Implements ``__aiter__`` so
    it is a drop-in replacement for ``AsyncIterator[AgentEvent]``::

        # Existing usage (unchanged):
        async for event in runner.run_stream(agent, ctx):
            print(event.text)

        # New usage — access the final result after iteration:
        stream = runner.run_stream(agent, ctx)
        async for event in stream:
            print(event.text)
        final = stream.result  # same object the on_after callback received
    """

    def __init__(
        self,
        aiter: AsyncIterator[AgentEvent],
        holder: _ResultHolder,
    ) -> None:
        self._aiter = aiter
        self._holder = holder

    def __aiter__(self) -> StreamResult:
        return self

    async def __anext__(self) -> AgentEvent:
        return await self._aiter.__anext__()

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after the stream ends.

        This is the **same object instance** that lifecycle callbacks
        (``on_after``) receive, so any mutations (e.g. writing to
        ``result.extra``) are visible here.

        Returns ``None`` if the stream has not been fully consumed yet
        or was interrupted before a result could be built.
        """
        return self._holder.result


# ---------------------------------------------------------------------------
# Sandbox options (for RunnerConfig)
# ---------------------------------------------------------------------------


class SandboxOptions(CustomModel):
    """Sandbox / workspace isolation configuration."""

    enabled: bool = False
    workspace_root: Path = Path("/tmp/cckit_workspaces")
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
            workspace_root: Path = Path("/tmp/cckit_workspaces")
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
