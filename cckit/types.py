"""Data types for cckit.

Contains all data containers: ModelConfig, GitConfig, WorkspaceConfig,
RunContext, AgentResult, RunnerConfig, SandboxOptions, etc.

These are pure data — no SDK dependency, no I/O.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator

from cckit._models import CustomModel

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

if TYPE_CHECKING:
    from claude_agent_sdk import Message as SDKMessage
    from claude_agent_sdk import ResultMessage as SDKResultMessage
else:
    SDKMessage = Any
    SDKResultMessage = Any

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


TransportProtocol = Literal["chat", "responses", "anthropic"]


def _detect_model_base_url_protocol(value: str) -> TransportProtocol | None:
    """Infer the target protocol from a full endpoint URL, if present."""
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    if normalized.endswith("/v1/chat/completions") or normalized.endswith("/chat/completions"):
        return "chat"
    if normalized.endswith("/v1/responses") or normalized.endswith("/responses"):
        return "responses"
    if normalized.endswith("/v1/messages") or normalized.endswith("/messages"):
        return "anthropic"
    return None


def _normalize_model_base_url(value: str) -> str:
    """Normalize a provider API base URL.

    Users often paste a full endpoint path instead of an API base. Keep the
    normalization conservative and only trim well-known suffixes that Claude /
    LiteLLM callers commonly provide by mistake.
    """
    normalized = value.strip().rstrip("/")
    if not normalized:
        return ""
    suffix_rewrites = (
        ("/v1/messages", ""),
        ("/messages", ""),
        ("/v1/chat/completions", "/v1"),
        ("/chat/completions", ""),
    )
    for suffix, replacement in suffix_rewrites:
        if normalized.endswith(suffix):
            normalized = f"{normalized[: -len(suffix)]}{replacement}"
            break
    return normalized.rstrip("/")


def _normalize_anthropic_base_url(value: str) -> str:
    """Backward-compatible alias for older internal imports."""
    return _normalize_model_base_url(value)


class ModelConfig(CustomModel):
    """Unified model configuration interpreted with LiteLLM semantics.

    Usage::

        ModelConfig(model="anthropic/claude-sonnet-4-6")
        ModelConfig(model="openai/gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
        ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-...", base_url="https://gateway.example.com")

    Notes::

        ``base_url`` is intentionally kept as the public field name, but it is
        passed to LiteLLM as ``api_base``.
    """

    model: str = "anthropic/claude-sonnet-4-6"
    api_key: str = ""  # empty = let provider-specific env/defaults resolve it
    base_url: str = ""  # provider or gateway API base
    endpoint_protocol: TransportProtocol | None = Field(default=None, exclude=True)
    max_tokens: int = 16384
    max_turns: int = 50
    timeout_seconds: int = 300
    # Per-token pricing override (USD). When set, takes priority over LiteLLM's
    # built-in price table. Cache token costs are handled automatically by LiteLLM
    # using the model's standard cache pricing ratios.
    # Example: $1/M input, $2/M output → input_cost_per_token=1e-6, output_cost_per_token=2e-6
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_base_url_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        raw_base_url = str(data.get("base_url", "") or "")
        if raw_base_url:
            data["endpoint_protocol"] = _detect_model_base_url_protocol(raw_base_url)
            data["base_url"] = _normalize_model_base_url(raw_base_url)
        return data

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        return _normalize_model_base_url(value)


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

            with suppress(OSError):
                os.unlink(self._askpass_script)
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

    Fields
    ------
    enabled:
        Whether to create an isolated workspace directory.
    keep:
        When ``True``, the workspace directory is **never deleted** after
        execution — it will be suspended instead, allowing a subsequent
        ``RunContext(resume_session_id=..., workspace_dir=...)`` to reuse it.

        When ``False`` (the default), the workspace is cleaned up (deleted)
        on successful completion and suspended on failure/cancellation.

        Set this to ``True`` on the *first* execution whenever you plan to
        resume the session later::

            # First run — keep workspace for resume
            ctx1 = RunContext(
                prompt="Fix bug X",
                workspace=WorkspaceConfig(enabled=True, keep=True),
                git=GitConfig(repo_url="...", clone=True),
            )
            result1 = await runner.run(agent, ctx1)

            # Second run — resume from saved session
            ctx2 = RunContext(
                prompt="Add unit tests",
                resume_session_id=result1.session_id,
                workspace_dir=ctx1.workspace_dir,
            )
            result2 = await runner.run(agent, ctx2)
    """

    enabled: bool = True  # whether to create an isolated workspace directory
    keep: bool = False    # when True, always suspend instead of cleaning up


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
    # Optional per-run model override; credentials still come from Runner/Agent config.
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    user: str | None = None  # OS username for the Claude CLI subprocess, not business user id
    include_partial_messages: bool = False
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

    def resolved_git(self) -> GitConfig:
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
    output_text: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    is_error: bool = False
    error_message: str = ""
    session_id: str = ""
    stop_reason: str = ""
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    final_message: SDKResultMessage | None = Field(default=None, exclude=True)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a dict suitable for persisting execution results to a database.

        Maps AgentResult fields to common column names used by turn/task models.
        """
        return {
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'output_text': self.output_text or None,
            'error_message': self.error_message or None,
            'stop_reason': self.stop_reason or None,
            'cost_usd': self.cost_usd,
            'usage_payload': self.usage,
            'is_error': self.is_error,
            'session_id': self.session_id or None,
        }


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
    it is a drop-in replacement for the SDK's ``AsyncIterator[Message]``::

        # Existing usage (unchanged):
        async for message in runner.run_stream(agent, ctx):
            print(message)

        # New usage — access the final result after iteration:
        stream = runner.run_stream(agent, ctx)
        async for message in stream:
            print(message)
        final = stream.result  # same object the on_after callback received

        # Abort from another coroutine:
        stream.abort()  # cancels the consumer task, triggers cleanup
    """

    def __init__(
        self,
        aiter: AsyncIterator[SDKMessage],
        holder: _ResultHolder,
    ) -> None:
        self._aiter = aiter
        self._holder = holder
        self._aborted = False
        self._consumer_task: Any | None = None  # asyncio.Task set during iteration

    def __aiter__(self) -> StreamResult:
        return self

    async def __anext__(self) -> SDKMessage:
        if self._aborted:
            raise asyncio.CancelledError("Stream aborted")
        self._consumer_task = asyncio.current_task()
        return await self._aiter.__anext__()

    async def aclose(self) -> None:
        """Close the underlying async generator, triggering its finally block.

        Called automatically by ``async for`` on early exit (exception or
        break).  Also safe to call manually for explicit cleanup.
        """
        aclose_fn = getattr(self._aiter, 'aclose', None)
        if aclose_fn is not None:
            await aclose_fn()

    def abort(self) -> None:
        """Abort the running stream. Safe to call from any coroutine. Idempotent.

        Sets the aborted flag and cancels the consumer's asyncio task,
        causing ``CancelledError`` to propagate through the generator chain
        and trigger proper cleanup (including killing subprocesses).
        """
        if self._aborted:
            return
        self._aborted = True
        if self._consumer_task is not None and not self._consumer_task.done():
            self._consumer_task.cancel()

    @property
    def is_aborted(self) -> bool:
        """Whether :meth:`abort` has been called."""
        return self._aborted

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
# Context window & auto-compact configuration
# ---------------------------------------------------------------------------


class ContextConfig(CustomModel):
    """Context window and auto-compact configuration for an Agent.

    Controls when Claude Code automatically compresses the conversation
    context.  Under the hood, cckit translates these settings into CLI
    environment variables (``CLAUDE_CODE_AUTO_COMPACT_WINDOW``,
    ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``) that are injected **per
    subprocess**, so different Agents can have independent compact
    thresholds even when running concurrently.

    Fields
    ------
    max_context_tokens:
        Effective context window size in tokens.  **Must be set
        explicitly** — it is NOT the same as ``ModelConfig.max_tokens``
        (which controls max *output* tokens).  For example, a model
        with 8 K context but ``max_tokens=4096`` should set
        ``max_context_tokens=8192``.  When ``None`` (default), the CLI
        uses the model's native context window size.
    auto_compact_pct:
        Percentage of the context window (0–100) at which auto-compact
        fires.  Defaults to ``80`` (compact when 80 % of the window is
        consumed).  Lower values trigger compaction earlier — useful for
        small-context models.
    disable_auto_compact:
        When ``True``, disable automatic compaction entirely.  Manual
        ``/compact`` still works.

    Usage::

        from cckit import Agent, ContextConfig

        # 8 K context model — compact at 60 %
        agent = Agent(
            name="small-model",
            model=ModelConfig(model="openai/gpt-4o-mini", max_tokens=4096),
            context=ContextConfig(max_context_tokens=8192, auto_compact_pct=60),
        )

        # Only override compact percentage, keep native context window
        agent = Agent(
            name="early-compact",
            context=ContextConfig(auto_compact_pct=50),
        )
    """

    max_context_tokens: int | None = None
    auto_compact_pct: int = 80
    disable_auto_compact: bool = False

    def to_env(self) -> dict[str, str]:
        """Translate this config into CLI environment variables.

        Returns a dict suitable for merging into ``RunContext.env`` /
        ``ClaudeAgentOptions.env``.
        """
        env: dict[str, str] = {}
        if self.max_context_tokens is not None:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(self.max_context_tokens)
        if self.auto_compact_pct != 80:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(self.auto_compact_pct)
        if self.disable_auto_compact:
            env["DISABLE_AUTO_COMPACT"] = "1"
        return env


# ---------------------------------------------------------------------------
# Task budget
# ---------------------------------------------------------------------------


class TaskBudgetConfig(CustomModel):
    """API-side task budget configuration — controls the model's token budget awareness.

    When set, the model is made aware of its remaining token budget so it can
    pace tool use and wrap up gracefully before the limit is reached.  Sent as
    ``output_config.task_budget`` with the ``task-budgets-2026-03-13`` beta
    header by the SDK.

    Fields
    ------
    total:
        Total token budget for the task (input + output tokens combined).
        The model will try to finish within this budget.

    Usage::

        from cckit import Agent, TaskBudgetConfig

        agent = Agent(
            name="budget-agent",
            task_budget=TaskBudgetConfig(total=50_000),
            instruction="Analyze and summarize the codebase.",
        )
    """

    total: int


# ---------------------------------------------------------------------------
# Sandbox options
# ---------------------------------------------------------------------------


class SandboxOptions(CustomModel):
    """Sandbox isolation policy for a single Agent (macOS / Linux / WSL2 only).

    cckit translates this policy into a unified settings JSON passed through
    ``ClaudeAgentOptions.settings``. ``ClaudeAgentOptions.sandbox`` stays
    ``None`` so the SDK does not overwrite filesystem/network rules.

    Filesystem rules
    ----------------
    The concrete task workspace directory is **always** added to ``allowWrite``
    automatically at runtime, so the agent can only write inside its own
    workspace directory.

    ``deny_read`` defaults to ``["~/"]`` to prevent reading the home directory
    (e.g. ~/.ssh).  Set to ``[]`` to disable.

    Network rules
    -------------
    ``allowed_domains`` restricts which hosts Bash subprocesses can reach.
    Empty list means *block all* outbound traffic (allow-only semantics).
    ``denied_domains`` takes precedence over ``allowed_domains``.

    Bash-sandbox behaviour
    ----------------------
    ``excluded_commands`` lists commands that run *outside* the sandbox
    (e.g. ``["git", "docker"]``) so they can access the full network / filesystem.
    ``allow_unsandboxed_commands`` controls whether ``dangerouslyDisableSandbox``
    is honoured; set to ``False`` for stricter enforcement.
    """

    # --- core ---
    enabled: bool = False

    # --- sandbox.filesystem (settings JSON) ---
    # the concrete workspace_dir is appended to allow_write automatically at runtime
    # 写默认全部禁止，仅在allow_write可写
    allow_write: list[str] = Field(default_factory=list)   # extra write-allowed paths
    deny_write:  list[str] = Field(default_factory=list)   # write-blocked paths
    allow_read:  list[str] = Field(default_factory=list)   # re-allow inside deny_read zones
    # 读默认全部允许，仅在deny_read中不可读。不支持*这种通配符，但可以`//`代表所有
    deny_read:   list[str] = Field(default_factory=lambda: ["~/"])  # read-blocked paths

    # --- sandbox.network (settings JSON) ---
    allowed_domains: list[str] = Field(default_factory=list)  # empty = block all
    denied_domains: list[str] = Field(default_factory=list)   # takes precedence over allowed

    # --- SandboxSettings TypedDict (bash process behaviour) ---
    auto_allow_bash: bool = True           # autoAllowBashIfSandboxed
    excluded_commands: list[str] = Field(default_factory=list)   # e.g. ["git", "docker"]
    allow_unsandboxed_commands: bool = False   # allowUnsandboxedCommands
    enable_weaker_nested_sandbox: bool = False  # for unprivileged Docker (Linux)


# ---------------------------------------------------------------------------
# Runner configuration
# ---------------------------------------------------------------------------


class RunnerConfig(CustomModel):
    """Runner execution configuration — replaces the old global singletons.

    Users can either construct explicitly or use ``from_env()`` to read
    environment variables for runner-level infrastructure concerns.
    """

    default_model: ModelConfig = Field(default_factory=ModelConfig)
    workspace_root: Path = Path("/tmp/cckit_workspaces")
    skills_dir: Path = Path("/opt/agent-platform/skills")
    max_concurrent_agents: int = 5
    debug: bool = False
    log_level: str = "INFO"
    permission_mode: PermissionMode = "bypassPermissions"

    @classmethod
    def from_env(cls) -> RunnerConfig:
        """Create a RunnerConfig by reading environment variables.

        Reads ANTHROPIC_* and PLATFORM_* env vars. Sandbox policy belongs on
        ``Agent(..., sandbox=...)``; RunnerConfig only owns runner-level
        infrastructure such as workspace root and concurrency.
        """
        from pydantic_settings import BaseSettings

        class _EnvModel(BaseSettings):
            model_config = {"env_prefix": "ANTHROPIC_", "case_sensitive": False}
            api_key: str = ""
            base_url: str = ""
            model: str = "anthropic/claude-sonnet-4-6"
            max_tokens: int = 16384
            max_turns: int = 50
            timeout_seconds: int = 300

        class _EnvPlatform(BaseSettings):
            model_config = {"env_prefix": "PLATFORM_", "case_sensitive": False}
            debug: bool = False
            log_level: str = "INFO"
            max_concurrent_agents: int = 5
            skills_dir: Path = Path("/opt/agent-platform/skills")
            workspace_root: Path = Path("/tmp/cckit_workspaces")
            permission_mode: PermissionMode = "bypassPermissions"

        env_m = _EnvModel()
        env_p = _EnvPlatform()

        return cls(
            default_model=ModelConfig(
                model=env_m.model,
                api_key=env_m.api_key,
                base_url=env_m.base_url,
                max_tokens=env_m.max_tokens,
                max_turns=env_m.max_turns,
                timeout_seconds=env_m.timeout_seconds,
            ),
            workspace_root=env_p.workspace_root,
            skills_dir=env_p.skills_dir,
            max_concurrent_agents=env_p.max_concurrent_agents,
            debug=env_p.debug,
            log_level=env_p.log_level,
            permission_mode=env_p.permission_mode,
        )
