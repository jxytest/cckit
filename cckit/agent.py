"""Agent definition — the core abstraction of cckit.

An Agent is a declarative specification of *what* an agent does.
Execution is handled separately by ``Runner``.

Two usage modes:
  1. **Simple instantiation** (Google-ADK style) — no inheritance needed
  2. **Subclass** — for advanced lifecycle hooks or dynamic behavior
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cckit.types import AgentResult, ModelConfig, RunContext, SandboxOptions

logger = logging.getLogger(__name__)

# Type aliases for callbacks
InstructionFn = Callable[[RunContext], str]
LifecycleBeforeFn = Callable[[RunContext], Any]  # async def (ctx) -> None
LifecycleAfterFn = Callable[[RunContext, AgentResult], Any]  # async def (ctx, result) -> None
LifecycleErrorFn = Callable[[RunContext, Exception], Any]  # async def (ctx, error) -> None


class Agent:
    """A declarative agent specification.

    Parameters
    ----------
    name:
        Unique identifier for this agent (used in logs, sub-agent references).
    model:
        Model configuration. Accepts:
        - A string like ``"anthropic/claude-sonnet-4-6"`` or ``"openai/gpt-4o-mini"``
        - A ``ModelConfig`` instance for full control over LiteLLM-style model, auth, and base URL
        - ``None`` to inherit from Runner defaults
    description:
        Human-readable description (used when this agent is a sub-agent).
    instruction:
        System prompt. Accepts:
        - A string (static prompt)
        - A callable ``(ctx: RunContext) -> str`` (dynamic prompt)
    tools:
        List of tool names the agent can use (e.g. ``["Bash", "Read", "Write"]``).
    sub_agents:
        List of ``Agent`` instances that this agent can delegate to.
    skills:
        List of skill names to provision. Resolved from ``skills_dir`` or
        RunnerConfig.skills_dir.
    skills_dir:
        Base directory containing skill sub-directories. Overrides
        RunnerConfig.skills_dir for this agent.
    mcp_servers:
        MCP servers to expose to this agent.  Pass a dict of server name →
        :class:`~claude_agent_sdk.types.McpServerConfig` (stdio, SSE, HTTP or
        SDK in-process).  Passed directly to the SDK — no Runner-level
        registry needed.
    required_params:
        List of keys that must be present in ``RunContext.params``.
    max_turns:
        Max conversation turns. 0 = use model config default.
    sandbox:
        Optional sandbox policy for this agent. Workspace root remains a
        Runner-level infrastructure concern; per-agent sandbox rules such as
        enabled/read-write/network policy belong here.

    Lifecycle callbacks (optional):
    on_before:
        ``async def (ctx: RunContext) -> None`` — called before execution.
    on_after:
        ``async def (ctx: RunContext, result: AgentResult) -> None`` — called after.
    on_error:
        ``async def (ctx: RunContext, error: Exception) -> None`` — called on failure.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str | ModelConfig | None = None,
        description: str = "",
        instruction: str | InstructionFn = "",
        tools: list[str] | None = None,
        sub_agents: list[Agent] | None = None,
        skills: list[str] | None = None,
        skills_dir: str | Path | None = None,
        mcp_servers: dict[str, Any] | None = None,
        required_params: list[str] | None = None,
        max_turns: int = 0,
        sandbox: SandboxOptions | None = None,
        # Lifecycle callbacks
        on_before: LifecycleBeforeFn | None = None,
        on_after: LifecycleAfterFn | None = None,
        on_error: LifecycleErrorFn | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tools = tools or []
        self.sub_agents = sub_agents or []
        self.skills = skills or []
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.mcp_servers: dict[str, Any] = mcp_servers or {}
        self.required_params = required_params or []
        self.max_turns = max_turns
        self._sandbox = sandbox

        # Normalize model → ModelConfig | None
        if model is None:
            self._model_config: ModelConfig | None = None
        elif isinstance(model, str):
            self._model_config = ModelConfig(model=model)
        else:
            self._model_config = model

        # Store instruction (string or callable)
        self._instruction = instruction

        # Lifecycle hooks
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error

    # ------------------------------------------------------------------
    # Resolve instruction at runtime
    # ------------------------------------------------------------------

    def resolve_instruction(self, ctx: RunContext) -> str:
        """Return the system prompt, evaluating callable if needed."""
        if callable(self._instruction):
            return self._instruction(ctx)
        return self._instruction

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> ModelConfig | None:
        """Return the model config, or None if inheriting from Runner."""
        return self._model_config

    @property
    def sandbox_config(self) -> SandboxOptions | None:
        """Return the agent sandbox policy, or ``None`` when sandbox is not configured."""
        return self._sandbox

    # ------------------------------------------------------------------
    # Lifecycle hook invocation (used by Runner, overridable by subclass)
    # ------------------------------------------------------------------

    async def before_execute(self, ctx: RunContext) -> None:
        """Called before agent execution. Override or pass ``on_before`` callback."""
        if self._on_before:
            await self._on_before(ctx)

    async def after_execute(self, ctx: RunContext, result: AgentResult) -> None:
        """Called after agent execution. Override or pass ``on_after`` callback."""
        if self._on_after:
            await self._on_after(ctx, result)

    async def error_execute(self, ctx: RunContext, error: Exception) -> None:
        """Called on execution failure. Override or pass ``on_error`` callback."""
        if self._on_error:
            await self._on_error(ctx, error)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r}>"
