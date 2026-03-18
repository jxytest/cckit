"""Agent executor — the single module that interfaces with claude-agent-sdk.

Orchestration flow::

    on_before_execute
    → validate context
    → create workspace (optional)
    → git clone (optional)
    → provision skills (optional)
    → build_prompt
    → build ClaudeAgentOptions (sub-agents, MCP, sandbox, skills)
    → [Middleware chain] → SDK query()  →  StreamCollector  →  yield AgentEvent
    → on_after_execute  (or on_error if failed)
    → cleanup workspace
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from core.agent.base import BaseAgent
from core.agent.git import operations as git_ops
from core.agent.mcp.platform_tools import get_platform_mcp_server
from core.agent.middleware import AgentMiddleware
from core.agent.sandbox.config import SandboxConfig
from core.agent.sandbox.workspace import WorkspaceManager
from core.agent.schemas import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    ExecutionContext,
    TaskStatus,
)
from core.agent.skill import SkillProvisioner
from core.agent.streaming.collector import StreamCollector
from core.config import anthropic_settings, platform_settings
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute an agent instance using claude-agent-sdk.

    This is the **only** module in core/ that imports the SDK.

    Parameters
    ----------
    workspace_manager:
        Override the default workspace manager.
    sandbox_config:
        Override the default sandbox config builder.
    skill_provisioner:
        Override the default skill provisioner.
    middlewares:
        Optional list of ``AgentMiddleware`` instances.  Executed in order,
        the first middleware is the outermost wrapper.
    max_concurrent_clones:
        Limit on parallel ``git clone`` operations.  Defaults to
        ``PLATFORM_MAX_CONCURRENT_AGENTS``.  This prevents saturating
        network/disk when many tasks start simultaneously.
    """

    def __init__(
        self,
        *,
        workspace_manager: WorkspaceManager | None = None,
        sandbox_config: SandboxConfig | None = None,
        skill_provisioner: SkillProvisioner | None = None,
        middlewares: list[AgentMiddleware] | None = None,
        max_concurrent_clones: int | None = None,
    ) -> None:
        self._workspace = workspace_manager or WorkspaceManager()
        self._sandbox = sandbox_config or SandboxConfig()
        self._skill_provisioner = skill_provisioner or SkillProvisioner()
        self._middlewares: list[AgentMiddleware] = middlewares or []
        clone_limit = max_concurrent_clones or platform_settings.max_concurrent_agents
        self._clone_semaphore = asyncio.Semaphore(clone_limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_stream(
        self,
        agent: BaseAgent,
        ctx: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        """Run *agent* and yield ``AgentEvent`` objects as they arrive.

        This is an async generator — callers iterate with ``async for``.
        """
        cfg = agent.config()
        start = time.monotonic()
        workspace_dir: Path | None = None
        result: AgentResult | None = None

        try:
            # --- validate context ---
            missing = self._validate_context(agent, ctx)
            if missing:
                msg = f"Missing required parameters: {', '.join(missing)}"
                yield AgentEvent(
                    event_type=AgentEventType.ERROR,
                    task_id=ctx.task_id,
                    text=msg,
                )
                result = AgentResult(
                    task_id=ctx.task_id,
                    agent_type=cfg.agent_type,
                    status=TaskStatus.FAILED,
                    is_error=True,
                    error_message=msg,
                    duration_seconds=round(time.monotonic() - start, 2),
                )
                return

            # --- lifecycle: before ---
            await agent.on_before_execute(ctx)

            # --- workspace ---
            if cfg.needs_workspace:
                workspace_dir = await self._workspace.create(ctx.task_id)
                ctx.workspace_dir = workspace_dir

                if cfg.needs_git_clone and ctx.git_repo_url:
                    # Semaphore prevents saturating network/disk when
                    # many tasks clone concurrently.  ctx.env is passed
                    # so each task can authenticate with its own token
                    # (e.g. via GIT_ASKPASS or token-embedded URL).
                    async with self._clone_semaphore:
                        await git_ops.clone(
                            ctx.git_repo_url,
                            workspace_dir,
                            branch=ctx.git_branch,
                            extra_env=ctx.env or None,
                        )

                # --- provision skills ---
                if cfg.skills:
                    await self._skill_provisioner.provision(
                        cfg.skills, workspace_dir
                    )

            # --- prompt ---
            prompt = agent.build_prompt(ctx)
            if ctx.prompt:
                prompt = ctx.prompt  # allow override from context

            # --- build SDK options ---
            options = self._build_options(agent, ctx, workspace_dir)

            # --- stream from SDK (through middleware chain) ---
            collector = StreamCollector(ctx.task_id)

            query_fn = self._build_middleware_chain(collector, ctx)

            async for event in query_fn(prompt, options, collector):
                yield event

            # --- build result ---
            duration = time.monotonic() - start
            result = AgentResult(
                task_id=ctx.task_id,
                agent_type=cfg.agent_type,
                status=TaskStatus.COMPLETED,
                result_text=collector.full_text,
                cost_usd=collector.cost_usd,
                duration_seconds=round(duration, 2),
                session_id=collector.session_id,
            )

        except Exception as exc:
            duration = time.monotonic() - start
            result = AgentResult(
                task_id=ctx.task_id,
                agent_type=cfg.agent_type,
                status=TaskStatus.FAILED,
                is_error=True,
                error_message=str(exc),
                duration_seconds=round(duration, 2),
            )
            yield AgentEvent(
                event_type=AgentEventType.ERROR,
                task_id=ctx.task_id,
                text=str(exc),
            )
            logger.exception("Agent %s failed: %s", cfg.agent_type, exc)

            # --- lifecycle: on_error ---
            try:
                await agent.on_error(ctx, exc)
            except Exception:
                logger.exception("on_error hook failed for %s", cfg.agent_type)

        finally:
            # --- lifecycle: after ---
            if result is not None:
                try:
                    await agent.on_after_execute(ctx, result)
                except Exception:
                    logger.exception(
                        "on_after_execute hook failed for %s", cfg.agent_type
                    )

                # --- log final summary ---
                logger.info(
                    "Agent %s completed: task_id=%s status=%s cost=$%.4f duration=%.2fs",
                    cfg.agent_type,
                    result.task_id,
                    result.status,
                    result.cost_usd,
                    result.duration_seconds,
                )

            # --- cleanup ---
            if workspace_dir:
                await self._workspace.cleanup(workspace_dir)

    # ------------------------------------------------------------------
    # Context validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_context(agent: BaseAgent, ctx: ExecutionContext) -> list[str]:
        """Check that all required parameters are present.

        Returns a list of missing parameter names (empty = OK).
        """
        cfg = agent.config()
        missing: list[str] = []

        # Check agent-declared required params
        for param in agent.required_params():
            if param not in ctx.extra_params or not ctx.extra_params[param]:
                missing.append(param)

        # Check structural requirements
        if cfg.needs_git_clone and not ctx.git_repo_url:
            missing.append("git_repo_url")

        # Skills require a workspace for SDK discovery
        if cfg.skills and not cfg.needs_workspace:
            missing.append("needs_workspace (required when skills are declared)")

        return missing

    # ------------------------------------------------------------------
    # Middleware chain builder
    # ------------------------------------------------------------------

    def _build_middleware_chain(
        self,
        collector: StreamCollector,
        ctx: ExecutionContext,
    ) -> Any:
        """Wrap ``_run_sdk_query`` with the middleware stack.

        Returns a callable with signature (prompt, options, collector).
        """

        # The innermost function — actual SDK call
        async def inner(
            prompt: str, options: Any, collector_: Any
        ) -> AsyncIterator[AgentEvent]:
            async for event in self._run_sdk_query(prompt, options, collector_):
                yield event

        current = inner

        # Wrap from inside out (last middleware wraps first)
        for mw in reversed(self._middlewares):

            # Capture mw and current in closure
            def make_wrapper(middleware: AgentMiddleware, next_fn: Any) -> Any:
                async def wrapper(
                    prompt: str, options: Any, collector_: Any
                ) -> AsyncIterator[AgentEvent]:
                    async for event in middleware.wrap(
                        next_fn, prompt, options, collector_, ctx
                    ):
                        yield event

                return wrapper

            current = make_wrapper(mw, current)

        return current

    # ------------------------------------------------------------------
    # SDK interaction (private)
    # ------------------------------------------------------------------

    async def _run_sdk_query(
        self,
        prompt: str,
        options: Any,
        collector: StreamCollector,
    ) -> AsyncIterator[AgentEvent]:
        """Call ``claude_agent_sdk.query()`` and yield events via collector."""
        try:
            from claude_agent_sdk import query  # noqa: WPS433
        except ImportError as exc:
            raise AgentExecutionError(
                "claude-agent-sdk is not installed",
                detail=str(exc),
            ) from exc

        try:
            async for message in query(prompt=prompt, options=options):
                events = collector.process_message(message)
                for event in events:
                    yield event
        except Exception as exc:
            raise AgentExecutionError(
                f"SDK query failed: {exc}",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Options builder (private)
    # ------------------------------------------------------------------

    def _build_options(
        self,
        agent: BaseAgent,
        ctx: ExecutionContext,
        workspace_dir: Path | None,
    ) -> Any:
        """Construct ``ClaudeAgentOptions`` from agent config + context."""
        from claude_agent_sdk import (  # noqa: WPS433
            AgentDefinition,
            ClaudeAgentOptions,
        )

        cfg = agent.config()

        # -- allowed tools --
        allowed_tools = list(cfg.allowed_tools)
        if cfg.sub_agents and "Agent" not in allowed_tools:
            allowed_tools.append("Agent")
        if cfg.skills and "Skill" not in allowed_tools:
            allowed_tools.append("Skill")

        # -- sub-agents → SDK AgentDefinition --
        agents: dict[str, AgentDefinition] = {}
        for sub in cfg.sub_agents:
            agents[sub.name] = AgentDefinition(
                description=sub.description,
                prompt=sub.system_prompt or "",
                tools=list(sub.tools),
            )

        # -- MCP servers --
        mcp_servers: dict[str, Any] = {}
        if cfg.mcp_tool_names:
            mcp_servers["platform"] = get_platform_mcp_server()

        # -- sandbox --
        sandbox_dict = self._sandbox.build(workspace_dir)

        # -- environment --
        env: dict[str, str] = {}
        if anthropic_settings.api_key:
            env["ANTHROPIC_API_KEY"] = anthropic_settings.api_key
        if anthropic_settings.base_url:
            env["ANTHROPIC_BASE_URL"] = anthropic_settings.base_url
        env.update(ctx.env)

        # -- model --
        model = anthropic_settings.model

        # -- configurable SDK params --
        max_turns = cfg.max_turns or anthropic_settings.max_turns
        permission_mode = anthropic_settings.permission_mode

        # -- assemble --
        opts = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            system_prompt=cfg.system_prompt or None,
            max_turns=max_turns,
            model=model,
            permission_mode=permission_mode,
            env=env,
        )

        # Set optional fields only when non-empty (SDK may reject empty dicts)
        if agents:
            opts.agents = agents
        if mcp_servers:
            opts.mcp_servers = mcp_servers
        if sandbox_dict:
            opts.sandbox = sandbox_dict
        if workspace_dir:
            opts.cwd = str(workspace_dir)

        # -- skills: enable SDK filesystem-based skill discovery --
        if cfg.skills:
            opts.setting_sources = ["project"]

        return opts
