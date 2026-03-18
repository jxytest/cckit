"""Runner — executes agents and yields streaming events.

This is the orchestration layer that replaces ``AgentExecutor``.
It is the only class that ties together workspace management,
git cloning, skill provisioning, middleware, and the SDK bridge.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from cckit._cli import check_claude_cli
from cckit._engine.collector import StreamCollector
from cckit._engine.sdk_bridge import run_sdk_query
from cckit.agent import Agent
from cckit.git import operations as git_ops
from cckit.middleware.base import Middleware
from cckit.sandbox.config import SandboxConfigBuilder
from cckit.sandbox.workspace import WorkspaceManager
from cckit.skill.provisioner import SkillProvisioner
from cckit.types import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    ModelConfig,
    RunContext,
    RunnerConfig,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class Runner:
    """Execute an Agent and yield AgentEvent objects.

    This is the primary execution engine of cckit.  It separates
    agent *definition* (what) from *execution* (how).

    Parameters
    ----------
    config:
        Explicit execution configuration.  If ``None``, reads from env
        vars via ``RunnerConfig.from_env()``.
    middlewares:
        Optional list of ``Middleware`` instances.  Executed in order,
        the first middleware is the outermost wrapper.
    mcp_servers:
        Optional dict of MCP server name → server factory callable.
        Agents reference these by name via ``Agent.mcp_tools``.
    workspace_manager:
        Override the default WorkspaceManager.
    skill_provisioner:
        Override the default SkillProvisioner.
    """

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        middlewares: list[Middleware] | None = None,
        mcp_servers: dict[str, Any] | None = None,
        workspace_manager: WorkspaceManager | None = None,
        skill_provisioner: SkillProvisioner | None = None,
    ) -> None:
        # Validate Claude CLI on first Runner instantiation
        check_claude_cli()

        self._config = config or RunnerConfig.from_env()
        self._middlewares: list[Middleware] = middlewares or []
        self._mcp_servers = mcp_servers or {}

        self._workspace = workspace_manager or WorkspaceManager(
            root=self._config.sandbox.workspace_root
        )
        self._skill_provisioner = skill_provisioner or SkillProvisioner(
            skills_dir=self._config.skills_dir
        )
        self._sandbox_builder = SandboxConfigBuilder(
            enabled=self._config.sandbox.enabled,
            network_hosts=list(self._config.sandbox.network_allowed_hosts),
        )
        self._clone_semaphore = asyncio.Semaphore(
            self._config.max_concurrent_agents
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        agent: Agent,
        ctx: RunContext,
    ) -> AgentResult:
        """Run agent to completion and return the final result.

        Convenience wrapper around ``run_stream`` that consumes all events
        and reconstructs an ``AgentResult`` from the RESULT/ERROR event.
        """
        last_result_event: AgentEvent | None = None
        last_error_event: AgentEvent | None = None

        async for event in self.run_stream(agent, ctx):
            if event.event_type == AgentEventType.RESULT:
                last_result_event = event
            elif event.event_type == AgentEventType.ERROR:
                last_error_event = event

        if last_result_event is not None:
            return AgentResult(
                task_id=ctx.task_id,
                agent_type=agent.name,
                status=TaskStatus.COMPLETED,
                result_text=last_result_event.text,
                cost_usd=last_result_event.data.get("cost_usd", 0.0),
                session_id=last_result_event.data.get("session_id", ""),
            )

        error_msg = last_error_event.text if last_error_event else "Unknown error"
        return AgentResult(
            task_id=ctx.task_id,
            agent_type=agent.name,
            status=TaskStatus.FAILED,
            is_error=True,
            error_message=error_msg,
        )

    async def run_stream(
        self,
        agent: Agent,
        ctx: RunContext,
    ) -> AsyncIterator[AgentEvent]:
        """Run *agent* and yield ``AgentEvent`` objects as they arrive.

        This is the primary execution method — an async generator.

        Orchestration flow:
            1. Validate context (required_params, structural checks)
            2. agent.before_execute(ctx)
            3. Create/resume workspace
            4. Git clone (if needed, skipped on resume)
            5. Provision skills (if needed, skipped on resume)
            6. Resolve instruction (string or callable)
            7. Build SDK options (model, tools, sub-agents, MCP, sandbox)
            8. [Middleware chain] → SDK bridge → StreamCollector → yield AgentEvent
            9. agent.after_execute(ctx, result) or agent.error_execute(ctx, error)
            10. Cleanup/suspend workspace
        """
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
                    agent_type=agent.name,
                    status=TaskStatus.FAILED,
                    is_error=True,
                    error_message=msg,
                    duration_seconds=round(time.monotonic() - start, 2),
                )
                return

            # --- lifecycle: before ---
            await agent.before_execute(ctx)

            # --- workspace ---
            if ctx.workspace.enabled:
                if ctx.resume_session_id and ctx.workspace_dir:
                    # Resume: reuse existing workspace
                    workspace_dir = await self._workspace.resume(ctx.workspace_dir)
                else:
                    # First execution: create a fresh workspace
                    workspace_dir = await self._workspace.create(ctx.task_id)
                    ctx.workspace_dir = workspace_dir

                    if ctx.workspace.git_clone and ctx.git_repo_url:
                        async with self._clone_semaphore:
                            await git_ops.clone(
                                ctx.git_repo_url,
                                workspace_dir,
                                branch=ctx.git_branch,
                                extra_env=ctx.env or None,
                            )

                    # --- provision skills ---
                    if agent.skills:
                        # Use agent-level skills_dir if specified, otherwise runner default
                        provisioner = self._skill_provisioner
                        if agent.skills_dir:
                            provisioner = SkillProvisioner(skills_dir=agent.skills_dir)
                        await provisioner.provision(agent.skills, workspace_dir)

            # --- instruction ---
            instruction = agent.resolve_instruction(ctx)

            # --- prompt ---
            prompt = ctx.prompt or ""

            # --- build SDK options ---
            model = self._resolve_model(agent)
            options = self._build_options(agent, ctx, model, workspace_dir, instruction)

            # --- stream from SDK (through middleware chain) ---
            collector = StreamCollector(ctx.task_id)
            query_fn = self._build_middleware_chain(collector, ctx)

            async for event in query_fn(prompt, options, collector):
                yield event

            # --- build result ---
            duration = time.monotonic() - start
            result = AgentResult(
                task_id=ctx.task_id,
                agent_type=agent.name,
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
                agent_type=agent.name,
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
            logger.exception("Agent %s failed: %s", agent.name, exc)

            # --- lifecycle: on_error ---
            try:
                await agent.error_execute(ctx, exc)
            except Exception:
                logger.exception("error_execute hook failed for %s", agent.name)

        finally:
            # --- lifecycle: after ---
            if result is not None:
                try:
                    await agent.after_execute(ctx, result)
                except Exception:
                    logger.exception(
                        "after_execute hook failed for %s", agent.name
                    )

                # --- log final summary ---
                logger.info(
                    "Agent %s completed: task_id=%s status=%s cost=$%.4f duration=%.2fs",
                    agent.name,
                    result.task_id,
                    result.status,
                    result.cost_usd,
                    result.duration_seconds,
                )

            # --- cleanup or suspend workspace ---
            if workspace_dir:
                should_cleanup = (
                    result is not None
                    and result.status == TaskStatus.COMPLETED
                    and not ctx.resume_session_id
                )
                if should_cleanup:
                    await self._workspace.cleanup(workspace_dir)
                else:
                    await self._workspace.suspend(workspace_dir)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model(self, agent: Agent) -> ModelConfig:
        """Merge agent model_config with runner defaults."""
        agent_model = agent.model_config
        base = self._config.default_model

        if agent_model is None:
            return base

        return ModelConfig(
            model=agent_model.model or base.model,
            api_key=agent_model.api_key or base.api_key,
            base_url=agent_model.base_url or base.base_url,
            max_tokens=agent_model.max_tokens,
            max_turns=agent_model.max_turns if agent_model.max_turns > 0 else base.max_turns,
            permission_mode=agent_model.permission_mode or base.permission_mode,
            timeout_seconds=agent_model.timeout_seconds or base.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Context validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_context(agent: Agent, ctx: RunContext) -> list[str]:
        """Check that all required parameters are present.

        Returns a list of missing parameter names (empty = OK).
        """
        missing: list[str] = []

        # Check agent-declared required params
        for param in agent.required_params:
            if param not in ctx.params or not ctx.params[param]:
                missing.append(param)

        # Check structural requirements
        if ctx.workspace.git_clone and not ctx.git_repo_url and not ctx.resume_session_id:
            missing.append("git_repo_url")

        # Skills require a workspace
        if agent.skills and not ctx.workspace.enabled:
            missing.append("workspace.enabled (required when skills are declared)")

        return missing

    # ------------------------------------------------------------------
    # Middleware chain builder
    # ------------------------------------------------------------------

    def _build_middleware_chain(
        self,
        collector: StreamCollector,
        ctx: RunContext,
    ) -> Any:
        """Wrap ``run_sdk_query`` with the middleware stack.

        Returns a callable with signature (prompt, options, collector).
        """

        # The innermost function — actual SDK call
        async def inner(
            prompt: str, options: Any, collector_: Any
        ) -> AsyncIterator[AgentEvent]:
            async for event in run_sdk_query(prompt, options, collector_):
                yield event

        current = inner

        # Wrap from inside out (last middleware wraps first)
        for mw in reversed(self._middlewares):

            def make_wrapper(middleware: Middleware, next_fn: Any) -> Any:
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
    # SDK options builder
    # ------------------------------------------------------------------

    def _build_options(
        self,
        agent: Agent,
        ctx: RunContext,
        model: ModelConfig,
        workspace_dir: Path | None,
        instruction: str,
    ) -> Any:
        """Construct ``ClaudeAgentOptions`` from Agent + RunContext + resolved model."""
        from claude_agent_sdk import (  # noqa: WPS433
            AgentDefinition,
            ClaudeAgentOptions,
        )

        # -- allowed tools --
        allowed_tools = list(agent.tools)
        if agent.sub_agents and "Agent" not in allowed_tools:
            allowed_tools.append("Agent")
        if agent.skills and "Skill" not in allowed_tools:
            allowed_tools.append("Skill")

        # -- sub-agents → SDK AgentDefinition --
        agents: dict[str, AgentDefinition] = {}
        for sub in agent.sub_agents:
            agents[sub.name] = AgentDefinition(
                description=sub.description,
                prompt=sub.resolve_instruction(ctx),
                tools=list(sub.tools),
            )

        # -- MCP servers --
        mcp_servers: dict[str, Any] = {}
        if agent.mcp_tools:
            for name, factory in self._mcp_servers.items():
                mcp_servers[name] = factory()

        # -- sandbox --
        sandbox_dict = self._sandbox_builder.build(workspace_dir)

        # -- environment --
        env: dict[str, str] = {}
        if model.api_key:
            env["ANTHROPIC_API_KEY"] = model.api_key
        if model.base_url:
            env["ANTHROPIC_BASE_URL"] = model.base_url
        env.update(ctx.env)

        # -- configurable SDK params --
        max_turns = agent.max_turns if agent.max_turns > 0 else model.max_turns

        # -- assemble --
        opts = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            system_prompt=instruction or None,
            max_turns=max_turns,
            model=model.model,
            permission_mode=model.permission_mode,
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
        if agent.skills:
            opts.setting_sources = ["project"]

        # -- resume: restore a previous session's conversation context --
        if ctx.resume_session_id:
            opts.resume = ctx.resume_session_id
            opts.fork_session = ctx.fork_session

        return opts
