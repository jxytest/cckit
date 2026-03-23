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
from typing import Any, Literal, cast

from claude_agent_sdk.types import SystemPromptPreset

from cckit._cli import check_api_connectivity, check_claude_cli
from cckit._engine.collector import StreamCollector
from cckit._engine.sdk_bridge import run_sdk_query
from cckit.agent import Agent
from cckit.exceptions import HookError
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
    StreamResult,
    TaskStatus,
    _ResultHolder,
)

logger = logging.getLogger(__name__)


def _is_windows() -> bool:
    import sys
    return sys.platform == "win32"


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
    workspace_manager:
        Override the default WorkspaceManager.
    skill_provisioner:
        Override the default SkillProvisioner.
    preflight_check:
        If ``True``, verify API key validity and network connectivity
        before each execution.  Fails fast with a clear error instead
        of waiting for a CLI initialization timeout.
    """

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        middlewares: list[Middleware] | None = None,
        workspace_manager: WorkspaceManager | None = None,
        skill_provisioner: SkillProvisioner | None = None,
        preflight_check: bool = False,
    ) -> None:
        # Validate Claude CLI on first Runner instantiation
        check_claude_cli()

        self._config = config or RunnerConfig.from_env()

        # Apply log_level to the cckit logger hierarchy only.
        # We intentionally do NOT touch the root logger — that is the
        # caller's responsibility.  Setting the level on "cckit" is enough
        # to control all cckit.* child loggers uniformly.
        _level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logging.getLogger("cckit").setLevel(_level)
        self._middlewares: list[Middleware] = middlewares or []
        self._preflight_check = preflight_check

        self._workspace = workspace_manager or WorkspaceManager(
            root=self._config.sandbox.workspace_root
        )
        self._skill_provisioner = skill_provisioner or SkillProvisioner(
            skills_dir=self._config.skills_dir
        )
        self._sandbox_builder = SandboxConfigBuilder(
            enabled=self._config.sandbox.enabled,
            workspace_root=self._config.sandbox.workspace_root,
            allow_write=list(self._config.sandbox.allow_write),
            deny_write=list(self._config.sandbox.deny_write),
            allow_read=list(self._config.sandbox.allow_read),
            deny_read=list(self._config.sandbox.deny_read),
            allowed_domains=list(self._config.sandbox.allowed_domains),
            auto_allow_bash=self._config.sandbox.auto_allow_bash,
            excluded_commands=list(self._config.sandbox.excluded_commands),
            allow_unsandboxed_commands=self._config.sandbox.allow_unsandboxed_commands,
            enable_weaker_nested_sandbox=self._config.sandbox.enable_weaker_nested_sandbox,
        )
        if self._config.sandbox.enabled and _is_windows():
            logger.warning(
                "Sandbox is enabled but the current platform is Windows. "
                "OS-level filesystem/network isolation (Seatbelt/bubblewrap) is "
                "not supported on native Windows — sandbox settings will have no effect. "
                "Use macOS, Linux, or WSL2 for full sandbox enforcement."
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

        Convenience wrapper around :meth:`run_stream` that consumes all
        events and returns the :class:`AgentResult`.  The returned object
        is the **same instance** that lifecycle callbacks (``on_after``)
        receive, so any mutations (e.g. ``result.extra["mr_url"] = ...``)
        are preserved.
        """
        stream = self.run_stream(agent, ctx)
        async for _event in stream:
            pass

        if stream.result is not None:
            return stream.result

        # Defensive fallback — should not happen in normal execution.
        return AgentResult(
            task_id=ctx.task_id,
            agent_type=agent.name,
            status=TaskStatus.FAILED,
            is_error=True,
            error_message="Stream completed without producing a result",
        )

    def run_stream(
        self,
        agent: Agent,
        ctx: RunContext,
    ) -> StreamResult:
        """Run *agent* and return a :class:`StreamResult` for streaming events.

        ``StreamResult`` is async-iterable, so existing code continues to
        work unchanged::

            async for event in runner.run_stream(agent, ctx):
                print(event.text)

        New usage — access the final result after the stream ends::

            stream = runner.run_stream(agent, ctx)
            async for event in stream:
                print(event.text)
            result = stream.result  # same object on_after received
        """
        holder = _ResultHolder()
        aiter = self._execute(agent, ctx, holder)
        return StreamResult(aiter, holder)

    # ------------------------------------------------------------------
    # Core execution (private async generator)
    # ------------------------------------------------------------------

    async def _execute(
        self,
        agent: Agent,
        ctx: RunContext,
        holder: _ResultHolder,
    ) -> AsyncIterator[AgentEvent]:
        """Internal async generator — the full orchestration flow.

        Orchestration flow:
            1. Validate context (required_params, structural checks)
            2. agent.before_execute(ctx)
            3. Preflight API connectivity check (if enabled)
            4. Create/resume workspace
            5. Git clone (if needed, skipped on resume)
            6. Provision skills (if needed, skipped on resume)
            7. Resolve instruction (string or callable)
            8. Build SDK options (model, tools, sub-agents, MCP, sandbox)
            9. [Middleware chain] → SDK bridge → StreamCollector → yield AgentEvent
            10. agent.after_execute(ctx, result) or agent.error_execute(ctx, error)
            11. Cleanup/suspend workspace
        """
        start = time.monotonic()
        git_cfg = ctx.resolved_git()

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
                holder.result = AgentResult(
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

            # --- preflight: API connectivity check ---
            if self._preflight_check:
                model = self._resolve_model(agent)
                api_key = model.api_key or ctx.env.get("ANTHROPIC_API_KEY", "")
                base_url = model.base_url or ctx.env.get("ANTHROPIC_BASE_URL", "")
                check_api_connectivity(
                    api_key=api_key, base_url=base_url, model=model.model,
                )

            # --- workspace ---
            if ctx.workspace.enabled:
                if ctx.resume_session_id and ctx.workspace_dir:
                    # Resume: reuse existing workspace
                    holder.workspace_dir = await self._workspace.resume(ctx.workspace_dir)
                else:
                    # First execution: create a fresh workspace
                    holder.workspace_dir = await self._workspace.create(ctx.task_id)
                    ctx.workspace_dir = holder.workspace_dir

                    if git_cfg.clone and git_cfg.repo_url:
                        git_env = git_cfg.build_git_env() or None
                        async with self._clone_semaphore:
                            await git_ops.clone(
                                git_cfg.repo_url,
                                holder.workspace_dir,
                                branch=git_cfg.branch,
                                depth=git_cfg.depth,
                                extra_env=git_env,
                            )

                    # --- provision skills ---
                    if agent.skills:
                        # Use agent-level skills_dir if specified, otherwise runner default
                        provisioner = self._skill_provisioner
                        if agent.skills_dir:
                            provisioner = SkillProvisioner(skills_dir=agent.skills_dir)
                        await provisioner.provision(agent.skills, holder.workspace_dir)

            # --- instruction ---
            instruction = agent.resolve_instruction(ctx)

            # --- prompt ---
            prompt = ctx.prompt or ""

            # --- build SDK options ---
            model = self._resolve_model(agent)
            options = self._build_options(
                agent, ctx, model, holder.workspace_dir, instruction,
            )

            # --- stream from SDK (through middleware chain) ---
            collector = StreamCollector(ctx.task_id)
            query_fn = self._build_middleware_chain(collector, ctx)

            async for event in query_fn(prompt, options, collector):
                yield event

            # --- build result ---
            duration = time.monotonic() - start
            holder.result = AgentResult(
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
            holder.result = AgentResult(
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
            except Exception as hook_exc:
                logger.exception("error_execute hook failed for %s", agent.name)
                raise HookError("error_execute", hook_exc) from hook_exc

        finally:
            # --- lifecycle: after ---
            if holder.result is not None:
                try:
                    await agent.after_execute(ctx, holder.result)
                except Exception as hook_exc:
                    logger.exception(
                        "after_execute hook failed for %s", agent.name
                    )
                    raise HookError("after_execute", hook_exc) from hook_exc

                # --- log final summary ---
                logger.info(
                    "Agent %s completed: task_id=%s session_id=%s status=%s cost=$%.4f duration=%.2fs",
                    agent.name,
                    holder.result.task_id,
                    holder.result.session_id,
                    holder.result.status,
                    holder.result.cost_usd,
                    holder.result.duration_seconds,
                )

            # --- cleanup temporary askpass script ---
            try:
                git_cfg.cleanup_askpass()
            except Exception:
                logger.exception("Failed to cleanup git askpass for task %s", ctx.task_id)

            # --- cleanup or suspend workspace ---
            # Cleanup policy:
            #   - workspace.keep=True  → always suspend (caller wants to resume later)
            #   - task failed          → suspend (preserve for debugging / resume)
            #   - task succeeded       → cleanup (delete) by default
            if holder.workspace_dir:
                should_suspend = (
                    ctx.workspace.keep
                    or holder.result is None
                    or holder.result.status != TaskStatus.COMPLETED
                )
                if should_suspend:
                    await self._workspace.suspend(holder.workspace_dir)
                else:
                    await self._workspace.cleanup(holder.workspace_dir)

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
        git_cfg = ctx.resolved_git()
        if git_cfg.clone and not git_cfg.repo_url and not ctx.resume_session_id:
            missing.append("git.repo_url")

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
            SandboxSettings,
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
        mcp_servers = agent.mcp_servers

        # -- sandbox --
        sandbox_dict, settings_json = self._sandbox_builder.build(workspace_dir)
        sandbox: SandboxSettings | None = (
            SandboxSettings(**sandbox_dict) if sandbox_dict else None
        )

        # -- environment (Agent subprocess only — NO git credentials) --
        env: dict[str, str] = {}
        if model.api_key:
            # ANTHROPIC_API_KEY  → sent as X-Api-Key header (direct Anthropic API)
            # ANTHROPIC_AUTH_TOKEN → sent as Authorization: Bearer header (LLM gateway / proxy)
            # Both are injected so the CLI authenticates correctly regardless of
            # whether the endpoint is a first-party Anthropic host or a third-party proxy.
            env["ANTHROPIC_API_KEY"] = model.api_key
            env["ANTHROPIC_AUTH_TOKEN"] = model.api_key
        if model.base_url:
            env["ANTHROPIC_BASE_URL"] = model.base_url
            # Third-party proxies often reject Anthropic-specific beta headers
            # and non-essential traffic (telemetry, autoupdater, etc.)
            env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
            env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            # Disable extended thinking — many proxies don't support it
            env.setdefault("MAX_THINKING_TOKENS", "0")
        # ctx.env is for the Agent subprocess (API keys, feature flags, etc.)
        # Git credentials live in ctx.git / GitConfig and are NEVER injected here.
        env.update(ctx.env)

        # -- configurable SDK params --
        max_turns = agent.max_turns if agent.max_turns > 0 else model.max_turns

        # -- assemble --
        def _stderr_cb(line: str) -> None:
            logger.debug("[CLI stderr] %s", line.rstrip())

        opts = ClaudeAgentOptions(
            system_prompt=SystemPromptPreset(
                type='preset',
                preset='claude_code',
                append=instruction or None,
            ),
            max_turns=max_turns,
            model=model.model,
            # When sandbox is enabled, switch to dontAsk so that permissions.deny
            # rules are enforced. bypassPermissions skips all permission checks
            # (including deny rules) and is only safe inside pre-isolated envs.
            permission_mode="dontAsk" if self._config.sandbox.enabled else self._config.permission_mode,
            env=env,
            stderr=_stderr_cb,
            sandbox=sandbox,
            settings=settings_json,
            extra_args={"debug-to-stderr": None},
            user=ctx.user,
        )

        if allowed_tools:
            opts.tools = allowed_tools
            opts.allowed_tools = allowed_tools

        # Set optional fields only when non-empty (SDK may reject empty dicts)
        if agents:
            opts.agents = agents
        if mcp_servers:
            opts.mcp_servers = mcp_servers
        if workspace_dir:
            opts.cwd = str(workspace_dir)

        # -- skills: enable SDK filesystem-based skill discovery --
        # Always set setting_sources explicitly to avoid SDK passing an empty
        # string for ``--setting-sources`` when the value is ``None``.  On
        # Windows the empty-string argument is silently dropped by the OS,
        # causing the CLI to swallow the next flag as the option value and
        # ultimately time-out.  "local" is the safe default (no project-level
        # settings); "project" enables skill discovery from ``.claude/``.
        opts.setting_sources = cast(
            list[Literal["user", "project", "local"]],
            ["project"] if agent.skills else ["local"],
        )

        # -- resume: restore a previous session's conversation context --
        if ctx.resume_session_id:
            opts.resume = ctx.resume_session_id
            opts.fork_session = ctx.fork_session

        # -- resume implies continue_conversation (unless forking) --
        if ctx.resume_session_id and not ctx.fork_session:
            opts.continue_conversation = True
            
        return opts