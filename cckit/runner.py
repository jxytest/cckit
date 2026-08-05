"""Runner — executes agents and yields streaming SDK messages.

This is the orchestration layer.  It is the only class that ties together
workspace management, git cloning, skill provisioning, middleware, and the
SDK bridge.  The mechanical work is delegated to focused modules:

===============================  ===========================================
Module                           Responsibility
===============================  ===========================================
``_engine.options_builder``      ``ClaudeAgentOptions`` construction
``_engine.model_resolver``       Agent model ⨯ Runner default merge
``_engine.runtime_env``          Subprocess env + host-env isolation
``_engine.tracing``              SDK message logging, cost recalculation
``_engine.session_files``        Session JSONL read/restore
``skill.planner``                Skill collection / provisioning / repair
``middleware.chain``             Middleware stack assembly
===============================  ===========================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from cckit._cli import check_api_connectivity, check_claude_cli
from cckit._engine import session_files
from cckit._engine.model_bridge import PreparedModelEndpoint, prepare_model_endpoint
from cckit._engine.model_resolver import resolve_model as _resolve_model_config
from cckit._engine.options_builder import build_options
from cckit._engine.runtime_env import is_root_user, is_windows
from cckit._engine.state import RunState
from cckit._engine.tracing import (
    log_run_summary,
    log_sdk_message,
    patch_result_message_costs,
)
from cckit.agent import Agent
from cckit.exceptions import HookError
from cckit.git import operations as git_ops
from cckit.middleware.base import Middleware
from cckit.middleware.chain import build_middleware_chain
from cckit.sandbox.workspace import WorkspaceManager
from cckit.skill import planner as skill_planner
from cckit.skill.provisioner import SkillProvisioner
from cckit.types import (
    AgentResult,
    ModelConfig,
    RunContext,
    RunnerConfig,
    SandboxOptions,
    StreamResult,
    TaskStatus,
    _ResultHolder,
)

logger = logging.getLogger(__name__)


class Runner:
    """Execute an Agent and yield SDK message objects.

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

        # LiteLLM's own loggers default to DEBUG and emit a line per
        # completion() call, which floods the stream when cckit runs at INFO.
        # Align them with the configured log_level so one setting governs the
        # whole run. LITELLM_LOG is LiteLLM's documented override: when the
        # caller set it explicitly, leave their choice alone.
        if not os.environ.get("LITELLM_LOG"):
            for _name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
                logging.getLogger(_name).setLevel(_level)
        self._middlewares: list[Middleware] = middlewares or []
        self._preflight_check = preflight_check

        self._workspace = workspace_manager or WorkspaceManager(
            root=self._config.workspace_root
        )
        self._skill_provisioner = skill_provisioner or SkillProvisioner(
            skills_dir=self._config.skills_dir
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
        messages and returns the :class:`AgentResult`.  The returned object
        is the **same instance** that lifecycle callbacks (``on_after``)
        receive, so any mutations (e.g. ``result.extra["mr_url"] = ...``)
        are preserved.
        """
        stream = self.run_stream(agent, ctx)
        async for _message in stream:
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
        """Run *agent* and return a :class:`StreamResult` for streaming messages.

        ``StreamResult`` is async-iterable, so existing code continues to
        work unchanged::

            async for message in runner.run_stream(agent, ctx):
                print(message)

        New usage — access the final result after the stream ends::

            stream = runner.run_stream(agent, ctx)
            async for message in stream:
                print(message)
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
    ) -> AsyncIterator[Any]:
        """Internal async generator — the full orchestration flow.

        Orchestration flow:
            1. Validate context (required_params, structural checks)
            2. agent.before_execute(ctx)
            3. Preflight API connectivity check (if enabled)
            4. Create/resume workspace
            5. Git clone (if needed, skipped on resume)
            6. Provision skills (if needed, skipped on resume)
            7. agent.prepare_workspace(ctx) — seed files (skipped on resume)
            8. Resolve instruction (string or callable)
            9. Build SDK options (model, tools, sub-agents, MCP, sandbox)
            10. [Middleware chain] → SDK bridge → yield SDK messages
            11. agent.after_execute(ctx, result) or agent.error_execute(ctx, error)
            12. Cleanup/suspend workspace
        """
        start = time.monotonic()
        git_cfg = ctx.resolved_git()
        effective_sandbox = self._resolve_sandbox(agent)
        # Effective permission mode for THIS run. Defaults to the configured
        # mode; downgraded to ``dontAsk`` when running as root with
        # ``bypassPermissions`` (Claude Code rejects --dangerously-skip-permissions
        # under root/sudo). Local variable so a downgraded run never mutates the
        # reusable RunnerConfig — see the root check below.
        effective_permission_mode: str = self._config.permission_mode
        prepared_model: PreparedModelEndpoint | None = None
        sdk_stream: AsyncIterator[Any] | None = None

        try:
            # --- validate context ---
            missing = self._validate_context(agent, ctx)
            if missing:
                msg = f"Missing required parameters: {', '.join(missing)}"
                holder.result = AgentResult(
                    task_id=ctx.task_id,
                    agent_type=agent.name,
                    status=TaskStatus.FAILED,
                    is_error=True,
                    error_message=msg,
                    duration_seconds=round(time.monotonic() - start, 2),
                )
                logger.error("Agent %s failed validation: %s", agent.name, msg)
                return

            # --- lifecycle: before ---
            await agent.before_execute(ctx)

            effective_sandbox, effective_permission_mode, _downgrade_event = (
                self._apply_root_permission_downgrade(
                    agent, effective_sandbox, effective_permission_mode,
                )
            )
            if _downgrade_event is not None:
                yield _downgrade_event

            if effective_sandbox.enabled and is_windows():
                logger.warning(
                    "Sandbox is enabled for agent %s, but native Windows does not support "
                    "OS-level sandbox enforcement. Use macOS, Linux, or WSL2 for full isolation.",
                    agent.name,
                )

            # --- workspace ---
            if ctx.workspace.enabled:
                await self._setup_workspace(agent, ctx, holder, git_cfg)

            # --- instruction ---
            instruction = agent.resolve_instruction(ctx)

            # --- prompt ---
            prompt = ctx.prompt or ""

            state = RunState(ctx.task_id)
            # --- build SDK options ---
            model = self._resolve_model(agent, ctx)

            # Collect sub-agent models that differ from the main model so the
            # bridge can route requests to the correct provider/credentials.
            extra_models: dict[str, ModelConfig] = {}
            for sub in agent.sub_agents:
                sub_cfg = self._resolve_model(sub, ctx, is_sub_agent=True)
                if sub_cfg.model != model.model:
                    extra_models[sub_cfg.model] = sub_cfg

            prepared_model = await prepare_model_endpoint(
                model, extra_models=extra_models or None,
            )
            # Expose the in-process bridge to the tracing middleware so it
            # can register the active OTEL context. This is the only way to
            # make gen_ai.chat spans children of cckit.agent.execute, since
            # the Claude Code CLI subprocess cannot inject traceparent.
            state.bridge = prepared_model.bridge

            self._register_subagent_systems(agent, ctx, prepared_model)

            if self._preflight_check:
                check_api_connectivity(
                    api_key=prepared_model.api_key or model.api_key,
                    base_url=prepared_model.base_url,
                    model=prepared_model.model,
                )
            options = self._build_options(
                agent,
                ctx,
                model,
                prepared_model,
                effective_sandbox,
                effective_permission_mode,
                holder.workspace_dir,
                instruction,
                state,
            )

            # --- stream from SDK (through middleware chain) ---
            query_fn = self._build_middleware_chain(ctx)
            sdk_stream = query_fn(prompt, options, state)

            # Build short-name → ModelConfig map once for cost recalculation
            all_configs = self._build_cost_config_map(model, extra_models)

            async for message in sdk_stream:
                # Trace every SDK message (tool calls, tool results, text,
                # result summary).  This is the only place the full agent↔CLI
                # interaction is observable from the Python side, so keep it
                # unconditional — it is what turns "the tool returned nothing"
                # into an actionable log line.
                log_sdk_message(message, ctx)

                # Recalculate costUSD on ResultMessage before yielding so that
                # downstream consumers (event serialisers, loggers, etc.) always
                # receive accurate pricing without needing their own patches.
                message = patch_result_message_costs(message, all_configs)

                # Lifecycle: on_message
                try:
                    await agent.on_message_received(ctx, message)
                except Exception:
                    logger.debug("on_message callback failed", exc_info=True)
                yield message

            # --- build result ---
            # state.final_message is the same object already patched above when
            # it was yielded, so no second recalculation is needed here.
            holder.result = self._build_result(
                agent, ctx, state, duration=time.monotonic() - start,
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
            logger.exception("Agent %s failed: %s", agent.name, exc)

            # --- lifecycle: on_error ---
            try:
                await agent.error_execute(ctx, exc)
            except Exception as hook_exc:
                logger.exception("error_execute hook failed for %s", agent.name)
                raise HookError("error_execute", hook_exc) from hook_exc

        finally:
            # --- close SDK stream first (kills CLI subprocess) ---
            # When this generator is closed early (e.g. via aclose()),
            # async generator cleanup does NOT guarantee inner generators
            # are finalized before the outer finally runs.  Explicitly
            # close the SDK stream so the subprocess is terminated before
            # we shut down the model bridge it connects to.
            if sdk_stream is not None:
                aclose_fn = getattr(sdk_stream, 'aclose', None)
                if aclose_fn is not None:
                    try:
                        await aclose_fn()
                    except Exception:
                        logger.debug("sdk_stream aclose failed", exc_info=True)

            # --- lifecycle: after ---
            if holder.result is not None:
                try:
                    await agent.after_execute(ctx, holder.result)
                except Exception as hook_exc:
                    logger.exception(
                        "after_execute hook failed for %s", agent.name
                    )
                    raise HookError("after_execute", hook_exc) from hook_exc

                log_run_summary(agent.name, holder.result)

            # --- cleanup temporary askpass script ---
            try:
                git_cfg.cleanup_askpass()
            except Exception:
                logger.exception("Failed to cleanup git askpass for task %s", ctx.task_id)

            await self._finalize_workspace(ctx, holder)

            if prepared_model is not None:
                await prepared_model.aclose()

    # ------------------------------------------------------------------
    # Execution steps
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_root_permission_downgrade(
            agent: Agent,
            sandbox: SandboxOptions,
            permission_mode: str,
    ) -> tuple[SandboxOptions, str, Any | None]:
        """Downgrade ``bypassPermissions`` → ``dontAsk`` when running as root.

        Claude Code rejects ``--dangerously-skip-permissions`` under root/sudo.
        Instead of failing, transparently enable the sandbox and downgrade for
        this run, then notify the caller via a system event.  Two things must
        happen together:

          1. Enable the sandbox so ``SandboxConfigBuilder`` emits the unified
             settings JSON (``sandbox.*`` + ``permissions.*`` +
             ``autoAllowBashIfSandboxed``). Without these settings, a bare
             ``dontAsk`` run still prompts for MCP/Bash tool authorization
             under root — the sandbox settings are what actually lets tools
             run without per-call approval.
          2. Switch the effective permission mode to ``dontAsk``. The sandbox
             branch in the options builder would do this anyway, but setting it
             here keeps the downgraded state explicit and makes the
             ``permission_degraded`` event accurate.

        ``dontAsk`` still enforces ``permissions.deny`` rules (unlike
        ``bypassPermissions``, which skips all checks), so this is the safe
        equivalent for a root env — isolated by the sandbox rather than by
        privilege.

        Returns ``(sandbox, permission_mode, event)``; *event* is ``None``
        when no downgrade applies, and the caller yields it otherwise.
        """
        if permission_mode != "bypassPermissions" or not is_root_user():
            return sandbox, permission_mode, None

        if not sandbox.enabled:
            sandbox = SandboxOptions(enabled=True)
        logger.warning(
            "Agent %s running as root with permission_mode='bypassPermissions' "
            "is not supported by Claude Code; enabling sandbox and downgrading "
            "to 'dontAsk' for this run.",
            agent.name,
        )
        from claude_agent_sdk import SystemMessage  # noqa: WPS433
        event = SystemMessage(
            subtype="permission_degraded",
            data={
                "original": "bypassPermissions",
                "effective": "dontAsk",
                "reason": "root_user",
                "sandbox_auto_enabled": True,
                "detail": (
                    "Claude Code rejects --dangerously-skip-permissions when "
                    "running as root/sudo; auto-enabled sandbox and downgraded "
                    "to dontAsk for this run."
                ),
            },
        )
        return sandbox, "dontAsk", event

    async def _setup_workspace(
            self,
            agent: Agent,
            ctx: RunContext,
            holder: _ResultHolder,
            git_cfg: Any,
    ) -> None:
        """Create or resume the workspace, then clone / provision as needed."""
        needs_init = False
        if ctx.resume_session_id and ctx.workspace_dir:
            # Resume: reuse existing workspace, or recreate at the
            # same path when the directory was cleaned up.
            holder.workspace_dir, was_recreated = await self._workspace.resume(
                ctx.workspace_dir, recreate=True
            )
            needs_init = was_recreated
        else:
            # First execution: create a fresh workspace
            holder.workspace_dir = await self._workspace.create(ctx.task_id)
            ctx.workspace_dir = holder.workspace_dir
            needs_init = True

        if needs_init:
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

            # --- provision skills (top-level + sub-agents) ---
            await self._provision_agent_skills(agent, holder.workspace_dir)

            # --- lifecycle: prepare_workspace ---
            await agent.prepare_workspace(ctx)
        elif ctx.resume_session_id:
            # Self-heal: the workspace dir survived resume (was_recreated
            # = False) so the needs_init block above skipped provisioning.
            # But the host may have wiped the tmpdir between turns and
            # recreated an empty dir before cckit ran (or otherwise left
            # .claude/skills/ absent). Re-provision so the agent can still
            # discover its skills on a resumed turn.
            if self._agent_needs_skill_repair(agent, holder.workspace_dir):
                await self._provision_agent_skills(agent, holder.workspace_dir)

    @staticmethod
    def _register_subagent_systems(
            agent: Agent,
            ctx: RunContext,
            prepared_model: PreparedModelEndpoint,
    ) -> None:
        """Register sub-agent system signatures with the model bridge.

        Sub-agent LLM observations then attach to the right
        ``subagent.<name>`` span (vs. the main agent span). Each sub-agent has
        a distinct instruction string, which the bridge fingerprints against
        incoming requests.  Telemetry registration must never break a run.
        """
        if prepared_model.bridge is None or not agent.sub_agents:
            return

        sub_systems: dict[str, str] = {}
        for sub in agent.sub_agents:
            try:
                sub_systems[sub.name] = sub.resolve_instruction(ctx) or ""
            except Exception:
                logger.debug(
                    "Failed to resolve sub-agent instruction for %s",
                    sub.name, exc_info=True,
                )
        if not sub_systems:
            return
        try:
            prepared_model.bridge.register_subagent_systems(sub_systems)
        except Exception:
            logger.debug("bridge.register_subagent_systems failed", exc_info=True)

    @staticmethod
    def _build_cost_config_map(
            model: ModelConfig,
            extra_models: dict[str, ModelConfig],
    ) -> dict[str, ModelConfig]:
        """Map short model name → ModelConfig for cost recalculation."""

        def _short(name: str) -> str:
            return name.split("/")[-1] if "/" in name else name

        all_configs: dict[str, ModelConfig] = {_short(model.model): model}
        for sub_cfg in extra_models.values():
            all_configs[_short(sub_cfg.model)] = sub_cfg
        return all_configs

    @staticmethod
    def _build_result(
            agent: Agent,
            ctx: RunContext,
            state: RunState,
            *,
            duration: float,
    ) -> AgentResult:
        """Assemble the final :class:`AgentResult` from the run state."""
        final_message = state.final_message
        if final_message is None:
            return AgentResult(
                task_id=ctx.task_id,
                agent_type=agent.name,
                status=TaskStatus.FAILED,
                is_error=True,
                error_message="SDK stream completed without a ResultMessage",
                duration_seconds=round(duration, 2),
                session_id=state.session_id,
            )

        output_text = final_message.result or ""
        is_error = bool(final_message.is_error)
        return AgentResult(
            task_id=ctx.task_id,
            agent_type=agent.name,
            status=TaskStatus.FAILED if is_error else TaskStatus.COMPLETED,
            output_text=output_text,
            cost_usd=final_message.total_cost_usd or 0.0,
            duration_seconds=round(duration, 2),
            is_error=is_error,
            error_message=output_text if is_error else "",
            session_id=final_message.session_id or state.session_id,
            stop_reason=final_message.stop_reason or "",
            usage=final_message.usage,
            structured_output=final_message.structured_output,
            final_message=final_message,
        )

    async def _finalize_workspace(
            self, ctx: RunContext, holder: _ResultHolder,
    ) -> None:
        """Suspend or delete the workspace according to the cleanup policy.

        Policy:
          - workspace.keep=True  → always suspend (caller wants to resume later)
          - task failed          → suspend (preserve for debugging / resume)
          - task succeeded       → cleanup (delete) by default
        """
        if not holder.workspace_dir:
            return
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
    # Session persistence helpers (delegated to _engine.session_files)
    # ------------------------------------------------------------------

    @staticmethod
    def read_session(session_id: str, workspace_dir: Path) -> str | None:
        """Read a Claude Code session JSONL from ~/.claude/projects/."""
        return session_files.read_session(session_id, workspace_dir)

    @staticmethod
    def restore_session(session_id: str, workspace_dir: Path, content: str) -> None:
        """Restore a persisted session JSONL so --resume works."""
        session_files.restore_session(session_id, workspace_dir, content)

    @staticmethod
    def read_session_dir(workspace_dir: Path) -> dict[str, bytes] | None:
        """Read all session files from the project directory (recursively)."""
        return session_files.read_session_dir(workspace_dir)

    @staticmethod
    def restore_session_dir(workspace_dir: Path, files: dict[str, bytes]) -> None:
        """Restore multiple session files to the project directory."""
        session_files.restore_session_dir(workspace_dir, files)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model(
        self,
        agent: Agent,
        ctx: RunContext | None = None,
        *,
        is_sub_agent: bool = False,
    ) -> ModelConfig:
        """Merge agent model_config with runner defaults."""
        return _resolve_model_config(
            agent,
            ctx,
            self._config.default_model,
            is_sub_agent=is_sub_agent,
        )

    def _resolve_sandbox(self, agent: Agent) -> SandboxOptions:
        """Return the sandbox policy for this run."""
        return agent.sandbox_config or SandboxOptions()

    # ------------------------------------------------------------------
    # Skill collection (delegated to skill.planner)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_all_skills(agent: Agent) -> list[str]:
        """Collect deduplicated skill names from agent and its sub-agents."""
        return skill_planner.collect_all_skills(agent)

    async def _provision_agent_skills(
        self, agent: Agent, workspace_dir: Path,
    ) -> None:
        """Provision all declared skills into the workspace's ``.claude/skills/``."""
        await skill_planner.provision_agent_skills(
            agent, workspace_dir, self._skill_provisioner,
        )

    @staticmethod
    def _agent_needs_skill_repair(agent: Agent, workspace_dir: Path) -> bool:
        """True when a declared skill is missing from the workspace."""
        return skill_planner.needs_skill_repair(agent, workspace_dir)

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
        any_skills = agent.skills or any(sub.skills for sub in agent.sub_agents)
        if any_skills and not ctx.workspace.enabled:
            missing.append("workspace.enabled (required when skills are declared)")

        return missing

    # ------------------------------------------------------------------
    # Delegating builders
    # ------------------------------------------------------------------

    def _build_middleware_chain(self, ctx: RunContext) -> Any:
        """Wrap ``run_sdk_query`` with the middleware stack.

        Returns a callable with signature ``(prompt, options, state)``.
        """
        return build_middleware_chain(self._middlewares, ctx)

    def _build_options(
            self,
            agent: Agent,
            ctx: RunContext,
            model: ModelConfig,
            prepared_model: PreparedModelEndpoint,
            sandbox: SandboxOptions,
            permission_mode: str,
            workspace_dir: Path | None,
            instruction: str,
            state: RunState,
    ) -> Any:
        """Construct ``ClaudeAgentOptions`` from Agent + RunContext + resolved model."""
        return build_options(
            agent,
            ctx,
            model,
            prepared_model,
            sandbox,
            permission_mode,
            workspace_dir,
            instruction,
            state,
            config=self._config,
            resolve_model=self._resolve_model,
        )
