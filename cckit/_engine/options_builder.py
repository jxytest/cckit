"""``ClaudeAgentOptions`` construction.

Translates the declarative ``Agent`` + ``RunContext`` + resolved model into
the SDK options object: allowed tools, sub-agent definitions, MCP servers,
sandbox settings, environment, and the assorted SDK capability passthroughs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cckit._engine.model_transport import resolve_model_transport
from cckit._engine.runtime_env import build_agent_env
from cckit._engine.tracing import log_startup_config
from cckit.sandbox.config import SandboxConfigBuilder

if TYPE_CHECKING:
    from cckit._engine.model_bridge import PreparedModelEndpoint
    from cckit._engine.state import RunState
    from cckit.agent import Agent
    from cckit.types import ModelConfig, RunContext, RunnerConfig, SandboxOptions

logger = logging.getLogger(__name__)


def build_options(
        agent: Agent,
        ctx: RunContext,
        model: ModelConfig,
        prepared_model: PreparedModelEndpoint,
        sandbox: SandboxOptions,
        permission_mode: str,
        workspace_dir: Path | None,
        instruction: str,
        state: RunState,
        *,
        config: RunnerConfig,
        resolve_model: Callable[..., ModelConfig],
) -> Any:
    """Construct ``ClaudeAgentOptions`` from Agent + RunContext + resolved model.

    ``resolve_model`` resolves a sub-agent's effective ModelConfig; it is
    injected so this module stays free of Runner state.
    """
    from claude_agent_sdk import ClaudeAgentOptions  # noqa: WPS433

    tools = _build_tools(agent)
    allowed_tools = _build_allowed_tools(agent, tools)
    agents = _build_sub_agent_definitions(agent, ctx, prepared_model, resolve_model)
    mcp_servers = agent.mcp_servers
    _warn_on_unregistered_mcp_servers(agent, allowed_tools, mcp_servers)

    # -- sandbox --
    # build() returns a unified settings JSON string (or None when disabled).
    # ClaudeAgentOptions.sandbox must be None to avoid the SDK overwriting
    # the sandbox section in settings JSON with a SandboxSettings TypedDict.
    settings_json = _build_sandbox_settings(sandbox, workspace_dir)

    # -- environment (Agent subprocess only — NO git credentials) --
    env = build_agent_env(agent, ctx, model, prepared_model)

    # -- configurable SDK params --
    max_turns = agent.max_turns if agent.max_turns > 0 else model.max_turns

    def _stderr_cb(line: str) -> None:
        state.observe_stderr(line)
        logger.debug("[CLI stderr] %s", line.rstrip())

    system_prompt: dict[str, str] = {
        "type": "preset",
        "preset": "claude_code",
    }
    if instruction:
        system_prompt["append"] = instruction

    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=max_turns,
        model=prepared_model.model,
        # When sandbox is enabled, switch to dontAsk so that permissions.deny
        # rules are enforced. bypassPermissions skips all permission checks
        # (including deny rules) and is only safe inside pre-isolated envs.
        # Otherwise honor the effective permission mode (which may already
        # have been downgraded from bypassPermissions to dontAsk when
        # running as root — see Runner._execute).
        permission_mode=(
            "dontAsk"
            if sandbox.enabled
            else permission_mode
        ),
        env=env,
        stderr=_stderr_cb,
        sandbox=None,
        settings=settings_json,
        extra_args={"debug-to-stderr": None},
        user=ctx.user,
        include_partial_messages=ctx.include_partial_messages,
    )

    if tools and tools != []:
        opts.tools = tools
    if allowed_tools:
        opts.allowed_tools = allowed_tools

    # Set optional fields only when non-empty (SDK may reject empty dicts)
    if agents:
        opts.agents = agents
    if mcp_servers:
        opts.mcp_servers = mcp_servers
    if workspace_dir:
        opts.cwd = str(workspace_dir)

    _apply_setting_sources(opts)
    _apply_resume(opts, ctx)
    _apply_session_store(opts, ctx, config)
    _apply_sdk_capabilities(opts, agent, model)

    log_startup_config(
        agent=agent,
        ctx=ctx,
        prepared_model_name=prepared_model.model,
        opts=opts,
        max_turns=max_turns,
        workspace_dir=workspace_dir,
        allowed_tools=allowed_tools,
        agents=agents,
        mcp_servers=mcp_servers,
        sandbox_enabled=sandbox.enabled,
        env=env,
        resolve_model=resolve_model,
    )

    return opts


# ----------------------------------------------------------------------
# Tools / sub-agents / MCP
# ----------------------------------------------------------------------


def _build_tools(agent: Agent) -> list[str]:
    """Agent's visible tool set: declared tools plus the implicit tools its
    capabilities require (``Agent`` for delegation, ``Skill`` for skills)."""
    tools = list(agent.tools)
    if agent.sub_agents and "Agent" not in tools:
        tools.append("Agent")
    if agent.skills and "Skill" not in tools:
        tools.append("Skill")
    return tools


def _build_allowed_tools(agent: Agent, tools: list[str]) -> list[str]:
    """Permission allow-list = visible tools ∪ ``agent.allowed_tools`` (deduped).

    ``agent.allowed_tools`` lets a top-level agent auto-approve extra tool names
    (e.g. a sub-agent's ``mcp__<server>__<tool>``) without exposing them to the
    top-level model. Empty → identical to ``tools`` (backward compatible with
    the pre-split behaviour where ``opts.allowed_tools == opts.tools``).
    """
    allowed = list(tools)
    for name in agent.allowed_tools:
        if name not in allowed:
            allowed.append(name)
    return allowed


def _build_sub_agent_definitions(
        agent: Agent,
        ctx: RunContext,
        prepared_model: PreparedModelEndpoint,
        resolve_model: Callable[..., ModelConfig],
) -> dict[str, Any]:
    """Translate ``agent.sub_agents`` into SDK ``AgentDefinition`` objects."""
    from claude_agent_sdk import AgentDefinition  # noqa: WPS433

    agents: dict[str, AgentDefinition] = {}
    for sub in agent.sub_agents:
        # Auto-inject "Skill" tool for sub-agents that declare skills
        sub_tools = list(sub.tools) if sub.tools else None
        if sub.skills and sub_tools is not None and "Skill" not in sub_tools:
            sub_tools.append("Skill")

        # Resolve the model name that gets passed to AgentDefinition.
        # When a bridge is active, sub-agents keep their full LiteLLM model
        # name (e.g. "openai/gpt-4o-mini") so the bridge can route the
        # request to the correct provider.  Without a bridge (all models use
        # Anthropic), strip the LiteLLM prefix so the CLI receives a bare
        # Anthropic model ID it understands (e.g. "claude-haiku-4-5").
        sub_cfg = resolve_model(sub, ctx, is_sub_agent=True)
        if prepared_model.bridge is not None:
            # Bridge is active — keep the full model name for routing.
            agent_model_name: str | None = sub_cfg.model
        else:
            # No bridge — all models are Anthropic.  Strip provider prefix
            # so the CLI gets a native model ID (e.g. "claude-haiku-4-5").
            sub_transport = resolve_model_transport(sub_cfg)
            agent_model_name = sub_transport.model

        agents[sub.name] = AgentDefinition(
            description=sub.description,
            prompt=sub.resolve_instruction(ctx),
            tools=sub_tools,
            disallowedTools=list(sub.disallowed_tools) if sub.disallowed_tools else None,
            model=agent_model_name,
            skills=list(sub.skills) if sub.skills else None,
            mcpServers=(
                [
                    {name: cfg} if isinstance(cfg, dict) else name
                    for name, cfg in sub.mcp_servers.items()
                ]
                if sub.mcp_servers
                else None
            ),
            maxTurns=sub.max_turns if sub.max_turns > 0 else None,
            effort=sub.effort,
        )
    return agents


def _warn_on_unregistered_mcp_servers(
        agent: Agent,
        allowed_tools: list[str],
        mcp_servers: dict[str, Any] | None,
) -> None:
    """Log loudly when an ``mcp__<server>__<tool>`` has no registered server.

    An agent may declare ``mcp__<server>__<tool>`` in its tool list while the
    corresponding server is missing from ``mcp_servers`` (e.g. a spec builds
    the server conditionally but declares the tool names unconditionally).
    The CLI then has nothing to route the call to and every such tool silently
    returns "(<tool> completed with no output)", which looks like "the tool
    found no data" rather than a wiring bug.  Surface it at startup instead.
    """
    declared_servers = {
        name.split("__")[1]
        for name in allowed_tools
        if name.startswith("mcp__") and name.count("__") >= 2
    }
    for sub in agent.sub_agents:
        for name in sub.tools or ():
            if name.startswith("mcp__") and name.count("__") >= 2:
                declared_servers.add(name.split("__")[1])
    missing_servers = declared_servers - set(mcp_servers or {})
    if missing_servers:
        logger.error(
            "Agent %s declares MCP tools for server(s) %s but no such server is "
            "registered in mcp_servers=%s. Every call to those tools will return "
            "an empty result. Check the spec that builds this agent — the server "
            "is likely created conditionally while the tool names are declared "
            "unconditionally.",
            agent.name,
            sorted(missing_servers),
            sorted(mcp_servers or {}),
        )


def _build_sandbox_settings(
        sandbox: SandboxOptions, workspace_dir: Path | None,
) -> str | None:
    return SandboxConfigBuilder(
        enabled=sandbox.enabled,
        allow_write=list(sandbox.allow_write),
        deny_write=list(sandbox.deny_write),
        allow_read=list(sandbox.allow_read),
        deny_read=list(sandbox.deny_read),
        allowed_domains=list(sandbox.allowed_domains),
        denied_domains=list(sandbox.denied_domains),
        auto_allow_bash=sandbox.auto_allow_bash,
        excluded_commands=list(sandbox.excluded_commands),
        allow_unsandboxed_commands=sandbox.allow_unsandboxed_commands,
        enable_weaker_nested_sandbox=sandbox.enable_weaker_nested_sandbox,
    ).build(workspace_dir)


# ----------------------------------------------------------------------
# Options post-processing
# ----------------------------------------------------------------------


def _apply_setting_sources(opts: Any) -> None:
    """Enable SDK filesystem-based skill discovery.

    Always set setting_sources explicitly to avoid the SDK passing an empty
    string for ``--setting-sources`` when the value is ``None``.  On Windows
    the empty-string argument is silently dropped by the OS, causing the CLI
    to swallow the next flag as the option value and ultimately time out.
    """
    opts.setting_sources = cast(
        list[Literal["user", "project", "local"]],
        ["user", "project", "local"],
    )


def _apply_resume(opts: Any, ctx: RunContext) -> None:
    """Restore a previous session's conversation context."""
    if not ctx.resume_session_id:
        return
    opts.resume = ctx.resume_session_id
    opts.fork_session = ctx.fork_session
    # resume implies continue_conversation (unless forking)
    if not ctx.fork_session:
        opts.continue_conversation = True


def _apply_session_store(opts: Any, ctx: RunContext, config: RunnerConfig) -> None:
    """Wire the pluggable session store (optional).

    Per-run value takes precedence over the runner-level default. When both
    are None (the default), ``opts.session_store`` is left unset so the SDK
    falls back to its built-in ~/.claude/projects/ file persistence. Flush
    defaults to "eager" (safest for single-shot runs); override via
    ctx/RunnerConfig.session_store_flush = "batched" for higher throughput.
    """
    session_store = ctx.session_store or config.session_store
    if session_store is None:
        return
    opts.session_store = session_store
    opts.session_store_flush = (
        ctx.session_store_flush or config.session_store_flush or "eager"
    )


def _apply_sdk_capabilities(opts: Any, agent: Agent, model: ModelConfig) -> None:
    """Pass Agent-declared SDK capabilities through to the options object."""
    # -- Claude native hooks --
    if agent.hooks:
        opts.hooks = agent.hooks  # type: ignore[assignment]

    # -- task budget --
    if agent.task_budget is not None:
        from claude_agent_sdk.types import TaskBudget  # noqa: WPS433
        opts.task_budget = TaskBudget(total=agent.task_budget.total)

    # -- effort (top-level agent) --
    # Historically agent.effort was only passed to sub-agents
    # (AgentDefinition.effort) and dropped for the main session. Wire it
    # into the top-level options too. Default None leaves opts.effort unset
    # (SDK default), so existing behavior is unchanged when not specified.
    if agent.effort:
        opts.effort = agent.effort

    # -- thinking (direct-Anthropic CLI path) --
    # Priority: ModelConfig.thinking > deprecated disable_thinking > unset.
    # The LiteLLM bridge path handles disable_thinking separately
    # (model_bridge.py sets kwargs["thinking"] on the HTTP request); this
    # only governs the CLI subprocess via opts.thinking. Both paths are
    # non-conflicting. Default (None + disable_thinking=False) leaves
    # opts.thinking unset, matching prior behavior.
    thinking_cfg = getattr(model, "thinking", None)
    if thinking_cfg is None and getattr(model, "disable_thinking", False):
        from cckit.types import ThinkingConfig  # noqa: WPS433
        thinking_cfg = ThinkingConfig.disabled()
    if thinking_cfg is not None:
        opts.thinking = thinking_cfg.to_sdk()

    # -- custom permission handler (can_use_tool) --
    if agent.permission_handler is not None:
        opts.can_use_tool = agent.permission_handler

    # -- structured output --
    if agent.output_format:
        opts.output_format = agent.output_format

    # -- USD budget hard cap --
    if agent.max_budget_usd is not None:
        opts.max_budget_usd = agent.max_budget_usd

    # -- SDK beta features (e.g. 1M context) --
    if agent.betas:
        opts.betas = list(agent.betas)

    # -- local plugins --
    if agent.plugins:
        opts.plugins = [
            p if isinstance(p, dict) else {"type": "local", "path": p}
            for p in agent.plugins
        ]
