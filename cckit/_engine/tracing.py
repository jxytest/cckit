"""SDK message tracing and cost post-processing.

Everything here is observability-only: it turns raw SDK messages into log
lines and patches recalculated pricing onto ``ResultMessage``.  None of it
may ever raise — tracing must not be able to break a run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cckit._cost import recalculate_model_usage_costs

if TYPE_CHECKING:
    from cckit.types import ModelConfig, RunContext

logger = logging.getLogger(__name__)

TRACE_MAX = 2000

# SystemMessage subtypes that fire at near-token frequency and carry no
# diagnostic payload. Dropped from tracing entirely at any log level.
NOISY_SYSTEM_SUBTYPES: frozenset[str] = frozenset(
    {
        "thinking_tokens",  # emitted per reasoning step — tens per turn
        "background_tasks_changed",  # internal task-registry bookkeeping
    }
)

# Tools whose input is a full sub-agent prompt (hundreds of lines). The prompt
# is already visible in the startup sub-agent config, so trace only enough to
# identify which invocation this is.
_PROMPT_HEAVY_TOOLS: frozenset[str] = frozenset({"Agent", "Task"})
_PROMPT_HEAVY_INPUT_MAX = 200


def patch_result_message_costs(
        message: Any,
        all_configs: dict[str, ModelConfig],
) -> Any:
    """Patch ``model_usage`` and ``total_cost_usd`` on a ``ResultMessage`` in-place.

    If *message* is not a ``ResultMessage`` or has no ``model_usage``, it is
    returned unchanged.  The patch is applied via ``object.__setattr__`` so it
    works on frozen dataclasses too.
    """
    try:
        from claude_agent_sdk import ResultMessage  # noqa: WPS433
    except ImportError:
        return message

    if not isinstance(message, ResultMessage):
        return message

    model_usage = getattr(message, "model_usage", None)
    if not model_usage or not isinstance(model_usage, dict):
        return message

    try:
        recalculated = recalculate_model_usage_costs(model_usage, all_configs)
        new_total_cost = sum(
            u.get("costUSD", 0.0) for u in recalculated.values()
            if isinstance(u, dict)
        )
        object.__setattr__(message, "model_usage", recalculated)
        object.__setattr__(message, "total_cost_usd", new_total_cost)
    except Exception:
        logger.debug("Could not patch ResultMessage costs", exc_info=True)

    return message


def describe_mcp_servers(mcp_servers: dict[str, Any] | None) -> dict[str, str]:
    """Render ``mcp_servers`` as ``{name: kind}`` for logging.

    Values are SDK server objects / config dicts that have no useful repr, so
    only the transport kind is kept.  An empty result is itself the signal that
    no MCP server reached the CLI.
    """
    described: dict[str, str] = {}
    for name, cfg in (mcp_servers or {}).items():
        if isinstance(cfg, dict):
            described[name] = str(cfg.get("type", "dict"))
        else:
            described[name] = type(cfg).__name__
    return described


def truncate(text: str, limit: int = TRACE_MAX) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, total {len(text)} chars]"


def render_content(content: Any) -> str:
    """Flatten an SDK content value (str | list[block] | None) to one line."""
    if content is None:
        return "<empty>"
    if isinstance(content, str):
        return truncate(content) if content.strip() else "<empty>"
    if isinstance(content, list):
        if not content:
            return "<empty list>"
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                kind = block.get("type")
                if kind == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(f"<{kind}>")
            else:
                parts.append(str(block))
        joined = " ".join(p for p in parts if p)
        return truncate(joined) if joined.strip() else "<empty blocks>"
    return truncate(str(content))


def _log_system_message(message: Any, task: str) -> None:
    """Log one ``SystemMessage``, keeping INFO reserved for ``init``.

    Only the ``init`` subtype carries the CLI's own view of what got wired up
    (mcp_servers, tools, model) — the ground truth to compare against the
    startup config.  Every other subtype has an empty ``data`` for those keys,
    so printing them at INFO produced a stream of ``mcp_servers=None
    tools=None model=None`` lines that drown out the useful ones.

    High-frequency internal subtypes (``thinking_tokens`` fires per reasoning
    step) are dropped entirely rather than merely demoted — they carry no
    diagnostic content at any level.
    """
    subtype = getattr(message, "subtype", None)
    if subtype in NOISY_SYSTEM_SUBTYPES:
        return

    data = getattr(message, "data", None)
    if subtype == "init" and isinstance(data, dict):
        logger.info(
            "[%s] system subtype=%s mcp_servers=%s tools=%s model=%s",
            task,
            subtype,
            truncate(repr(data.get("mcp_servers")), 800),
            truncate(repr(data.get("tools")), 800),
            data.get("model"),
        )
    else:
        logger.debug("[%s] system subtype=%s", task, subtype)


def log_sdk_message(message: Any, ctx: RunContext) -> None:
    """Log one SDK message: tool uses, tool results, assistant text, result.

    Never raises — tracing must not be able to break a run.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    try:
        task = ctx.task_id
        blocks = getattr(message, "content", None)
        kind = type(message).__name__

        if isinstance(blocks, list):
            for block in blocks:
                btype = type(block).__name__
                if btype == "ToolUseBlock":
                    tool_name = getattr(block, "name", "?")
                    limit = (
                        _PROMPT_HEAVY_INPUT_MAX
                        if tool_name in _PROMPT_HEAVY_TOOLS
                        else TRACE_MAX
                    )
                    logger.info(
                        "[%s] tool_use name=%s id=%s input=%s",
                        task,
                        tool_name,
                        getattr(block, "id", "?"),
                        truncate(repr(getattr(block, "input", None)), limit),
                    )
                elif btype == "ToolResultBlock":
                    content = getattr(block, "content", None)
                    rendered = render_content(content)
                    is_error = bool(getattr(block, "is_error", False))
                    # An empty tool result is almost always a wiring bug (the
                    # tool never ran), not a legitimate "no data" answer.
                    log = logger.warning if rendered.startswith("<empty") else logger.info
                    log(
                        "[%s] tool_result id=%s is_error=%s content=%s",
                        task,
                        getattr(block, "tool_use_id", "?"),
                        is_error,
                        rendered,
                    )
                elif btype == "TextBlock":
                    logger.debug(
                        "[%s] text %s", task, truncate(getattr(block, "text", ""), 500),
                    )
            return

        if kind == "ResultMessage":
            logger.info(
                "[%s] result subtype=%s is_error=%s turns=%s duration_ms=%s cost=%s",
                task,
                getattr(message, "subtype", None),
                getattr(message, "is_error", None),
                getattr(message, "num_turns", None),
                getattr(message, "duration_ms", None),
                getattr(message, "total_cost_usd", None),
            )
        elif kind == "SystemMessage":
            _log_system_message(message, task)
    except Exception:  # noqa: BLE001 - tracing must never break a run
        logger.debug("SDK message tracing failed", exc_info=True)


def log_startup_config(
        *,
        agent: Any,
        ctx: RunContext,
        prepared_model_name: str,
        opts: Any,
        max_turns: int,
        workspace_dir: Any,
        allowed_tools: list[str],
        agents: dict[str, Any],
        mcp_servers: dict[str, Any] | None,
        sandbox_enabled: bool,
        env: dict[str, str],
        resolve_model: Any,
) -> None:
    """Log the resolved agent + sub-agent configuration at startup.

    ``resolve_model`` is the runner's sub-agent model resolver, used only as a
    fallback when an ``AgentDefinition`` is missing for a declared sub-agent.
    """
    _secret_markers = (
        "API_KEY", "AUTH_TOKEN", "SECRET", "PASSWORD", "_AUTH", "GPG_KEY", "INVITE_CODE",
    )
    _safe_env = {
        k: ("***" if any(s in k.upper() for s in _secret_markers) else v)
        for k, v in env.items()
    }
    logger.info(
        "Agent startup config: name=%s task_id=%s model=%s "
        "permission_mode=%s max_turns=%d workspace=%s "
        "tools=%s sub_agents=%s skills=%s "
        "mcp_servers=%s sandbox_enabled=%s env_keys=%s",
        agent.name,
        ctx.task_id,
        prepared_model_name,
        opts.permission_mode,
        max_turns,
        str(workspace_dir) if workspace_dir else None,
        allowed_tools,
        list(agents.keys()) if agents else [],
        agent.skills or [],
        describe_mcp_servers(mcp_servers),
        sandbox_enabled,
        _safe_env,
    )

    for sub in agent.sub_agents:
        sub_def = agents.get(sub.name)
        sub_cfg = resolve_model(sub, ctx, is_sub_agent=True)
        logger.info(
            "  Sub-agent config: name=%s model=%s "
            "description=%s tools=%s disallowed_tools=%s "
            "skills=%s max_turns=%s effort=%s "
            "mcp_servers=%s",
            sub.name,
            sub_def.model if sub_def else sub_cfg.model,
            sub.description or "(none)",
            sub_def.tools if sub_def else sub.tools,
            sub_def.disallowedTools if sub_def else sub.disallowed_tools,
            sub_def.skills if sub_def else sub.skills,
            sub_def.maxTurns if sub_def else sub.max_turns,
            sub_def.effort if sub_def else sub.effort,
            list(sub.mcp_servers.keys()) if sub.mcp_servers else [],
        )


def log_run_summary(agent_name: str, result: Any) -> None:
    """Log the final one-line run summary (status, cost, token totals).

    Token counts are aggregated from ``model_usage`` (accurate) rather than
    ``result.usage``, which is the raw SDK usage object and is often zeroed
    for non-Anthropic providers.
    """
    model_usage: dict = {}
    final_message = getattr(result, "final_message", None)
    if final_message is not None:
        model_usage = getattr(final_message, "model_usage", None) or {}
    inp = sum(
        int(u.get("inputTokens", 0)) for u in model_usage.values() if isinstance(u, dict)
    )
    out = sum(
        int(u.get("outputTokens", 0)) for u in model_usage.values() if isinstance(u, dict)
    )
    logger.info(
        (
            "Agent %s completed: task_id=%s session_id=%s "
            "status=%s cost=$%.4f duration=%.2fs "
            "input_tokens=%d output_tokens=%d total_tokens=%d"
        ),
        agent_name,
        result.task_id,
        result.session_id,
        result.status,
        result.cost_usd,
        result.duration_seconds,
        inp,
        out,
        inp + out,
    )
