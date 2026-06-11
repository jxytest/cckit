"""OpenTelemetry tracing middleware for agent execution.

Records agent lifecycle events (tool calls, sub-agent delegation) as
span events.  Cost and usage tracking is handled by LiteLLM's native
OTEL callback — this middleware does NOT duplicate that concern.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from cckit.middleware.base import Middleware, SdkQueryFunc
from cckit.telemetry import get_tracer
from cckit.types import RunContext

# SDK message types — imported lazily to avoid hard dependency.
_sdk_types: dict[str, type] | None = None

# Soft truncation cap for trace-level input/output attributes (langfuse).
_TRACE_ATTR_MAX_BYTES = 100_000

# Trace-level attribute keys that langfuse recommends propagating to every
# observation in the trace (so server-side filters/aggregations work).
_TRACE_PROPAGATED_KEYS = (
    "langfuse.session.id",
    "session.id",
    "langfuse.user.id",
    "user.id",
    "langfuse.release",
    "langfuse.version",
    "langfuse.environment",
    "langfuse.trace.tags",
    "langfuse.trace.name",
)

# Tool names the Claude Agent SDK uses to spawn sub-agents. Newer SDKs
# (>=2.x) emit "Agent"; older ones used "Task". Accept both so sub-agent
# spans are detected regardless of SDK version.
_SUBAGENT_TOOL_NAMES = ("Agent", "Task")


def _truncate(value: str) -> str:
    if len(value) <= _TRACE_ATTR_MAX_BYTES:
        return value
    return value[:_TRACE_ATTR_MAX_BYTES] + "...[truncated]"


def _current_otel_context() -> Any:
    """Return the active OpenTelemetry Context, or None when SDK absent."""
    try:
        from opentelemetry import context as otel_context

        return otel_context.get_current()
    except ImportError:
        return None


def _set_span_in_context(span: Any, base_ctx: Any) -> Any:
    """Return an OTEL Context where *span* is the active span.

    Used so that child spans started later (sub-agent's LLM/tool calls)
    pick this span up as their parent. Falls back to ``base_ctx`` when
    the OpenTelemetry SDK is not installed.
    """
    try:
        from opentelemetry import trace as otel_trace

        return otel_trace.set_span_in_context(span, base_ctx)
    except ImportError:
        return base_ctx
    except Exception:
        return base_ctx


def _load_sdk_types() -> dict[str, type]:
    global _sdk_types
    if _sdk_types is not None:
        return _sdk_types
    try:
        from claude_agent_sdk.types import (
            AssistantMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            ToolResultBlock,
            ToolUseBlock,
        )

        _sdk_types = {
            "AssistantMessage": AssistantMessage,
            "TaskStartedMessage": TaskStartedMessage,
            "TaskNotificationMessage": TaskNotificationMessage,
            "ToolUseBlock": ToolUseBlock,
            "ToolResultBlock": ToolResultBlock,
        }
        # UserMessage carries ToolResultBlocks in current SDK versions.
        try:
            from claude_agent_sdk.types import UserMessage  # type: ignore

            _sdk_types["UserMessage"] = UserMessage
        except ImportError:
            pass
    except ImportError:
        _sdk_types = {}
    return _sdk_types


def _serialize_tool_value(value: Any) -> str:
    """Serialize a tool input/output for ``langfuse.observation.input/output``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value)
    try:
        return _truncate(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        try:
            return _truncate(str(value))
        except Exception:
            return ""


class _ChildSpanRegistry:
    """Track in-flight tool / sub-agent spans by their SDK identifier.

    Every tool call (including Task, the sub-agent launcher) is keyed
    by its ``tool_use_id``.  We also remember the OTEL Context in
    which each tool span is active, so that *child* operations
    (a sub-agent's nested tool calls, or its LLM requests routed
    through the bridge) can use it as their parent context.

    Span hierarchy:

    - ``cckit.agent.execute``                ← root span / "main agent"
        - ``tool.Bash`` (parent_tool_use_id=None) ← main agent calls Bash
        - ``subagent.researcher`` (Task)       ← main agent invokes sub-agent
            - ``gen_ai.chat`` (via bridge fingerprint routing)
            - ``tool.WebSearch`` (parent_tool_use_id=Task's id)
            - ``gen_ai.chat``
        - ``gen_ai.chat``                      ← main agent's next LLM call

    On agent shutdown, ``end_all`` finalises any orphaned spans so a
    crash mid-tool does not leak unfinished observations to langfuse.
    """

    __slots__ = ("entries",)

    def __init__(self) -> None:
        # tool_use_id -> {span, ctx, is_subagent, task_type}
        self.entries: dict[str, dict[str, Any]] = {}

    def remember(
        self,
        tool_use_id: str,
        *,
        span: Any,
        ctx: Any,
        is_subagent: bool = False,
        task_type: str | None = None,
        task_prompt: str = "",
    ) -> None:
        self.entries[tool_use_id] = {
            "span": span,
            "ctx": ctx,
            "is_subagent": is_subagent,
            "task_type": task_type,
            "task_prompt": task_prompt,
        }

    def get(self, tool_use_id: str | None) -> dict[str, Any] | None:
        if not tool_use_id:
            return None
        return self.entries.get(tool_use_id)

    def pop(self, tool_use_id: str | None) -> dict[str, Any] | None:
        if not tool_use_id:
            return None
        return self.entries.pop(tool_use_id, None)

    def end_all(self, *, error: BaseException | None = None) -> list[dict[str, Any]]:
        leaked: list[dict[str, Any]] = []
        for entry in self.entries.values():
            span = entry["span"]
            try:
                if error is not None:
                    try:
                        from opentelemetry.trace import StatusCode

                        span.set_status(StatusCode.ERROR)
                    except ImportError:
                        pass
                    try:
                        span.record_exception(error)
                    except Exception:
                        pass
                span.set_attribute("cckit.span_unclosed", True)
                span.end()
            except Exception:
                pass
            leaked.append(entry)
        self.entries.clear()
        return leaked


def _resolve_parent_ctx(
    registry: _ChildSpanRegistry,
    parent_tool_use_id: str | None,
    main_ctx: Any,
) -> Any:
    """Return the OTEL Context whose active span should be the parent."""
    if parent_tool_use_id:
        entry = registry.get(parent_tool_use_id)
        if entry is not None:
            return entry["ctx"]
    return main_ctx


def _record_message_event(
    tracer: Any,
    main_span: Any,
    main_ctx: Any,
    registry: _ChildSpanRegistry,
    bridge: Any,
    message: Any,
) -> None:
    """Translate one SDK message into child spans for langfuse.

    See :class:`_ChildSpanRegistry` for the resulting hierarchy.
    Anything we cannot map is silently ignored — telemetry must never
    block message delivery.
    """
    types = _load_sdk_types()
    if not types:
        return

    AssistantMessage = types.get("AssistantMessage")
    UserMessage = types.get("UserMessage")
    TaskStartedMessage = types.get("TaskStartedMessage")
    TaskNotificationMessage = types.get("TaskNotificationMessage")
    ToolUseBlock = types.get("ToolUseBlock")
    ToolResultBlock = types.get("ToolResultBlock")

    # Tool blocks may show up on assistant messages (the model emitting a
    # tool_use) or on user messages (the SDK feeding the tool_result back).
    is_block_carrier = (
        (AssistantMessage and isinstance(message, AssistantMessage))
        or (UserMessage and isinstance(message, UserMessage))
    )
    if is_block_carrier:
        # parent_tool_use_id tells us which sub-agent (if any) emitted
        # this message — used to nest its tool/subagent spans correctly.
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        parent_ctx = _resolve_parent_ctx(registry, parent_tool_use_id, main_ctx)
        for block in getattr(message, "content", None) or []:
            if ToolUseBlock and isinstance(block, ToolUseBlock):
                _open_tool_or_subagent_span(
                    tracer, registry, bridge, block, parent_ctx,
                )
            elif ToolResultBlock and isinstance(block, ToolResultBlock):
                _close_tool_or_subagent_span(
                    registry, bridge, block, main_span,
                )
        return

    if TaskStartedMessage and isinstance(message, TaskStartedMessage):
        _decorate_task_started(tracer, registry, bridge, message, main_ctx)
        return

    if TaskNotificationMessage and isinstance(message, TaskNotificationMessage):
        _decorate_task_notification(registry, message)


def _open_tool_or_subagent_span(
    tracer: Any,
    registry: _ChildSpanRegistry,
    bridge: Any,
    block: Any,
    parent_ctx: Any,
) -> None:
    """Open a tool span. Task-tool calls are upgraded to sub-agent spans."""
    tool_id = getattr(block, "id", None)
    if not tool_id or tool_id in registry.entries:
        return
    name = getattr(block, "name", "tool") or "tool"
    tool_input = getattr(block, "input", None)

    is_subagent = name in _SUBAGENT_TOOL_NAMES
    task_prompt: str = ""
    if is_subagent and isinstance(tool_input, dict):
        # Standard Task-tool input keys per Claude Agent SDK.
        subagent_type = (
            tool_input.get("subagent_type")
            or tool_input.get("agent")
            or tool_input.get("agent_type")
        )
        # ``prompt`` is the actual text the sub-agent sees as its first
        # user message — much more distinctive than ``description`` and
        # therefore the preferred routing key. Keep ``description`` as
        # a span-name fallback only.
        task_prompt = str(tool_input.get("prompt") or "")
        description = (
            tool_input.get("description")
            or task_prompt
            or ""
        )
        first_line = description.strip().splitlines()[0] if description else ""
        span_name = f"subagent.{subagent_type or first_line[:60] or 'task'}"
        attrs: dict[str, Any] = {
            "langfuse.observation.type": "span",
            "tool.name": name,
            "tool.id": str(tool_id),
            "subagent.tool_use_id": str(tool_id),
        }
        if subagent_type:
            attrs["subagent.task_type"] = str(subagent_type)
    else:
        subagent_type = None
        span_name = f"tool.{name}"
        attrs = {
            "langfuse.observation.type": "span",
            "tool.name": str(name),
            "tool.id": str(tool_id),
        }

    try:
        span = tracer.start_span(span_name, context=parent_ctx, attributes=attrs)
    except Exception:
        return

    if tool_input is not None:
        try:
            value = _serialize_tool_value(tool_input)
            if value:
                span.set_attribute("langfuse.observation.input", value)
        except Exception:
            pass

    # Build the OTEL Context whose active span is THIS span. Children
    # opened later (sub-agent's LLM calls via bridge, nested tool calls
    # via SDK messages) use it as their parent.
    span_ctx = _set_span_in_context(span, parent_ctx)

    registry.remember(
        str(tool_id),
        span=span,
        ctx=span_ctx,
        is_subagent=is_subagent,
        task_type=str(subagent_type) if subagent_type else None,
        task_prompt=task_prompt,
    )

    # Register the sub-agent's invocation with the bridge so its LLM
    # requests resolve to THIS span as their parent. Both signals are
    # forwarded — the bridge prefers the prompt match (per-invocation,
    # disambiguates parallel calls) and falls back to system signature.
    if is_subagent and bridge is not None:
        if task_prompt:
            try:
                bridge.push_task_prompt(task_prompt, span_ctx)
            except Exception:
                pass
        if subagent_type:
            try:
                bridge.push_subagent_context(str(subagent_type), span_ctx)
            except Exception:
                pass


def _close_tool_or_subagent_span(
    registry: _ChildSpanRegistry,
    bridge: Any,
    block: Any,
    main_span: Any,
) -> None:
    """Close the span associated with a ToolResultBlock."""
    tool_id = getattr(block, "tool_use_id", None)
    entry = registry.pop(tool_id)
    if entry is None:
        # Result without a matching open span — emit a marker event so
        # the information isn't lost entirely.
        try:
            main_span.add_event(
                "tool_result.orphan",
                {"tool.id": str(tool_id) if tool_id else ""},
            )
        except Exception:
            pass
        return

    span = entry["span"]
    is_error = bool(getattr(block, "is_error", False))
    if is_error:
        try:
            span.set_attribute("tool.is_error", True)
            span.set_attribute("langfuse.observation.level", "ERROR")
            from opentelemetry.trace import StatusCode

            span.set_status(StatusCode.ERROR)
        except ImportError:
            pass
        except Exception:
            pass

    content = getattr(block, "content", None)
    if content is not None:
        try:
            value = _serialize_tool_value(content)
            if value:
                span.set_attribute("langfuse.observation.output", value)
        except Exception:
            pass

    # Pop bridge routing entries BEFORE ending the span — once ended,
    # late sub-agent requests should fall through to the main parent
    # rather than attaching to a finished span.
    if entry.get("is_subagent") and bridge is not None:
        prompt = entry.get("task_prompt") or ""
        if prompt:
            try:
                bridge.pop_task_prompt(prompt, entry["ctx"])
            except Exception:
                pass
        task_type = entry.get("task_type")
        if task_type:
            try:
                bridge.pop_subagent_context(task_type, entry["ctx"])
            except Exception:
                pass

    try:
        span.end()
    except Exception:
        pass


def _decorate_task_started(
    tracer: Any,
    registry: _ChildSpanRegistry,
    bridge: Any,
    message: Any,
    main_ctx: Any,
) -> None:
    """Add task metadata to the sub-agent span (or open one if missing)."""
    tool_use_id = getattr(message, "tool_use_id", None)
    task_id = getattr(message, "task_id", None)
    task_type = getattr(message, "task_type", None) or None
    description = getattr(message, "description", "") or ""

    entry = registry.get(tool_use_id)
    if entry is None:
        # Defensive: TaskStartedMessage arrived without a preceding
        # ToolUseBlock(name="Task"). Open a synthetic sub-agent span so
        # the task is still observable.
        first_line = description.strip().splitlines()[0] if description else ""
        span_name = f"subagent.{task_type or first_line[:60] or 'task'}"
        attrs = {
            "langfuse.observation.type": "span",
            "subagent.task_id": str(task_id) if task_id else "",
            "subagent.task_type": str(task_type) if task_type else "",
        }
        try:
            span = tracer.start_span(span_name, context=main_ctx, attributes=attrs)
        except Exception:
            return
        if description:
            try:
                span.set_attribute(
                    "langfuse.observation.input",
                    _serialize_tool_value(description),
                )
            except Exception:
                pass
        span_ctx = _set_span_in_context(span, main_ctx)
        # Use a synthetic key so the span can still be closed by id.
        synthetic_key = f"task:{task_id}" if task_id else f"task:{id(span)}"
        registry.remember(
            synthetic_key,
            span=span,
            ctx=span_ctx,
            is_subagent=True,
            task_type=str(task_type) if task_type else None,
        )
        if bridge is not None and task_type:
            try:
                bridge.push_subagent_context(str(task_type), span_ctx)
            except Exception:
                pass
        return

    span = entry["span"]
    try:
        if task_id:
            span.set_attribute("subagent.task_id", str(task_id))
        if task_type:
            span.set_attribute("subagent.task_type", str(task_type))
            entry["task_type"] = str(task_type)
        if description:
            # Backfill input if it wasn't set at ToolUseBlock time
            # (input there came from block.input, but description in
            # TaskStartedMessage may be more readable).
            span.set_attribute("subagent.description", str(description))
    except Exception:
        pass


def _decorate_task_notification(
    registry: _ChildSpanRegistry, message: Any,
) -> None:
    """Apply terminal task metadata (status, usage, summary) to the span.

    Span is NOT closed here — the matching ToolResultBlock will do that.
    """
    tool_use_id = getattr(message, "tool_use_id", None)
    entry = registry.get(tool_use_id)
    if entry is None and getattr(message, "task_id", None):
        # Fall back to the synthetic key used in _decorate_task_started.
        entry = registry.get(f"task:{message.task_id}")
    if entry is None:
        return
    span = entry["span"]
    try:
        status = getattr(message, "status", None)
        if status:
            span.set_attribute("subagent.status", str(status))
            if str(status).lower() in ("failed", "error", "cancelled"):
                span.set_attribute("langfuse.observation.level", "ERROR")
                try:
                    from opentelemetry.trace import StatusCode

                    span.set_status(StatusCode.ERROR)
                except ImportError:
                    pass
        summary = getattr(message, "summary", None)
        if summary:
            span.set_attribute("subagent.summary", _serialize_tool_value(summary))
        usage = getattr(message, "usage", None)
        if usage is not None:
            try:
                span.set_attribute(
                    "subagent.usage",
                    _serialize_tool_value(usage),
                )
            except Exception:
                pass
    except Exception:
        pass


class TracingMiddleware(Middleware):
    """Emit OpenTelemetry spans for agent execution.

    Records tool calls, tool results, and sub-agent lifecycle as span
    events.  Cost/usage is NOT recorded here — that is the responsibility
    of LiteLLM's native OTEL callback configured on the bridge.

    Parameters
    ----------
    attributes:
        Static span attributes applied to every execution handled by this
        middleware instance (e.g. ``{"service.namespace": "my-app"}``).
        Per-run dynamic attributes are supplied via ``RunContext.span_attributes``.
    """

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        self._static_attributes: dict[str, Any] = dict(attributes or {})

    async def wrap(
        self,
        next_call: SdkQueryFunc,
        prompt: str,
        options: Any,
        state: Any,
        ctx: RunContext,
    ) -> AsyncIterator[Any]:
        tracer = get_tracer("cckit")
        span_name = ctx.span_name or "cckit.agent.execute"

        # Priority: static (deployment-level) < per-run (ctx.span_attributes) < cckit-own
        attributes: dict[str, Any] = {
            **self._static_attributes,
            **ctx.span_attributes,
            "cckit.task_id": ctx.task_id,
            # Langfuse: this span is a logical "span" observation, not a
            # generation. We also expose the trace name explicitly so it
            # shows up nicely in the langfuse UI even when the root span
            # name is overridden.
            "langfuse.observation.type": "span",
            "langfuse.trace.name": span_name,
        }
        if ctx.user:
            attributes["cckit.user"] = ctx.user
        # Capture the agent prompt as the trace-level input. Without this
        # the langfuse UI shows an empty Input column on the trace row.
        if prompt:
            attributes["langfuse.trace.input"] = _truncate(str(prompt))

        # Honour user-supplied session.id — never overwrite it with the
        # Claude SDK's session id later. Same for user.id.
        user_provided_session = (
            "langfuse.session.id" in attributes or "session.id" in attributes
        )

        registry = _ChildSpanRegistry()

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
            # The Context whose active span is the main agent span —
            # used as the default parent for top-level child spans
            # (those without a parent_tool_use_id).
            main_ctx = _set_span_in_context(span, _current_otel_context())

            # Pin the active OTEL context onto the in-process bridge so
            # every gen_ai.chat span becomes a child of THIS span. The
            # Claude Code CLI runs as a Node.js subprocess and cannot
            # inject ``traceparent`` HTTP headers, so header-based
            # propagation does not work for the CLI -> bridge hop.
            bridge = getattr(state, "bridge", None)
            if bridge is not None:
                try:
                    if main_ctx is not None:
                        bridge.set_parent_otel_context(main_ctx)
                    # Forward trace-level attributes (session.id, user.id,
                    # tags, …) so the bridge can stamp them on every LLM
                    # observation it emits — required for langfuse filters
                    # to work at the observation level.
                    propagated = {
                        k: attributes[k]
                        for k in _TRACE_PROPAGATED_KEYS
                        if k in attributes
                    }
                    if propagated:
                        bridge.set_trace_attributes(propagated)
                except Exception:
                    # Tracing must never break the actual agent run.
                    pass

            message_count = 0
            session_already_set = False
            run_error: BaseException | None = None
            try:
                async for message in next_call(prompt, options, state):
                    message_count += 1
                    try:
                        _record_message_event(
                            tracer, span, main_ctx, registry, bridge, message,
                        )
                    except Exception:
                        # Telemetry must never break the message stream.
                        pass
                    # As soon as the SDK reports the session id (init
                    # message), propagate it onto the span and onto the
                    # bridge so subsequent LLM observations carry it too.
                    if (
                        not session_already_set
                        and getattr(state, "session_id", "")
                    ):
                        sid = state.session_id
                        # Don't overwrite a user-supplied session.id;
                        # cckit-side attributes have lower priority than
                        # ctx.span_attributes by design.
                        if not user_provided_session:
                            span.set_attribute("langfuse.session.id", sid)
                            span.set_attribute("session.id", sid)
                        if bridge is not None:
                            try:
                                merged = dict(getattr(bridge, "_trace_attributes", {}) or {})
                                merged.setdefault("langfuse.session.id", sid)
                                merged.setdefault("session.id", sid)
                                bridge.set_trace_attributes(merged)
                            except Exception:
                                pass
                        session_already_set = True
                    yield message
            except Exception as exc:
                run_error = exc
                span.record_exception(exc)
                try:
                    from opentelemetry.trace import StatusCode

                    span.set_status(StatusCode.ERROR)
                except ImportError:
                    pass
                raise
            finally:
                # Close any child spans (tool / sub-agent) that never
                # received their matching close event — typically due
                # to early termination or aborted runs. Also unwind any
                # subagent contexts still pinned on the bridge.
                try:
                    leaked = registry.end_all(error=run_error)
                    if bridge is not None:
                        for entry in leaked:
                            if not entry.get("is_subagent"):
                                continue
                            prompt = entry.get("task_prompt") or ""
                            if prompt:
                                try:
                                    bridge.pop_task_prompt(prompt, entry["ctx"])
                                except Exception:
                                    pass
                            tt = entry.get("task_type")
                            if tt:
                                try:
                                    bridge.pop_subagent_context(tt, entry["ctx"])
                                except Exception:
                                    pass
                except Exception:
                    pass

                span.set_attribute("cckit.message_count", message_count)
                if state.session_id:
                    span.set_attribute("cckit.session_id", state.session_id)
                    if not session_already_set and not user_provided_session:
                        span.set_attribute("langfuse.session.id", state.session_id)
                        span.set_attribute("session.id", state.session_id)

                # Capture the agent's final answer as the trace output so
                # the langfuse UI shows a non-empty Output column.
                final_msg = getattr(state, "final_message", None)
                if final_msg is not None:
                    output_text = getattr(final_msg, "result", None) or ""
                    if output_text:
                        span.set_attribute(
                            "langfuse.trace.output", _truncate(str(output_text)),
                        )

                # Detach the bridge binding so the bridge does not leak
                # this Context into a subsequent run (bridges may be
                # reused on retries via the same prepared_model object).
                if bridge is not None:
                    try:
                        bridge.set_parent_otel_context(None)
                    except Exception:
                        pass