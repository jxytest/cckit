"""OpenTelemetry tracing middleware for agent execution.

Records agent lifecycle events (tool calls, sub-agent delegation) as
span events.  Cost and usage tracking is handled by LiteLLM's native
OTEL callback — this middleware does NOT duplicate that concern.
"""

from __future__ import annotations

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
    except ImportError:
        _sdk_types = {}
    return _sdk_types


def _record_message_event(span: Any, message: Any) -> None:
    """Extract tool-use / sub-agent events from an SDK message."""
    types = _load_sdk_types()
    if not types:
        return

    AssistantMessage = types.get("AssistantMessage")
    TaskStartedMessage = types.get("TaskStartedMessage")
    TaskNotificationMessage = types.get("TaskNotificationMessage")
    ToolUseBlock = types.get("ToolUseBlock")
    ToolResultBlock = types.get("ToolResultBlock")

    if AssistantMessage and isinstance(message, AssistantMessage):
        for block in getattr(message, "content", []):
            if ToolUseBlock and isinstance(block, ToolUseBlock):
                span.add_event("tool_use", {"tool.name": block.name, "tool.id": block.id})
            elif ToolResultBlock and isinstance(block, ToolResultBlock):
                attrs: dict[str, Any] = {"tool.id": block.tool_use_id}
                if block.is_error:
                    attrs["tool.is_error"] = True
                span.add_event("tool_result", attrs)
        return

    if TaskStartedMessage and isinstance(message, TaskStartedMessage):
        span.add_event("subagent.started", {
            "subagent.task_id": message.task_id,
            "subagent.description": message.description,
        })
        return

    if TaskNotificationMessage and isinstance(message, TaskNotificationMessage):
        attrs: dict[str, Any] = {"subagent.task_id": message.task_id}
        status = getattr(message, "status", None)
        if status:
            attrs["subagent.status"] = status
        span.add_event("subagent.completed", attrs)


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

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
            # Pin the active OTEL context onto the in-process bridge so
            # every gen_ai.chat span becomes a child of THIS span. The
            # Claude Code CLI runs as a Node.js subprocess and cannot
            # inject ``traceparent`` HTTP headers, so header-based
            # propagation does not work for the CLI -> bridge hop.
            bridge = getattr(state, "bridge", None)
            if bridge is not None:
                try:
                    parent_ctx = _current_otel_context()
                    if parent_ctx is not None:
                        bridge.set_parent_otel_context(parent_ctx)
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
            try:
                async for message in next_call(prompt, options, state):
                    message_count += 1
                    _record_message_event(span, message)
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
                span.record_exception(exc)
                try:
                    from opentelemetry.trace import StatusCode

                    span.set_status(StatusCode.ERROR)
                except ImportError:
                    pass
                raise
            finally:
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