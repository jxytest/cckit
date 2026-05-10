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
        }
        if ctx.user:
            attributes["cckit.user"] = ctx.user

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
            message_count = 0
            try:
                async for message in next_call(prompt, options, state):
                    message_count += 1
                    _record_message_event(span, message)
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
