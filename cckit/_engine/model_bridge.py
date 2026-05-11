"""Temporary local Anthropic-compatible HTTP bridge backed by LiteLLM.

The bridge starts a lightweight Starlette server on ``127.0.0.1:<random-port>``
that accepts Anthropic ``/v1/messages`` requests and forwards them to the
real provider through LiteLLM.  Anthropic-protocol models bypass the bridge
entirely.

Multi-model routing
-------------------
When sub-agents use different models (or even different providers), the bridge
acts as a **model router**: it inspects the ``model`` field in each incoming
request and dispatches to the correct provider + credentials.  This allows
the main agent to use e.g. ``openai/gpt-4o`` while a sub-agent uses
``anthropic/claude-haiku-4-5`` — all through a single local HTTP server.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from cckit._engine._patches.deepseek_reasoning import patch_deepseek_reasoning
from cckit._engine.model_transport import (
    ResolvedTransport,  # noqa: F401 – used internally
    clamp_max_tokens,
    resolve_model_transport,  # noqa: F401 – re-exported for backward compat
    sanitize_payload,
)
from cckit.exceptions import AgentExecutionError
from cckit.types import ModelConfig

logger = logging.getLogger(__name__)


# ── Direct OTEL span helpers ─────────────────────────────────────


def _extract_trace_context(headers: Any) -> Any:
    """Extract W3C trace context from HTTP request headers.

    Returns a context object to pass as ``context=`` to
    ``start_as_current_span()``, so bridge spans become children of the
    caller's span even though the bridge runs in a separate asyncio Task.
    Returns None when opentelemetry is not installed.
    """
    try:
        from opentelemetry.propagate import extract
        return extract(dict(headers))
    except ImportError:
        return None


def _span_attrs_for_route(route: Any, model_str: str) -> dict[str, Any]:
    """Build GenAI + Langfuse span attributes from a resolved route."""
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": getattr(route.transport, "custom_llm_provider", None) or "anthropic",
        "gen_ai.request.model": model_str,
        # Langfuse: classify this span as a generation (LLM call) observation.
        # Without this, langfuse falls back to inferring type from `model`
        # presence and can mis-categorise tool / sub-agent spans.
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model_str,
    }


# Soft cap to keep span attributes within OTLP exporter limits.
_MAX_ATTR_BYTES = 100_000


def _truncate_for_attr(value: str) -> str:
    """Clip an attribute value to a safe size."""
    if len(value) <= _MAX_ATTR_BYTES:
        return value
    return value[:_MAX_ATTR_BYTES] + "...[truncated]"


def _serialize_input(payload: dict[str, Any]) -> str:
    """Build a JSON string suitable for langfuse.observation.input.

    Includes both ``system`` and ``messages`` so reviewers see the full prompt.
    """
    try:
        body = {
            "system": payload.get("system"),
            "messages": payload.get("messages"),
            # Generation parameters help debugging.
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "tools": payload.get("tools"),
        }
        body = {k: v for k, v in body.items() if v is not None}
        return _truncate_for_attr(json.dumps(body, ensure_ascii=False, default=str))
    except Exception:
        return ""


def _serialize_output_from_anthropic_response(resp: Any) -> str:
    """Convert an Anthropic non-streaming response to a JSON string."""
    try:
        body = (
            resp.model_dump(mode="json", exclude_none=True)
            if hasattr(resp, "model_dump")
            else resp
        )
        # Keep only the user-relevant pieces to avoid bloat.
        slim = {
            "stop_reason": body.get("stop_reason") if isinstance(body, dict) else None,
            "content": body.get("content") if isinstance(body, dict) else body,
        }
        slim = {k: v for k, v in slim.items() if v is not None}
        return _truncate_for_attr(json.dumps(slim, ensure_ascii=False, default=str))
    except Exception:
        return ""


def _try_extract_stream_content(raw: bytes, acc: dict[str, Any]) -> None:
    """Best-effort accumulation of streamed Anthropic SSE deltas.

    Populates ``acc["text"]`` (concatenated text chunks) and
    ``acc["stop_reason"]`` if present.
    """
    try:
        text = raw.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            obj = json.loads(data_str)
            ev = obj.get("type")
            if ev == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    acc.setdefault("text", "")
                    acc["text"] += str(delta.get("text", ""))
                elif delta.get("type") == "thinking_delta":
                    acc.setdefault("thinking", "")
                    acc["thinking"] += str(delta.get("thinking", ""))
            elif ev == "message_delta":
                delta = obj.get("delta") or {}
                if delta.get("stop_reason"):
                    acc["stop_reason"] = delta["stop_reason"]
    except Exception:
        pass


def _compute_cost(usage_data: dict[str, int], route: Any) -> dict[str, float] | None:
    """Compute USD cost split for one LLM call, mirroring cckit/_cost.py.

    Priority:
        1. ``ModelConfig.input_cost_per_token`` / ``output_cost_per_token``
           overrides — applied flat to base input and output (consistent
           with the cost recalculation pass in ``cckit._cost``).
        2. ``litellm.cost_per_token(...)`` — handles Anthropic prompt
           caching markups (×1.25 for creation, ×0.1 for reads) using
           the provider's published rates.

    Returns ``None`` when no price source is available so the caller can
    omit ``cost_details`` rather than emit a misleading $0.
    """
    cfg = getattr(route, "config", None)
    if cfg is None:
        return None

    prompt = int(usage_data.get("prompt_tokens", 0) or 0)
    completion = int(usage_data.get("completion_tokens", 0) or 0)
    cache_creation = int(usage_data.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage_data.get("cache_read_input_tokens", 0) or 0)

    if not (prompt or completion or cache_creation or cache_read):
        return None

    # Priority 1: explicit per-token overrides on ModelConfig.
    in_cost = getattr(cfg, "input_cost_per_token", None)
    out_cost = getattr(cfg, "output_cost_per_token", None)
    if in_cost is not None and out_cost is not None:
        # Match cckit/_cost.py: overrides apply to base input and output;
        # cache tokens are billed at the same input rate (so totals stay
        # consistent across observation and trace level when overrides
        # are configured).
        in_total = (prompt + cache_creation + cache_read) * float(in_cost)
        out_total = completion * float(out_cost)
        return {
            "input": prompt * float(in_cost),
            "input_cache_creation": cache_creation * float(in_cost),
            "input_cache_read": cache_read * float(in_cost),
            "output": out_total,
            "total": in_total + out_total,
        }

    # Priority 2: LiteLLM price table (cache-aware).
    try:
        litellm = importlib.import_module("litellm")
        transport_model = getattr(getattr(route, "transport", None), "model", None)
        for candidate in (transport_model, getattr(cfg, "model", None)):
            if not candidate:
                continue
            try:
                p_cost, c_cost = litellm.cost_per_token(
                    model=candidate,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                )
                if (p_cost or 0) + (c_cost or 0) > 0:
                    return {
                        "input": float(p_cost),
                        "output": float(c_cost),
                        "total": float(p_cost) + float(c_cost),
                    }
            except Exception:
                continue
    except Exception:
        pass

    return None


def _set_usage_on_span(span: Any, usage_data: dict[str, int], route: Any) -> None:
    """Record token counts and cost on a span (gen_ai + langfuse conventions)."""
    prompt = int(usage_data.get("prompt_tokens", 0) or 0)
    completion = int(usage_data.get("completion_tokens", 0) or 0)
    cache_creation = int(usage_data.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage_data.get("cache_read_input_tokens", 0) or 0)

    # Total input includes cache (Anthropic semantics): the model "saw"
    # all of these tokens. Langfuse aggregates per observation, so we
    # need an honest total rather than just the non-cache slice.
    total_input = prompt + cache_creation + cache_read

    # gen_ai semantic-convention attributes (legacy).
    if total_input:
        span.set_attribute("gen_ai.usage.prompt_tokens", total_input)
    if completion:
        span.set_attribute("gen_ai.usage.completion_tokens", completion)
    if total_input or completion:
        span.set_attribute("gen_ai.usage.total_tokens", total_input + completion)

    # Langfuse-preferred: granular usage_details with cache breakdown.
    if total_input or completion:
        details: dict[str, int] = {
            "input": prompt,
            "output": completion,
            "total": total_input + completion,
        }
        if cache_creation:
            details["input_cache_creation"] = cache_creation
        if cache_read:
            details["input_cache_read"] = cache_read
        span.set_attribute(
            "langfuse.observation.usage_details", json.dumps(details),
        )

    cost = _compute_cost(usage_data, route)
    if cost is not None:
        span.set_attribute("gen_ai.usage.cost", cost["total"])
        span.set_attribute(
            "langfuse.observation.cost_details", json.dumps(cost),
        )



def _load_litellm() -> Any:
    """Import LiteLLM lazily so the dependency is only touched at runtime."""
    try:
        return importlib.import_module("litellm")
    except ImportError as exc:
        raise AgentExecutionError(
            "cckit model execution requires LiteLLM bridge dependencies",
            detail="Install `litellm`, `starlette`, and `uvicorn`.",
        ) from exc


def _load_module(name: str) -> Any:
    """Import an optional runtime dependency with a good error."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AgentExecutionError(
            "cckit model execution requires bridge runtime dependencies",
            detail=f"Missing import: {name}",
        ) from exc


def _error_sse_frame(message: str) -> bytes:
    """Encode an Anthropic-style error SSE frame."""
    payload = {"type": "error", "error": {"type": "api_error", "message": message}}
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"event: error\ndata: {body}\n\n".encode()


def _absorb_usage_from_obj(obj: Any, usage_data: dict[str, int]) -> None:
    """Pull token counts out of one Anthropic SSE event payload.

    Handles both layouts (``message_start`` nests ``usage`` under
    ``message``; ``message_delta`` keeps it at top level) and also
    accepts OpenAI-style names so providers whose LiteLLM adapters do
    not fully translate to Anthropic format are still captured.

    Cache tokens (``cache_creation_input_tokens`` /
    ``cache_read_input_tokens``) are recorded separately so that
    pricing — which differs from base input — can be computed
    correctly downstream.
    """
    if not isinstance(obj, dict):
        return

    usage = obj.get("usage")
    if not isinstance(usage, dict):
        msg = obj.get("message")
        if isinstance(msg, dict):
            usage = msg.get("usage")
    if not isinstance(usage, dict):
        return

    # Accept both Anthropic and OpenAI naming.
    inp = usage.get("input_tokens")
    if inp is None:
        inp = usage.get("prompt_tokens")
    out = usage.get("output_tokens")
    if out is None:
        out = usage.get("completion_tokens")

    if inp is not None:
        try:
            usage_data["prompt_tokens"] = int(inp)
        except (TypeError, ValueError):
            pass
    if out is not None:
        try:
            usage_data["completion_tokens"] = int(out)
        except (TypeError, ValueError):
            pass

    cc = usage.get("cache_creation_input_tokens")
    if cc is not None:
        try:
            usage_data["cache_creation_input_tokens"] = int(cc)
        except (TypeError, ValueError):
            pass
    cr = usage.get("cache_read_input_tokens")
    if cr is not None:
        try:
            usage_data["cache_read_input_tokens"] = int(cr)
        except (TypeError, ValueError):
            pass


def _try_extract_usage(raw: bytes, usage_data: dict[str, int]) -> None:
    """Best-effort parse of streaming chunks to accumulate token counts.

    Tries SSE framing first (``data: ...`` lines, the Anthropic format
    LiteLLM normally emits); falls back to parsing the whole payload
    as a single JSON object so we still pick up usage when chunks come
    through as already-decoded objects (some LiteLLM adapter paths).
    """
    try:
        text = raw.decode("utf-8", errors="ignore")
        sse_seen = False
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            sse_seen = True
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
            except Exception:
                continue
            _absorb_usage_from_obj(obj, usage_data)
        if not sse_seen:
            try:
                _absorb_usage_from_obj(json.loads(text), usage_data)
            except Exception:
                pass
    except Exception:
        pass


# ── public data classes ──────────────────────────────────────────


@dataclass(slots=True)
class PreparedModelEndpoint:
    """Resolved model settings that the Claude SDK should see."""

    model: str
    api_key: str
    base_url: str
    bridge: LiteLLMAnthropicBridge | None = None

    async def aclose(self) -> None:
        if self.bridge is not None:
            await self.bridge.aclose()


# ── bridge server ────────────────────────────────────────────────


class _ModelRoute:
    """Pre-resolved route for a single model used by the bridge."""

    __slots__ = ("config", "transport")

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.transport = resolve_model_transport(config)

    def __repr__(self) -> str:
        return (
            f"<_ModelRoute model={self.config.model!r} "
            f"protocol={self.transport.protocol!r}>"
        )


class LiteLLMAnthropicBridge:
    """Temporary local Anthropic-compatible HTTP bridge backed by LiteLLM.

    Parameters
    ----------
    primary:
        The main agent's :class:`ModelConfig`.  Used as the default route
        when a request's ``model`` field doesn't match any registered route.
    extra_models:
        Optional mapping of **model name** → :class:`ModelConfig` for
        sub-agents that need different provider credentials.  The bridge
        dispatches to the matching route based on the ``model`` field in
        each incoming request payload.
    """

    def __init__(
        self,
        primary: ModelConfig,
        extra_models: dict[str, ModelConfig] | None = None,
    ) -> None:
        self._primary = _ModelRoute(primary)
        # Model routing table: model_name → _ModelRoute
        self._routes: dict[str, _ModelRoute] = {primary.model: self._primary}
        if extra_models:
            for name, cfg in extra_models.items():
                self._routes[name] = _ModelRoute(cfg)
        self._server: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self.base_url: str = ""

        # OTEL parent context registered by the cckit tracing middleware.
        # The Claude Code CLI is a Node.js subprocess and cannot propagate
        # `traceparent` HTTP headers, so we pin the parent context here
        # so every gen_ai.chat span becomes a child of cckit.agent.execute.
        self._parent_otel_context: Any = None
        # Trace-level attributes (langfuse.session.id, langfuse.user.id, …)
        # that should be copied onto every gen_ai.chat span so that langfuse
        # filters/aggregations work at the observation level.
        self._trace_attributes: dict[str, Any] = {}

        # Sub-agent routing
        # =================
        # ``_subagent_systems`` maps a sub-agent's ``task_type`` (its name in
        # cckit.Agent) to a *signature* — the textual instruction string we
        # expect to find inside the request's ``system`` field. We use
        # substring matching against this signature to fingerprint incoming
        # HTTP requests back to the sub-agent that issued them, because the
        # Claude Code CLI offers no other in-band identifier.
        #
        # ``_subagent_contexts`` is a per-task_type stack of OTEL Contexts
        # whose active span is the sub-agent's logical ``subagent.<name>``
        # span. The tracing middleware pushes/pops as it observes the SDK
        # message stream (ToolUseBlock(name="Task") / ToolResultBlock).
        self._subagent_systems: dict[str, str] = {}
        self._subagent_contexts: dict[str, list[Any]] = {}

    # ── tracing wiring ────────────────────────────────────────────

    def set_parent_otel_context(self, ctx: Any) -> None:
        """Pin the parent OTEL Context for child gen_ai.chat spans.

        Called by the tracing middleware once ``cckit.agent.execute`` is
        in flight.  ``ctx`` must be an ``opentelemetry.context.Context``;
        passing ``None`` clears the binding.
        """
        self._parent_otel_context = ctx

    def set_trace_attributes(self, attrs: dict[str, Any]) -> None:
        """Set attributes that the bridge will copy onto every LLM span.

        Typical entries: ``langfuse.session.id``, ``langfuse.user.id``,
        ``langfuse.trace.tags`` — anything you need available at the
        observation level for filtering inside langfuse.
        """
        self._trace_attributes = {k: v for k, v in (attrs or {}).items() if v is not None}

    # ── sub-agent routing ────────────────────────────────────────

    def register_subagent_systems(self, mapping: dict[str, str]) -> None:
        """Register the textual signature for each sub-agent's system prompt.

        The runner calls this once per agent run with
        ``{sub.name: sub.resolve_instruction(ctx)}``. Used by
        :meth:`_resolve_parent_context` to fingerprint inbound requests so
        sub-agent LLM observations attach to the right sub-agent span
        instead of the main agent span.

        Empty / very short signatures are dropped because they're prone
        to false-positive substring matches against the main agent's
        system text.
        """
        cleaned: dict[str, str] = {}
        for name, signature in (mapping or {}).items():
            sig = self._flatten_system(signature) if signature else ""
            if name and sig and len(sig) >= 32:
                cleaned[name] = sig
        self._subagent_systems = cleaned

    def push_subagent_context(self, task_type: str, ctx: Any) -> None:
        """Mark *ctx* as the current parent for sub-agent ``task_type``."""
        if not task_type or ctx is None:
            return
        self._subagent_contexts.setdefault(task_type, []).append(ctx)

    def pop_subagent_context(self, task_type: str, ctx: Any) -> None:
        """Remove *ctx* from sub-agent ``task_type``'s active stack."""
        if not task_type:
            return
        stack = self._subagent_contexts.get(task_type)
        if not stack:
            return
        # Prefer identity-based removal so out-of-order completions do
        # not pop the wrong context. Fall back to LIFO when the same
        # ctx was pushed twice (shouldn't happen, but defensive).
        try:
            stack.remove(ctx)
        except ValueError:
            try:
                stack.pop()
            except IndexError:
                pass
        if not stack:
            self._subagent_contexts.pop(task_type, None)

    def _resolve_parent_context(self, payload: dict[str, Any]) -> Any:
        """Pick the OTEL parent for a gen_ai.chat span based on the request.

        Strategy:
            1. Flatten the request's ``system`` field.
            2. Find every registered sub-agent whose signature is a
               substring of it; pick the one with the longest signature
               (most specific).
            3. Use the top of that sub-agent's context stack.
            4. Fall back to the pinned main-agent context.
        """
        if not self._subagent_systems:
            return self._parent_otel_context
        sys_text = self._flatten_system(payload.get("system"))
        if not sys_text:
            return self._parent_otel_context

        best_name: str | None = None
        best_len = 0
        for name, signature in self._subagent_systems.items():
            if signature and signature in sys_text and len(signature) > best_len:
                best_name = name
                best_len = len(signature)

        if best_name:
            stack = self._subagent_contexts.get(best_name)
            if stack:
                return stack[-1]
        return self._parent_otel_context

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> LiteLLMAnthropicBridge:
        """Boot the local bridge server and wait until it is ready."""
        primary_t = self._primary.transport
        logger.debug(
            "Starting bridge: primary=%s protocol=%s provider=%s api_base=%s "
            "routes=%d",
            self._primary.config.model,
            primary_t.protocol,
            primary_t.custom_llm_provider,
            primary_t.api_base,
            len(self._routes),
        )
        for name, route in self._routes.items():
            if name != self._primary.config.model:
                logger.debug(
                    "  route %s → protocol=%s provider=%s api_base=%s",
                    name,
                    route.transport.protocol,
                    route.transport.custom_llm_provider,
                    route.transport.api_base,
                )

        # Apply monkey-patches before first use.
        from cckit._engine._patches._stream_patch import apply_stream_patch
        from cckit._engine._patches.deepseek_reasoning import apply_deepseek_reasoning_patch
        apply_stream_patch()
        apply_deepseek_reasoning_patch()

        uvicorn = _load_module("uvicorn")
        config = uvicorn.Config(
            self._build_app(), host="127.0.0.1", port=0,
            log_level="warning", access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        self._socket = sock

        self.base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))

        for _ in range(100):
            if getattr(self._server, "started", False):
                return self
            if self._task.done():
                error = self._task.exception()
                detail = str(error) if error else "Bridge server exited unexpectedly."
                exc = AgentExecutionError("Failed to start the LiteLLM Anthropic bridge", detail=detail)
                raise exc from error if error else exc
            await asyncio.sleep(0.05)

        await self.aclose()
        raise AgentExecutionError(
            "Timed out waiting for the LiteLLM Anthropic bridge to start",
            detail="The temporary local HTTP server did not become ready within 5 seconds.",
        )

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with suppress(Exception):
                await self._task
        if self._socket is not None:
            self._socket.close()
        self._task = self._server = self._socket = None
        self.base_url = ""

    # ── ASGI app ──────────────────────────────────────────────────

    def _build_app(self) -> Any:
        starlette_app = _load_module("starlette.applications")
        starlette_resp = _load_module("starlette.responses")
        starlette_rt = _load_module("starlette.routing")
        litellm = _load_litellm()
        litellm.drop_params = True

        Route = starlette_rt.Route
        JSONResponse = starlette_resp.JSONResponse
        StreamingResponse = starlette_resp.StreamingResponse
        Response = starlette_resp.Response

        async def health(_: Any) -> Any:
            return JSONResponse({"ok": True})

        async def create_message(request: Any) -> Any:
            payload = await request.json()
            try:
                kwargs = self._build_kwargs(payload)
                route = self._resolve_route(payload.get("model"))
                # Resolve the parent OTEL context for this request:
                # sub-agent stack first (so sub-agent LLM calls nest under
                # their subagent.<name> span), then the pinned main-agent
                # context, finally header-based extraction. The Claude
                # Code CLI subprocess cannot inject traceparent so the
                # header path is essentially never used.
                parent_ctx = (
                    self._resolve_parent_context(payload)
                    or _extract_trace_context(request.headers)
                )
                if payload.get("stream"):
                    kwargs["stream"] = True
                    return StreamingResponse(
                        self._wrap_stream(kwargs, route, parent_ctx, payload),
                        media_type="text/event-stream",
                        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                    )
                from cckit.telemetry import get_tracer
                tracer = get_tracer("cckit.litellm")
                attrs = _span_attrs_for_route(route, kwargs.get("model", ""))
                # Propagate trace-level attrs (session.id, user.id, …) onto
                # every LLM span so langfuse can filter at observation level.
                attrs.update(self._trace_attributes)
                input_json = _serialize_input(payload)
                if input_json:
                    attrs["langfuse.observation.input"] = input_json
                with tracer.start_as_current_span(
                    "gen_ai.chat",
                    context=parent_ctx,
                    attributes=attrs,
                ) as span:
                    resp = await litellm.anthropic.messages.acreate(**kwargs)
                    # Telemetry must never break the LLM response path.
                    # Any failure recording usage/output is swallowed so
                    # the caller still receives the model's reply.
                    try:
                        usage = getattr(resp, "usage", None)
                        if usage:
                            # Pull every field through the same accessor —
                            # tolerant of pydantic-like objects, dicts, and
                            # OpenAI-style naming. Cache tokens included so
                            # cost matches the streaming path exactly.
                            def _u(name: str) -> int:
                                value = (
                                    getattr(usage, name, None)
                                    if not isinstance(usage, dict)
                                    else usage.get(name)
                                )
                                try:
                                    return int(value or 0)
                                except (TypeError, ValueError):
                                    return 0

                            usage_data: dict[str, int] = {
                                "prompt_tokens": _u("input_tokens") or _u("prompt_tokens"),
                                "completion_tokens": _u("output_tokens") or _u("completion_tokens"),
                                "cache_creation_input_tokens": _u("cache_creation_input_tokens"),
                                "cache_read_input_tokens": _u("cache_read_input_tokens"),
                            }
                            _set_usage_on_span(span, usage_data, route)
                        output_json = _serialize_output_from_anthropic_response(resp)
                        if output_json:
                            span.set_attribute("langfuse.observation.output", output_json)
                    except Exception:
                        logger.debug("telemetry recording failed", exc_info=True)
                body = resp.model_dump(mode="json", exclude_none=True) if hasattr(resp, "model_dump") else resp
                return JSONResponse(body)
            except Exception as exc:
                return JSONResponse(
                    {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                    status_code=500,
                )

        async def count_tokens(request: Any) -> Any:
            payload = await request.json()
            return JSONResponse({"input_tokens": self._count_tokens(payload)})

        return starlette_app.Starlette(routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/messages", create_message, methods=["POST"]),
            Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
            Route("/{path:path}", lambda _: Response(status_code=404), methods=["GET", "POST"]),
        ])

    # ── routing ──────────────────────────────────────────────────

    def _resolve_route(self, request_model: str | None) -> _ModelRoute:
        """Look up the route for *request_model*, falling back to primary."""
        if request_model and request_model in self._routes:
            return self._routes[request_model]
        # Fallback: use the primary (main agent) route.
        return self._primary

    # ── request building ─────────────────────────────────────────

    def _build_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build kwargs for ``litellm.anthropic.messages.acreate``.

        Dispatches to the correct provider credentials by looking up the
        ``model`` field in the incoming request against the routing table.
        """
        route = self._resolve_route(payload.get("model"))
        transport = route.transport
        cfg = route.config

        kwargs = sanitize_payload(payload, transport)
        kwargs = clamp_max_tokens(kwargs, transport, cfg.max_tokens)
        kwargs = patch_deepseek_reasoning(kwargs, transport.model)
        kwargs["model"] = transport.model
        kwargs["custom_llm_provider"] = transport.custom_llm_provider
        if cfg.max_tokens is not None:
            kwargs.setdefault("max_tokens", cfg.max_tokens)
        kwargs["timeout"] = cfg.timeout_seconds
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if transport.api_base:
            kwargs["api_base"] = transport.api_base
        return kwargs

    # ── streaming ─────────────────────────────────────────────────

    async def _wrap_stream(
        self,
        kwargs: dict[str, Any],
        route: _ModelRoute,
        parent_ctx: Any = None,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """Start the LLM call, forward SSE bytes, and emit an OTEL span.

        The span covers the entire streaming duration — from the initial
        ``acreate()`` call to the last byte yielded to the client.
        """
        from cckit.telemetry import get_tracer
        tracer = get_tracer("cckit.litellm")
        litellm = _load_litellm()
        usage_data: dict[str, int] = {}
        content_acc: dict[str, Any] = {}
        attrs = _span_attrs_for_route(route, kwargs.get("model", ""))
        # Propagate trace-level attrs onto each LLM span (langfuse filtering).
        attrs.update(self._trace_attributes)
        if payload is not None:
            input_json = _serialize_input(payload)
            if input_json:
                attrs["langfuse.observation.input"] = input_json
        with tracer.start_as_current_span(
            "gen_ai.chat",
            context=parent_ctx,
            attributes=attrs,
        ) as span:
            try:
                stream = await litellm.anthropic.messages.acreate(**kwargs)
                async for chunk in stream:
                    if isinstance(chunk, (bytes, bytearray)):
                        raw = bytes(chunk)
                    elif isinstance(chunk, str):
                        raw = chunk.encode()
                    else:
                        raw = json.dumps(chunk).encode()
                    _try_extract_usage(raw, usage_data)
                    _try_extract_stream_content(raw, content_acc)
                    yield raw
            except Exception as exc:
                try:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR)
                except ImportError:
                    pass
                span.record_exception(exc)
                yield _error_sse_frame(str(exc))
            else:
                _set_usage_on_span(span, usage_data, route)
                # Emit accumulated streaming output for langfuse.
                if content_acc:
                    try:
                        slim: dict[str, Any] = {}
                        if content_acc.get("text"):
                            slim["content"] = [
                                {"type": "text", "text": content_acc["text"]},
                            ]
                        if content_acc.get("thinking"):
                            slim["thinking"] = content_acc["thinking"]
                        if content_acc.get("stop_reason"):
                            slim["stop_reason"] = content_acc["stop_reason"]
                        if slim:
                            span.set_attribute(
                                "langfuse.observation.output",
                                _truncate_for_attr(
                                    json.dumps(slim, ensure_ascii=False, default=str),
                                ),
                            )
                    except Exception:
                        pass

    # ── token counting ────────────────────────────────────────────

    def _count_tokens(self, payload: dict[str, Any]) -> int:
        litellm = _load_litellm()
        counter = getattr(litellm, "token_counter", None)
        if not callable(counter):
            return 0

        route = self._resolve_route(payload.get("model"))
        messages = list(payload.get("messages") or [])
        system = payload.get("system")
        if system:
            text = self._flatten_system(system)
            if text:
                messages = [{"role": "system", "content": text}, *messages]
        try:
            return int(counter(model=route.transport.model, messages=messages))
        except Exception:
            return 0

    @staticmethod
    def _flatten_system(system: Any) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            parts = []
            for item in system:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p)
        return ""


# ── public entry point ───────────────────────────────────────────


async def prepare_model_endpoint(
    model: ModelConfig,
    extra_models: dict[str, ModelConfig] | None = None,
) -> PreparedModelEndpoint:
    """Resolve the SDK-facing endpoint for this run.

    Parameters
    ----------
    model:
        The main agent's :class:`ModelConfig`.
    extra_models:
        Optional mapping of ``model_name → ModelConfig`` for sub-agents
        whose models differ from the main agent.  When provided and at
        least one entry requires a non-Anthropic transport, a multi-model
        bridge is started so that every model (including Anthropic ones)
        can be routed through a single local HTTP endpoint.
    """
    transport = resolve_model_transport(model)

    # Determine whether *any* model (main or sub) needs a bridge.
    need_bridge = transport.protocol != "anthropic"
    if not need_bridge and extra_models:
        for cfg in extra_models.values():
            if resolve_model_transport(cfg).protocol != "anthropic":
                need_bridge = True
                break

    if not need_bridge:
        return PreparedModelEndpoint(
            model=transport.model,
            api_key=model.api_key,
            base_url=transport.api_base,
        )

    bridge = await LiteLLMAnthropicBridge(model, extra_models).start()
    return PreparedModelEndpoint(
        model=model.model,
        api_key="cckit-bridge",
        base_url=bridge.base_url,
        bridge=bridge,
    )
