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

from cckit._engine.model_transport import (
    ResolvedTransport,  # noqa: F401 – used internally
    clamp_max_tokens,
    resolve_model_transport,  # noqa: F401 – re-exported for backward compat
    sanitize_payload,
)
from cckit.exceptions import AgentExecutionError
from cckit.types import ModelConfig

logger = logging.getLogger(__name__)


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

        # Apply the streaming bug-fix patch before first use.
        from cckit._engine._stream_patch import apply_stream_patch
        apply_stream_patch()

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
                if payload.get("stream"):
                    kwargs["stream"] = True
                    stream = await litellm.anthropic.messages.acreate(**kwargs)
                    return StreamingResponse(
                        self._wrap_stream(stream),
                        media_type="text/event-stream",
                        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                    )
                resp = await litellm.anthropic.messages.acreate(**kwargs)
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
        kwargs["model"] = transport.model
        kwargs["custom_llm_provider"] = transport.custom_llm_provider
        kwargs.setdefault("max_tokens", cfg.max_tokens)
        kwargs["timeout"] = cfg.timeout_seconds
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if transport.api_base:
            kwargs["api_base"] = transport.api_base
        return kwargs

    # ── streaming ─────────────────────────────────────────────────

    async def _wrap_stream(self, stream: Any) -> AsyncIterator[bytes]:
        """Pass through SSE bytes from litellm, with error boundary.

        ``litellm.anthropic.messages.acreate(stream=True)`` returns an
        ``AsyncIterator[bytes]`` (via ``async_anthropic_sse_wrapper``),
        so chunks are already SSE-encoded.  We just forward them and
        catch any mid-stream exceptions as an SSE error event.
        """
        try:
            async for chunk in stream:
                if isinstance(chunk, (bytes, bytearray)):
                    yield bytes(chunk)
                elif isinstance(chunk, str):
                    yield chunk.encode()
                else:
                    # Shouldn't happen with current litellm, but be safe.
                    yield json.dumps(chunk).encode()
        except Exception as exc:
            yield _error_sse_frame(str(exc))

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
