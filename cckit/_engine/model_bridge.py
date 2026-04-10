"""Temporary local Anthropic-compatible HTTP bridge backed by LiteLLM.

The bridge starts a lightweight Starlette server on ``127.0.0.1:<random-port>``
that accepts Anthropic ``/v1/messages`` requests and forwards them to the
real provider through LiteLLM.  Anthropic-protocol models bypass the bridge
entirely.
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


def _jsonable(value: Any) -> Any:
    """Convert LiteLLM / Pydantic payloads to plain JSON-compatible objects."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        raw = value.json()
        return json.loads(raw) if isinstance(raw, str) else raw
    return value


def _sse_frame(event_type: str, payload: Any) -> bytes:
    """Encode a single Anthropic-style SSE frame."""
    body = json.dumps(_jsonable(payload), ensure_ascii=True, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n".encode()


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


class LiteLLMAnthropicBridge:
    """Temporary local Anthropic-compatible HTTP bridge backed by LiteLLM."""

    def __init__(self, model: ModelConfig) -> None:
        self._model = model
        self._transport = resolve_model_transport(model)
        self._server: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self.base_url: str = ""

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> LiteLLMAnthropicBridge:
        """Boot the local bridge server and wait until it is ready."""
        logger.debug(
            "Starting bridge: protocol=%s provider=%s model=%s api_base=%s",
            self._transport.protocol,
            self._transport.custom_llm_provider,
            self._transport.model,
            self._transport.api_base,
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
                return JSONResponse(_jsonable(resp))
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

    # ── request building ─────────────────────────────────────────

    def _build_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build kwargs for ``litellm.anthropic.messages.acreate``."""
        kwargs = sanitize_payload(payload, self._transport)
        kwargs = clamp_max_tokens(kwargs, self._transport, self._model.max_tokens)
        kwargs["model"] = self._transport.model
        kwargs["custom_llm_provider"] = self._transport.custom_llm_provider
        kwargs.setdefault("max_tokens", self._model.max_tokens)
        kwargs["timeout"] = self._model.timeout_seconds
        if self._model.api_key:
            kwargs["api_key"] = self._model.api_key
        if self._transport.api_base:
            kwargs["api_base"] = self._transport.api_base
        return kwargs

    # ── streaming ─────────────────────────────────────────────────

    async def _wrap_stream(self, stream: Any) -> AsyncIterator[bytes]:
        """Normalize heterogeneous stream chunks into SSE bytes."""
        try:
            async for chunk in stream:
                data = _jsonable(chunk)
                if isinstance(data, (bytes, bytearray)):
                    yield bytes(data)
                elif isinstance(data, str):
                    yield data.encode()
                else:
                    yield _sse_frame(str(data.get("type", "message_delta")), data)
        except Exception as exc:
            yield _sse_frame("error", {
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            })

    # ── token counting ────────────────────────────────────────────

    def _count_tokens(self, payload: dict[str, Any]) -> int:
        litellm = _load_litellm()
        counter = getattr(litellm, "token_counter", None)
        if not callable(counter):
            return 0

        messages = list(payload.get("messages") or [])
        system = payload.get("system")
        if system:
            text = self._flatten_system(system)
            if text:
                messages = [{"role": "system", "content": text}, *messages]
        try:
            return int(counter(model=self._transport.model, messages=messages))
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


async def prepare_model_endpoint(model: ModelConfig) -> PreparedModelEndpoint:
    """Resolve the SDK-facing endpoint for this run."""
    transport = resolve_model_transport(model)
    if transport.protocol == "anthropic":
        return PreparedModelEndpoint(
            model=transport.model,
            api_key=model.api_key,
            base_url=transport.api_base,
        )
    bridge = await LiteLLMAnthropicBridge(model).start()
    return PreparedModelEndpoint(
        model=model.model,
        api_key="cckit-bridge",
        base_url=bridge.base_url,
        bridge=bridge,
    )