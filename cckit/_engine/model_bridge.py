"""Internal model endpoint preparation for Claude SDK execution.

``cckit`` exposes a single user-facing model type: ``ModelConfig``.
All model strings, API bases, and credentials are interpreted with LiteLLM
semantics. Non-Anthropic protocols are bridged through a temporary local
Anthropic-compatible server; Anthropic protocol models are passed directly
to the Claude SDK.
"""

from __future__ import annotations

import asyncio
import importlib
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
from cckit._engine.sse_converter import (
    _jsonable,
    sse_frame,
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
                if payload.get("stream"):
                    stream = await self._handle_streaming(litellm, payload)
                    return StreamingResponse(
                        self._wrap_stream(stream),
                        media_type="text/event-stream",
                        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                    )
                kwargs = self._build_kwargs(payload)
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

    def _build_extra_kwargs(self) -> dict[str, Any]:
        """Build provider credentials dict for ``_prepare_completion_kwargs``."""
        extra: dict[str, Any] = {
            "custom_llm_provider": self._transport.custom_llm_provider,
            "timeout": self._model.timeout_seconds,
        }
        if self._model.api_key:
            extra["api_key"] = self._model.api_key
        if self._transport.api_base:
            extra["api_base"] = self._transport.api_base
        return extra

    # ── streaming ─────────────────────────────────────────────────

    async def _handle_streaming(
        self, litellm_mod: Any, payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Route streaming to the correct backend based on protocol.

        * **responses** – delegates to ``litellm.anthropic.messages.acreate``
          which internally routes ``openai`` provider to the Responses API.
        * **chat** – uses ``litellm.acompletion`` with our own SSE converter
          to work around LiteLLM's tool_call argument loss bug.
        """
        if self._transport.protocol == "responses":
            kwargs = self._build_kwargs(payload)
            kwargs["stream"] = True
            return await litellm_mod.anthropic.messages.acreate(**kwargs)

        return await self._stream_via_chat_completions(litellm_mod, payload)

    async def _stream_via_chat_completions(
        self, litellm_mod: Any, payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream via ``litellm.acompletion`` with custom SSE conversion.

        Uses our own OpenAI→Anthropic SSE converter instead of LiteLLM's
        ``AnthropicStreamWrapper`` to avoid its trigger-chunk argument
        loss bug (see ``sse_converter.py`` module docstring for details).
        """
        try:
            from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
                LiteLLMMessagesToCompletionTransformationHandler as _Handler,
            )
        except ImportError:
            logger.warning("LiteLLM adapter API unavailable; falling back to acreate")
            kwargs = self._build_kwargs(payload)
            kwargs["stream"] = True
            return await litellm_mod.anthropic.messages.acreate(**kwargs)

        from cckit._engine.model_transport import _coerce_positive_int

        max_tokens = payload.get("max_tokens", self._model.max_tokens)
        limit = _coerce_positive_int(self._model.max_tokens)
        requested = _coerce_positive_int(max_tokens)
        if limit and requested and requested > limit:
            max_tokens = limit

        completion_kwargs, tool_name_mapping = _Handler._prepare_completion_kwargs(
            max_tokens=max_tokens,
            messages=payload["messages"],
            model=self._transport.model,
            stream=True,
            system=payload.get("system"),
            temperature=payload.get("temperature"),
            tools=payload.get("tools"),
            tool_choice=payload.get("tool_choice"),
            thinking=payload.get("thinking"),
            top_k=payload.get("top_k"),
            top_p=payload.get("top_p"),
            metadata=payload.get("metadata"),
            stop_sequences=payload.get("stop_sequences"),
            output_format=payload.get("output_format"),
            extra_kwargs=self._build_extra_kwargs(),
        )

        openai_stream = await litellm_mod.acompletion(**completion_kwargs)

        from cckit._engine.sse_converter import openai_stream_to_anthropic_sse
        return openai_stream_to_anthropic_sse(
            openai_stream,
            model_name=self._model.model,
            tool_name_mapping=tool_name_mapping,
        )

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
                    yield sse_frame(str(data.get("type", "message_delta")), data)
        except Exception as exc:
            yield sse_frame("error", {
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