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
import json
import logging
import socket
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from cckit.exceptions import AgentExecutionError
from cckit.types import ModelConfig, TransportProtocol

logger = logging.getLogger(__name__)

_CHAT_DROPPED_ANTHROPIC_FIELDS = frozenset(
    {
        "context_management",
        "output_config",
    }
)


def _coerce_positive_int(value: Any) -> int | None:
    """Best-effort coercion for positive integer request fields."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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


def _load_litellm() -> Any:
    """Import LiteLLM lazily so the dependency is only touched at runtime."""
    try:
        return importlib.import_module("litellm")
    except ImportError as exc:
        raise AgentExecutionError(
            "cckit model execution requires LiteLLM bridge dependencies",
            detail=(
                "Install `litellm`, `starlette`, and `uvicorn`, then configure "
                "the model with LiteLLM-style provider semantics."
            ),
        ) from exc


@dataclass(slots=True)
class PreparedModelEndpoint:
    """Resolved model settings that the Claude SDK should see."""

    model: str
    api_key: str
    base_url: str
    bridge: LiteLLMAnthropicBridge | None = None

    async def aclose(self) -> None:
        """Stop the temporary bridge, if any."""
        if self.bridge is not None:
            await self.bridge.aclose()


@dataclass(slots=True)
class ResolvedTransport:
    """Concrete transport settings used for a single LiteLLM request."""

    protocol: TransportProtocol
    custom_llm_provider: str
    model: str
    api_base: str


def _split_model_prefix(model: str) -> tuple[str | None, str]:
    """Split the first provider prefix from a LiteLLM model string."""
    if "/" not in model:
        return None, model
    prefix, bare_model = model.split("/", 1)
    if not prefix or not bare_model:
        return None, model
    return prefix, bare_model


def resolve_model_transport(model: ModelConfig) -> ResolvedTransport:
    """Resolve the outgoing protocol from model prefix and full-path base URL hints."""
    prefix, bare_model = _split_model_prefix(model.model)

    if prefix == "openai":
        return ResolvedTransport(
            protocol="responses",
            custom_llm_provider="openai",
            model=bare_model,
            api_base=model.base_url,
        )
    if prefix == "anthropic":
        return ResolvedTransport(
            protocol="anthropic",
            custom_llm_provider="anthropic",
            model=bare_model,
            api_base=model.base_url,
        )
    if prefix:
        return ResolvedTransport(
            protocol="chat",
            custom_llm_provider=prefix,
            model=bare_model,
            api_base=model.base_url,
        )

    if model.endpoint_protocol == "responses":
        return ResolvedTransport(
            protocol="responses",
            custom_llm_provider="openai",
            model=model.model,
            api_base=model.base_url,
        )
    if model.endpoint_protocol == "anthropic":
        return ResolvedTransport(
            protocol="anthropic",
            custom_llm_provider="anthropic",
            model=model.model,
            api_base=model.base_url,
        )

    return ResolvedTransport(
        protocol="chat",
        custom_llm_provider="custom_openai",
        model=model.model,
        api_base=model.base_url,
    )


def _sanitize_payload_for_transport(
    payload: dict[str, Any],
    transport: ResolvedTransport,
) -> dict[str, Any]:
    """Drop protocol-incompatible Anthropic params before calling LiteLLM."""
    kwargs = dict(payload)
    if transport.protocol != "chat":
        return kwargs

    dropped = sorted(
        key for key in _CHAT_DROPPED_ANTHROPIC_FIELDS if key in kwargs
    )
    for key in dropped:
        kwargs.pop(key, None)

    if dropped:
        logger.debug(
            "Dropping Anthropic-only params for chat/completions transport: %s",
            ", ".join(dropped),
        )
    return kwargs


def _apply_max_tokens_policy(
    payload: dict[str, Any],
    transport: ResolvedTransport,
    configured_max_tokens: int,
) -> dict[str, Any]:
    """Clamp Claude SDK defaults to the transport-safe model max_tokens."""
    kwargs = dict(payload)
    if transport.protocol != "chat":
        return kwargs

    configured = _coerce_positive_int(configured_max_tokens)
    if configured is None:
        return kwargs

    requested = _coerce_positive_int(kwargs.get("max_tokens"))
    if requested is None:
        kwargs["max_tokens"] = configured
        return kwargs
    if requested > configured:
        logger.debug(
            "Clamping max_tokens for chat/completions transport: requested=%s configured=%s",
            requested,
            configured,
        )
        kwargs["max_tokens"] = configured
    return kwargs


class LiteLLMAnthropicBridge:
    """Temporary local Anthropic-compatible HTTP bridge backed by LiteLLM."""

    def __init__(self, model: ModelConfig) -> None:
        self._model = model
        self._transport = resolve_model_transport(model)
        self._server: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self.base_url: str = ""

    async def start(self) -> LiteLLMAnthropicBridge:
        """Boot the local bridge server and wait until it is ready."""
        app = self._build_app()

        uvicorn = self._load_module("uvicorn")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        self._socket = sock

        port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))

        for _ in range(100):
            if getattr(self._server, "started", False):
                return self
            if self._task.done():
                error = self._task.exception()
                detail = (
                    str(error) if error is not None else "Bridge server exited unexpectedly."
                )
                exc = AgentExecutionError(
                    "Failed to start the LiteLLM Anthropic bridge",
                    detail=detail,
                )
                if error is not None:
                    raise exc from error
                raise exc
            await asyncio.sleep(0.05)

        await self.aclose()
        raise AgentExecutionError(
            "Timed out waiting for the LiteLLM Anthropic bridge to start",
            detail="The temporary local HTTP server did not become ready within 5 seconds.",
        )

    async def aclose(self) -> None:
        """Shut down the temporary bridge server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with suppress(Exception):
                await self._task
        if self._socket is not None:
            self._socket.close()

        self._task = None
        self._server = None
        self._socket = None
        self.base_url = ""

    def _build_app(self) -> Any:
        """Create a minimal Anthropic-compatible ASGI app."""
        starlette_app = self._load_module("starlette.applications")
        starlette_responses = self._load_module("starlette.responses")
        starlette_routing = self._load_module("starlette.routing")
        litellm = _load_litellm()

        # Anthropic-compatibility is the entire point of this bridge: be
        # permissive with provider-specific unsupported params by default.
        litellm.drop_params = True

        json_response = starlette_responses.JSONResponse
        response_cls = starlette_responses.Response
        streaming_response = starlette_responses.StreamingResponse
        route = starlette_routing.Route
        starlette = starlette_app.Starlette

        async def health(_: Any) -> Any:
            return json_response({"ok": True})

        async def create_message(request: Any) -> Any:
            payload = await request.json()
            kwargs = self._build_litellm_kwargs(payload)

            try:
                if bool(payload.get("stream")):
                    stream = await self._call_litellm_messages(
                        litellm,
                        kwargs,
                    )
                    return streaming_response(
                        self._stream_events(stream),
                        media_type="text/event-stream",
                        headers={
                            "cache-control": "no-cache",
                            "x-accel-buffering": "no",
                        },
                    )

                response = await self._call_litellm_messages(
                    litellm,
                    kwargs,
                )
                return json_response(_jsonable(response))
            except Exception as exc:
                return json_response(
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": str(exc),
                        },
                    },
                    status_code=500,
                )

        async def count_tokens(request: Any) -> Any:
            payload = await request.json()
            response = {"input_tokens": self._count_tokens(payload)}
            return json_response(response)

        async def not_found(_: Any) -> Any:
            return response_cls(status_code=404)

        return starlette(
            routes=[
                route("/health", health, methods=["GET"]),
                route("/v1/messages", create_message, methods=["POST"]),
                route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
                route("/{path:path}", not_found, methods=["GET", "POST"]),
            ]
        )

    async def _stream_events(self, stream: Any) -> Any:
        """Convert LiteLLM streaming chunks into Anthropic-style SSE frames."""
        try:
            async for chunk in stream:
                payload = _jsonable(chunk)
                if isinstance(payload, (bytes, bytearray)):
                    yield bytes(payload)
                    continue
                if isinstance(payload, str):
                    yield payload.encode("utf-8")
                    continue
                event_type = str(payload.get("type", "message_delta"))
                yield _sse_frame(event_type, payload)
        except Exception as exc:
            yield _sse_frame(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": str(exc),
                    },
                },
            )

    def _build_litellm_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge Anthropic request payload with provider credentials."""
        kwargs = _sanitize_payload_for_transport(payload, self._transport)
        kwargs = _apply_max_tokens_policy(
            kwargs,
            self._transport,
            self._model.max_tokens,
        )
        kwargs["model"] = self._transport.model
        kwargs["custom_llm_provider"] = self._transport.custom_llm_provider
        kwargs.setdefault("max_tokens", self._model.max_tokens)

        if self._model.api_key:
            kwargs["api_key"] = self._model.api_key
        if self._transport.api_base:
            kwargs["api_base"] = self._transport.api_base

        kwargs["timeout"] = self._model.timeout_seconds
        return kwargs

    async def _call_litellm_messages(
        self,
        litellm: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Call LiteLLM's Anthropic pass-through endpoint."""
        return await litellm.anthropic.messages.acreate(**kwargs)

    def _count_tokens(self, payload: dict[str, Any]) -> int:
        """Best-effort token counting for Anthropic-compatible callers."""
        litellm = _load_litellm()
        counter = getattr(litellm, "token_counter", None)
        if not callable(counter):
            return 0

        messages = list(payload.get("messages") or [])
        system_prompt = payload.get("system")
        if system_prompt:
            system_text = self._flatten_system_prompt(system_prompt)
            if system_text:
                messages = [{"role": "system", "content": system_text}, *messages]

        try:
            return int(counter(model=self._model.model, messages=messages))
        except Exception:
            return 0

    @staticmethod
    def _flatten_system_prompt(system_prompt: Any) -> str:
        """Normalize Anthropic system prompt structures into plain text."""
        if isinstance(system_prompt, str):
            return system_prompt
        if isinstance(system_prompt, list):
            parts: list[str] = []
            for item in system_prompt:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part)
        return ""

    @staticmethod
    def _load_module(name: str) -> Any:
        """Import an optional runtime dependency with a good error."""
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise AgentExecutionError(
                "cckit model execution requires bridge runtime dependencies",
                detail=f"Missing import: {name}",
            ) from exc


async def prepare_model_endpoint(model: ModelConfig) -> PreparedModelEndpoint:
    """Resolve the SDK-facing endpoint for this run."""
    transport = resolve_model_transport(model)
    if transport.protocol == "anthropic":
        return PreparedModelEndpoint(
            model=transport.model,
            api_key=model.api_key,
            base_url=transport.api_base,
            bridge=None,
        )

    bridge = await LiteLLMAnthropicBridge(model).start()
    return PreparedModelEndpoint(
        model=model.model,
        api_key="cckit-bridge",
        base_url=bridge.base_url,
        bridge=bridge,
    )
