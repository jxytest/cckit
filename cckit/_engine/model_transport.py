"""Model transport resolution — maps ``ModelConfig`` to concrete LiteLLM provider settings.

Routing rules
-------------
* ``responses/<model>`` – Responses API via ``openai`` provider; bare model passed.
* ``openai/<model>``    – Chat/completions via ``custom_openai`` provider (NOT Responses API).
* ``anthropic/<model>`` – Direct Anthropic protocol (no bridge).
* ``<provider>/<model>`` – Chat/completions via the named LiteLLM provider.
* No prefix             – ``base_url`` suffix detection or plain chat/completions fallback.

Provider semantics (LiteLLM internal routing)
----------------------------------------------
* ``"openai"``        → litellm routes to **Responses API** (``/responses``).
* ``"custom_openai"`` → litellm routes to **Chat Completions** (``/chat/completions``).

Use ``"openai"`` **only** when the ``responses`` protocol is explicitly requested.
All other OpenAI-compatible endpoints must use ``"custom_openai"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from cckit.types import ModelConfig, TransportProtocol

logger = logging.getLogger(__name__)

_CHAT_DROPPED_ANTHROPIC_FIELDS = frozenset({"context_management", "output_config"})


@dataclass(slots=True)
class ResolvedTransport:
    """Concrete transport settings used for a single LiteLLM request."""

    protocol: TransportProtocol
    custom_llm_provider: str
    model: str
    api_base: str


def _split_model_prefix(model: str) -> tuple[str | None, str]:
    """Split ``"prefix/bare_model"`` → ``("prefix", "bare_model")``."""
    if "/" not in model:
        return None, model
    prefix, bare = model.split("/", 1)
    return (prefix, bare) if prefix and bare else (None, model)


def resolve_model_transport(model: ModelConfig) -> ResolvedTransport:
    """Resolve :class:`ModelConfig` into concrete LiteLLM transport settings."""
    prefix, bare = _split_model_prefix(model.model)
    base = model.base_url or ""

    if prefix == "responses":
        return ResolvedTransport("responses", "openai", bare, base)

    if prefix == "openai":
        # "openai/" prefix → OpenAI-compatible chat/completions, NOT Responses API.
        # Use "custom_openai" to prevent litellm from routing to /responses.
        return ResolvedTransport("chat", "custom_openai", bare, base)

    if prefix == "anthropic":
        return ResolvedTransport("anthropic", "anthropic", bare, base)

    if prefix:
        return ResolvedTransport("chat", prefix, bare, base)

    # No prefix — use endpoint_protocol hint from base_url suffix detection.
    if model.endpoint_protocol == "responses":
        return ResolvedTransport("responses", "openai", model.model, base)
    if model.endpoint_protocol == "anthropic":
        return ResolvedTransport("anthropic", "anthropic", model.model, base)
    return ResolvedTransport("chat", "custom_openai", model.model, base)


# ── payload helpers ──────────────────────────────────────────────


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def sanitize_payload(
    payload: dict[str, Any],
    transport: ResolvedTransport,
) -> dict[str, Any]:
    """Drop Anthropic-only params that are incompatible with the transport protocol."""
    kwargs = dict(payload)
    if transport.protocol == "anthropic":
        return kwargs
    dropped = sorted(k for k in _CHAT_DROPPED_ANTHROPIC_FIELDS if k in kwargs)
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        logger.debug("Dropping Anthropic-only params for %s: %s", transport.protocol, ", ".join(dropped))
    return kwargs


def clamp_max_tokens(
    payload: dict[str, Any],
    transport: ResolvedTransport,
    configured: int,
) -> dict[str, Any]:
    """Clamp ``max_tokens`` to the configured model limit for non-Anthropic transports."""
    kwargs = dict(payload)
    if transport.protocol == "anthropic":
        return kwargs
    limit = _coerce_positive_int(configured)
    if limit is None:
        return kwargs
    requested = _coerce_positive_int(kwargs.get("max_tokens"))
    if requested is None:
        kwargs["max_tokens"] = limit
    elif requested > limit:
        logger.debug("Clamping max_tokens: %s → %s", requested, limit)
        kwargs["max_tokens"] = limit
    return kwargs
