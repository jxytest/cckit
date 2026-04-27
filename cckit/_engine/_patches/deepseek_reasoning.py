"""DeepSeek V4 ``reasoning_content`` fix for multi-turn conversations.

DeepSeek V4 models (v4-pro, v4-flash) enable thinking mode by default and
**require** ``reasoning_content`` on every assistant message in the conversation
history.  LiteLLM's ``Message.__init__`` strips the field when it is ``None``,
causing the second turn onward to fail with::

    The `reasoning_content` in the thinking mode must be passed back to the API.

This patch injects a non-empty ``reasoning_content`` placeholder into assistant
messages before the request is forwarded to the provider, without modifying
LiteLLM internals.

See: https://github.com/BerriAI/litellm/issues/26395
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEEPSEEK_V4_RE = re.compile(r"deepseek-v4", re.IGNORECASE)
# Some DeepSeek V4 gateways treat an empty string the same as a missing
# reasoning_content field.  Claude CLI / Anthropic history does not preserve the
# provider's original OpenAI-specific reasoning_content, so use the conventional
# non-empty marker returned/displayed by many gateways instead of "".
_PLACEHOLDER_REASONING_CONTENT = "已深度思考"


def _is_deepseek_v4(model: str) -> bool:
    """Return True only for DeepSeek V4 series (v4-pro, v4-flash, etc.)."""
    return _DEEPSEEK_V4_RE.search(model) is not None


def patch_deepseek_reasoning(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Ensure ``reasoning_content`` is present on assistant messages for DeepSeek V4.

    Only applies to DeepSeek V4 models (v4-pro, v4-flash) which require
    ``reasoning_content`` in thinking mode.  Older models (R1, v3.2, etc.)
    are left untouched — R1 actively rejects the field.

    This is a **request-level** transform applied in the bridge's ``_build_kwargs``
    pipeline, right after ``sanitize_payload`` / ``clamp_max_tokens``.
    """
    if not _is_deepseek_v4(model):
        return payload

    messages = payload.get("messages")
    if not messages:
        return payload

    patched = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if not msg.get("reasoning_content"):
            msg["reasoning_content"] = _PLACEHOLDER_REASONING_CONTENT
            patched += 1

    if patched:
        logger.debug("Injected reasoning_content on %d assistant message(s)", patched)

    return payload
