"""DeepSeek V4 ``reasoning_content`` fix for multi-turn conversations.

DeepSeek V4 models (v4-pro, v4-flash) enable thinking mode by default and
**require** ``reasoning_content`` on every assistant message in the conversation
history.  LiteLLM's ``Message.__init__`` strips the field when it is ``None``,
causing the second turn onward to fail with::

    The `reasoning_content` in the thinking mode must be passed back to the API.

**Two-phase fix:**

1. ``patch_deepseek_reasoning()`` — injects ``reasoning_content`` on assistant
   messages in the Anthropic-format payload *before* LiteLLM processes it.
2. ``apply_deepseek_reasoning_patch()`` — monkey-patches LiteLLM's Anthropic
   adapter so that ``reasoning_content`` survives the Anthropic → OpenAI
   message translation (``translate_anthropic_messages_to_openai`` creates new
   dicts and drops unknown fields).

Both phases are required: phase 1 sets the value, phase 2 carries it through.

See: https://github.com/BerriAI/litellm/issues/26395
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEEPSEEK_V4_RE = re.compile(r"deepseek-v4", re.IGNORECASE)
_PLACEHOLDER_REASONING_CONTENT = "已深度思考"

_PATCHED = False


def _is_deepseek_v4(model: str) -> bool:
    """Return True only for DeepSeek V4 series (v4-pro, v4-flash, etc.)."""
    return _DEEPSEEK_V4_RE.search(model) is not None


def _extract_reasoning_from_content(content: Any) -> str | None:
    """Extract reasoning text from Anthropic ``thinking`` content blocks.

    When DeepSeek returns ``reasoning_content``, the response adapter converts
    it to a ``{type: "thinking", thinking: "..."}`` block.  If the SDK echoes
    this block back on the next turn, we can recover the original reasoning
    instead of using a meaningless placeholder.
    """
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            text = block.get("thinking")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else None


def patch_deepseek_reasoning(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Phase 1: inject ``reasoning_content`` on Anthropic-format assistant messages.

    Only applies to DeepSeek V4 models (v4-pro, v4-flash) which require
    ``reasoning_content`` in thinking mode.  Older models (R1, v3.2, etc.)
    are left untouched — R1 actively rejects the field.

    Tries to recover real reasoning from ``thinking`` content blocks first;
    falls back to a non-empty placeholder only when nothing is available.

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
        if msg.get("reasoning_content"):
            continue
        recovered = _extract_reasoning_from_content(msg.get("content"))
        msg["reasoning_content"] = recovered or _PLACEHOLDER_REASONING_CONTENT
        patched += 1

    if patched:
        logger.debug("Injected reasoning_content on %d assistant message(s)", patched)

    return payload


def apply_deepseek_reasoning_patch() -> None:
    """Phase 2: monkey-patch the Anthropic adapter to preserve ``reasoning_content``.

    LiteLLM's ``translate_anthropic_messages_to_openai`` creates fresh
    ``ChatCompletionAssistantMessage`` dicts and never copies the
    ``reasoning_content`` field from the source message — even though the
    TypedDict supports it.  This patch wraps the method so that after
    translation, ``reasoning_content`` is carried over from the original
    Anthropic-format messages to the translated OpenAI-format messages.

    Idempotent and safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError:
        logger.debug("LiteLLM adapter not available; skipping reasoning patch")
        return

    _original = LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai

    def _patched_translate(
        self: Any, messages: list, model: str | None = None
    ) -> list:
        rc_by_idx: dict[int, str] = {}
        assistant_idx = 0
        for m in messages:
            if m.get("role") == "assistant":
                rc = m.get("reasoning_content")
                if rc is not None:
                    rc_by_idx[assistant_idx] = rc
                assistant_idx += 1

        result = _original(self, messages, model=model)

        assistant_idx = 0
        for msg in result:
            if msg.get("role") != "assistant":
                continue
            if assistant_idx in rc_by_idx:
                msg["reasoning_content"] = rc_by_idx[assistant_idx]
            elif (
                model
                and _is_deepseek_v4(model)
                and "reasoning_content" not in msg
            ):
                tbs = msg.get("thinking_blocks")
                if tbs:
                    parts = [
                        tb.get("thinking", "")
                        for tb in tbs
                        if tb.get("type") == "thinking" and tb.get("thinking")
                    ]
                    if parts:
                        msg["reasoning_content"] = "\n".join(parts)
            assistant_idx += 1

        return result

    LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai = (
        _patched_translate
    )
    _PATCHED = True
    logger.debug("Applied DeepSeek reasoning_content adapter patch")
