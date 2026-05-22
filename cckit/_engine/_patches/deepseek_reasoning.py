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


def _sanitize_text_blocks_in_messages(messages: Any) -> int:
    """Repair text content blocks missing or with null ``text`` field.

    Strict upstream gateways (e.g. NetEase's Rust/serde-based gateway in
    front of DeepSeek V4) reject requests when a block declares
    ``"type": "text"`` but the ``text`` field is missing or ``None``::

        Failed to deserialize the JSON body into the target type:
        messages[N]: missing field `text`

    The malformation can hide in any of these places:

    * ``message.content[i]`` — top-level user/assistant text blocks
    * ``message.content[i].content[j]`` — text blocks **nested inside
      tool_result blocks**. LiteLLM's translator reads
      ``c.get("text", "")`` which returns ``None`` when the upstream
      message had an explicit ``"text": null`` (the default only kicks
      in for missing keys, not null values), so the malformed block
      survives translation and reaches the wire.
    * ``system[i]`` — Anthropic system prompt as a list of text blocks
    * ``message.tool_calls[i].function.arguments`` is JSON, not text;
      we don't touch it.

    We backfill missing/null ``text`` with an empty string and coerce
    non-string values to ``str`` for safety. Empty text blocks are
    valid in OpenAI/Anthropic schemas, just unwelcome to the strictest
    deserializers.

    Returns the number of blocks repaired (for debug logging).
    """
    fixed = 0

    def _walk(blocks: Any) -> None:
        nonlocal fixed
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if text is None:
                    block["text"] = ""
                    fixed += 1
                elif not isinstance(text, str):
                    block["text"] = str(text)
                    fixed += 1
            # Recurse into tool_result blocks — their nested content
            # array can contain its own text blocks that the translator
            # passes through with ``c.get("text", "")`` (which returns
            # None when text is explicitly null in the input).
            inner = block.get("content")
            if isinstance(inner, list):
                _walk(inner)

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            _walk(msg.get("content"))

    return fixed


def _sanitize_text_blocks_in_system(system: Any) -> int:
    """Repair the Anthropic ``system`` field when it's a list of text blocks."""
    if not isinstance(system, list):
        return 0
    fixed = 0
    for block in system:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if text is None:
            block["text"] = ""
            fixed += 1
        elif not isinstance(text, str):
            block["text"] = str(text)
            fixed += 1
    return fixed


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

    Also runs an unconditional content-block sanitizer that repairs
    ``{"type": "text"}`` blocks missing the ``text`` field — strict
    gateways (NetEase Rust/serde) reject these even though the upstream
    OpenAI/Anthropic schemas tolerate them.
    """
    messages = payload.get("messages")
    try:
        repaired = _sanitize_text_blocks_in_messages(messages)
        repaired += _sanitize_text_blocks_in_system(payload.get("system"))
        if repaired:
            logger.debug("Sanitized %d text content block(s) missing `text`", repaired)
    except Exception:
        logger.debug("Pre-translation sanitize raised", exc_info=True)

    if not _is_deepseek_v4(model):
        return payload

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

        # Phase 2c: repair text blocks the translator produced without a
        # ``text`` field. Strict serde-based gateways (NetEase) reject
        # these. Operates on every model — the bug is independent of
        # whether we're talking to DeepSeek.
        # Wrapped in try/except: this is a best-effort defensive patch on
        # the LLM critical path, must never break translation itself.
        try:
            repaired = _sanitize_text_blocks_in_messages(result)
            if repaired:
                logger.debug(
                    "Post-translation sanitize fixed %d text block(s)", repaired
                )
        except Exception:
            logger.debug("Post-translation sanitize raised", exc_info=True)

        return result

    LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai = (
        _patched_translate
    )

    # Phase 2b: patch _translate_thinking_to_openai so that DeepSeek V4's
    # ``thinking: {"type": "disabled"}`` is carried through as-is instead
    # of being silently consumed (the default adapter converts "disabled"
    # to None which means nothing reaches the DeepSeek API).
    _original_thinking = LiteLLMAnthropicMessagesAdapter._translate_thinking_to_openai

    def _patched_translate_thinking(
        self: Any,
        anthropic_message_request: Any,
        new_kwargs: dict,
    ) -> None:
        thinking = anthropic_message_request.get("thinking")
        model = new_kwargs.get("model", "")
        if (
            thinking
            and isinstance(thinking, dict)
            and thinking.get("type") == "disabled"
            and _is_deepseek_v4(model)
        ):
            new_kwargs["thinking"] = thinking
            return
        return _original_thinking(self, anthropic_message_request, new_kwargs)

    LiteLLMAnthropicMessagesAdapter._translate_thinking_to_openai = (
        _patched_translate_thinking
    )

    _PATCHED = True
    logger.debug("Applied DeepSeek reasoning_content and thinking adapter patches")
