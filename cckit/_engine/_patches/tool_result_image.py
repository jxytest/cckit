"""Tool-result image fix — keep single-image tool results visible as images.

LiteLLM's ``translate_anthropic_messages_to_openai`` flattens a
``tool_result`` whose content list holds exactly **one** ``image`` block into
a plain data-URI **string**::

    # transformation.py (single-item branch)
    tool_result = ChatCompletionToolMessage(
        role="tool",
        tool_call_id=...,
        content=openai_image_url,   # "data:image/png;base64,AAAA…"
    )

The multi-item branch a few lines below does the right thing and builds a
list of ``ChatCompletionImageObject`` parts.  So a tool that returns *one*
image silently degrades to text while the same tool returning *two* blocks
works — the inconsistency is the whole bug.

Downstream OpenAI-compatible providers then receive a very long string
instead of an image.  The model never sees the pixels: it answers from the
file name, hallucinates a description, or (as seen with Qwen behind the
NetEase gateway) reports that the tool "returned a URL instead of image
content" and tries to ``curl`` it.  Gateways that spill oversized bodies to
object storage make this especially confusing, because the model then
genuinely does see an OSS link.

This is the exact path taken by the Claude Code CLI's ``Read`` tool on a
local image file, which is how cckit's vision sub-agent looks at
screenshots.

Upstream status (checked 2026-08-13, still unfixed on 1.96.2)
------------------------------------------------------------
* https://github.com/BerriAI/litellm/issues/24968 — bug report, closed by
  the stale bot without a fix.
* https://github.com/BerriAI/litellm/pull/25476 — fix identical in spirit to
  this one (structure the single-item branch like the multi-item branch);
  CI green, closed by the stale bot.
* https://github.com/BerriAI/litellm/pull/34462 — broader fix (also moves
  images into a following ``user`` message and repairs the Responses
  adapter, which drops tool images entirely); still open against
  ``litellm_internal_staging``.

We deliberately mirror pull/25476's shape: normalise the single-image case
into the list form the multi-item branch already produces.  Should upstream
land either PR, this patch becomes a no-op and can be deleted outright.

Note this patch fixes the **chat/completions** adapter path only.  The
Responses-API path (``responses/`` prefixed models) drops tool images
upstream of this translation; see pull/34462.  cckit's vision agents run on
chat-completions models, so that gap is not currently reachable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def _normalize_tool_message(msg: Any) -> bool:
    """Rewrite one translated ``role:"tool"`` message carrying a bare data URI.

    Returns ``True`` when the message was rewritten.  Anything that is not a
    lone image data URI (plain text results, already-structured list content,
    http(s) URLs a provider can fetch itself) is left untouched.
    """
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return False

    content = msg.get("content")
    if not isinstance(content, str) or not content.startswith("data:image/"):
        return False
    if ";base64," not in content:
        return False

    msg["content"] = [{"type": "image_url", "image_url": {"url": content}}]
    return True


def apply_tool_result_image_patch() -> None:
    """Apply the tool-result image fix once.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError:
        logger.debug("LiteLLM adapter not available; skipping tool-result image patch")
        return

    _original = LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai

    def _patched_translate(self: Any, messages: list, model: str | None = None) -> list:
        result = _original(self, messages, model=model)
        # Best-effort: a defect here would break every request, so never let
        # the normalisation itself raise on the LLM critical path.
        try:
            fixed = sum(1 for m in result if _normalize_tool_message(m))
            if fixed:
                logger.debug(
                    "Restored %d tool_result image(s) to structured image_url", fixed,
                )
        except Exception:
            logger.debug("tool_result image normalisation raised", exc_info=True)
        return result

    LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai = (
        _patched_translate
    )
    _PATCHED = True
    logger.debug("Applied tool_result single-image fix")
