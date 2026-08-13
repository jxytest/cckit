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

We follow pull/34462's shape rather than pull/25476's: structuring the tool
message in place (``content=[{"type": "image_url", ...}]``) is enough for
some providers but **not** for others.  Measured against the NetEase gateway
on 2026-08-13 with an image whose watermark text is unguessable:

===========================================  ==========  =================
tool_result shape                            qwen3.8-max  claude-sonnet-4-6
===========================================  ==========  =================
bare data-URI string (the bug)               blind        blind
``[{"type": "image_url", …}]`` in ``tool``   **sees it**  blind
image moved to a following ``user`` message  **sees it**  **sees it**
===========================================  ==========  =================

Worse than blind, a model that gets the in-``tool`` form reports the tool as
having returned nothing — it goes on to retry ``Read``, call the vision
sub-agent again, and finally tell the user it cannot see images, even though
the sub-agent did return a full description.  Only the relocated form works
across providers, which is why the placeholder + user-message dance below is
not gratuitous.

Upstream status (checked 2026-08-13, still unfixed on 1.96.2)
------------------------------------------------------------
* https://github.com/BerriAI/litellm/issues/24968 — bug report, closed by
  the stale bot without a fix.
* https://github.com/BerriAI/litellm/pull/25476 — structures the single-item
  branch in place; CI green, closed by the stale bot.  Insufficient on its
  own, per the table above.
* https://github.com/BerriAI/litellm/pull/34462 — moves tool images into a
  following ``user`` message (and repairs the Responses adapter, which drops
  them entirely); still open against ``litellm_internal_staging``.  This is
  the behaviour reproduced here.

Should upstream land pull/34462, this patch becomes redundant and can be
deleted outright.

Note this fixes the **chat/completions** adapter path only.  The
Responses-API path (``responses/`` prefixed models) drops tool images
upstream of this translation; see pull/34462.  cckit's vision agents run on
chat-completions models, so that gap is not currently reachable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False

# What the tool message says once its image has been moved out. Some
# providers reject an empty tool result, and the model needs to know the
# call succeeded, so leave a human-readable breadcrumb rather than "".
_IMAGE_MOVED_PLACEHOLDER = "[Tool returned an image - see the following user message]"


def _extract_image_url(msg: Any) -> str | None:
    """Return the data URI of a ``role:"tool"`` message that is *only* an image.

    Handles both shapes LiteLLM can produce for a single-image tool_result:
    the bare data-URI string (the bug), and the structured single-part list.
    Anything else — plain text, http(s) URLs a provider can fetch itself,
    mixed text+image content — returns ``None`` and is left untouched.
    """
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return None

    content = msg.get("content")

    if isinstance(content, str):
        if content.startswith("data:image/") and ";base64," in content:
            return content
        return None

    # Structured but still buried inside the tool message: providers that
    # ignore images on role:"tool" need it moved out just the same.
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if isinstance(url, str) and url.startswith("data:image/"):
                return url

    return None


def _relocate_tool_result_images(messages: list) -> int:
    """Move tool-result images into a ``user`` message after the tool block.

    OpenAI's documented shape for image-producing tools is a ``user`` message
    carrying the ``image_url`` parts, immediately after the ``tool``
    message(s) that produced them.  Rewriting in place would break the
    tool_call ↔ tool_result pairing, so images are collected and re-emitted
    once the contiguous run of tool messages ends.

    Returns the number of images relocated.
    """
    out: list = []
    pending: list = []
    moved = 0

    def _flush() -> None:
        """Emit collected images as one user message once the tool run ends."""
        nonlocal pending
        if pending:
            out.append({"role": "user", "content": pending})
            pending = []

    for msg in messages:
        is_tool = isinstance(msg, dict) and msg.get("role") == "tool"
        # A non-tool message ends the run: flush before appending it, so the
        # images land directly after their tool messages.
        if not is_tool:
            _flush()
            out.append(msg)
            continue

        url = _extract_image_url(msg)
        if url is None:
            out.append(msg)
            continue

        msg["content"] = _IMAGE_MOVED_PLACEHOLDER
        out.append(msg)
        pending.append({"type": "image_url", "image_url": {"url": url}})
        moved += 1

    _flush()
    messages[:] = out
    return moved


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
        # the relocation itself raise on the LLM critical path.
        try:
            moved = _relocate_tool_result_images(result)
            if moved:
                logger.debug(
                    "Relocated %d tool_result image(s) into a following user message",
                    moved,
                )
        except Exception:
            logger.debug("tool_result image relocation raised", exc_info=True)
        return result

    LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai = (
        _patched_translate
    )
    _PATCHED = True
    logger.debug("Applied tool_result single-image fix")
