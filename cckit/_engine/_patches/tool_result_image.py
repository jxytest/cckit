"""Tool-result fixes — keep tool output visible to OpenAI-compatible models.

Two related defects in LiteLLM's ``translate_anthropic_messages_to_openai``,
both of which end with the model insisting a tool returned nothing.

Defect 1 — a single image block is flattened to text
----------------------------------------------------
A ``tool_result`` whose content list holds exactly **one** ``image`` block
becomes a plain data-URI **string**::

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

Defect 2 — list-shaped text results are dropped entirely
---------------------------------------------------------
A ``tool_result`` carrying **text** blocks is translated to a ``role:"tool"``
message whose ``content`` is a *list* of ``{"type": "text"}`` parts.  That is
legal OpenAI schema, but several providers only read a plain string there and
treat the list as an empty result.  Measured on the NetEase gateway
(2026-08-13), asking the model to echo a token returned by a tool:

===================================  ==========  =================
``role:"tool"`` content shape        qwen3.8-max  claude-sonnet-4-6
===================================  ==========  =================
``"TOKEN"``                          sees it      sees it
``[{"type":"text","text":"TOKEN"}]`` sees it      **blind**
two ``text`` parts                   sees it      **blind**
===================================  ==========  =================

This is what a cckit ``Task`` sub-agent returns: its description plus a
trailing ``agentId``/``<usage>`` part — two text blocks.  The sub-agent runs
fine and returns a full answer, the main agent reports "the sub-agent didn't
return output", retries, and eventually tells the user the feature is
broken.  Flattening pure-text parts into one string fixes it, and matches
what the single-text-block branch upstream already produces.

Only *pure text* lists are joined; anything carrying an image keeps its list
shape (it has to — see the relocation below).

Why images move to a ``user`` message
--------------------------------------
For defect 1 we follow pull/34462's shape rather than pull/25476's:
structuring the tool message in place (``content=[{"type": "image_url",
…}]``) is enough for some providers but **not** for others.  Same gateway,
same day, with an image whose watermark text is unguessable:

===========================================  ==========  =================
tool_result shape                            qwen3.8-max  claude-sonnet-4-6
===========================================  ==========  =================
bare data-URI string (the bug)               blind        blind
``[{"type": "image_url", …}]`` in ``tool``   **sees it**  blind
image moved to a following ``user`` message  **sees it**  **sees it**
===========================================  ==========  =================

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

Defect 2 has no upstream ticket that we could find; the list shape is valid
per the OpenAI schema, so it is arguably a provider-side limitation rather
than a LiteLLM bug.  Flattening is safe either way.

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


def _flatten_text_only_content(msg: Any) -> bool:
    """Join a ``role:"tool"`` message whose content is only text parts.

    Some providers (claude-sonnet-4-6 behind the NetEase gateway, measured
    2026-08-13) read ``role:"tool"`` content only when it is a plain string
    and report an empty tool result for the list form.  A cckit ``Task``
    sub-agent always hits this: it returns its answer plus a trailing
    ``agentId``/``<usage>`` part, i.e. two text blocks.

    Returns ``True`` when the message was flattened.  Lists holding anything
    other than text (notably images) are left alone.
    """
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return False

    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False

    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            return False
        text = part.get("text")
        texts.append(text if isinstance(text, str) else "")

    msg["content"] = "\n\n".join(texts)
    return True


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
        # these repairs raise on the LLM critical path.
        try:
            # Relocate first: it replaces an image tool message's content with
            # a plain string, so nothing is left for the flattener to touch.
            moved = _relocate_tool_result_images(result)
            if moved:
                logger.debug(
                    "Relocated %d tool_result image(s) into a following user message",
                    moved,
                )
            flattened = sum(1 for m in result if _flatten_text_only_content(m))
            if flattened:
                logger.debug(
                    "Flattened %d text-only tool_result(s) to a plain string", flattened,
                )
        except Exception:
            logger.debug("tool_result repair raised", exc_info=True)
        return result

    LiteLLMAnthropicMessagesAdapter.translate_anthropic_messages_to_openai = (
        _patched_translate
    )
    _PATCHED = True
    logger.debug("Applied tool_result single-image fix")
