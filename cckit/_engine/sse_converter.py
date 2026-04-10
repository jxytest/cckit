"""OpenAI (chat/completions) → Anthropic SSE stream converter.

Converts an ``litellm.acompletion`` streaming response into Anthropic-format
Server-Sent Events bytes.

Why not use LiteLLM's built-in ``AnthropicStreamWrapper``?
-----------------------------------------------------------
``AnthropicStreamWrapper.__anext__`` (streaming_iterator.py:306-332) has a
confirmed bug: when a content-block type changes (e.g. text → tool_use), the
trigger chunk's ``processed_chunk`` — which carries the ``input_json_delta``
— is **intentionally discarded** with the comment "The trigger chunk itself
is not emitted as a delta since the content_block_start already carries the
relevant information."  However, ``content_block_start.input`` is always
``{}`` (empty), so the actual tool arguments are lost.

This only manifests when a provider sends tool_call ``id + name + arguments``
in a **single chunk** (non-incremental), which is common with Chinese LLM
gateways, self-hosted proxies, and some aggregation APIs.  Standard OpenAI
streams arguments incrementally and is not affected.

This module is used **only** for the ``chat`` protocol path (third-party
chat/completions providers).  The ``responses`` protocol path delegates
entirely to ``litellm.anthropic.messages.acreate``.
"""

from __future__ import annotations

import json
import uuid as _uuid
from collections.abc import AsyncIterator
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert LiteLLM / Pydantic objects to JSON-compatible dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "json"):
        raw = value.json()
        return json.loads(raw) if isinstance(raw, str) else raw
    return value


def sse_frame(event_type: str, payload: Any) -> bytes:
    """Encode a single Anthropic-style SSE frame."""
    body = json.dumps(_jsonable(payload), ensure_ascii=True, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n".encode()


def extract_usage(usage_obj: Any) -> dict[str, int]:
    """Build an Anthropic-style usage dict from a LiteLLM usage object."""
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0}
    input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    result: dict[str, int] = {
        "input_tokens": max(input_tokens - cached, 0),
        "output_tokens": output_tokens,
    }
    if cached:
        result["cache_read_input_tokens"] = cached
    return result


_STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


async def openai_stream_to_anthropic_sse(
    stream: Any,
    *,
    model_name: str,
    tool_name_mapping: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Convert OpenAI streaming chunks to Anthropic SSE bytes.

    Unlike LiteLLM's ``AnthropicStreamWrapper``, this generator **never**
    discards tool_call arguments — the trigger chunk's ``function.arguments``
    is emitted as ``input_json_delta`` immediately after ``content_block_start``.
    """
    sent_start = False
    block_index = -1
    block_open = False
    last_usage: Any = None

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            last_usage = chunk_usage
        if not choices:
            continue

        delta = choices[0].delta
        finish_reason = choices[0].finish_reason

        # message_start (once)
        if not sent_start:
            sent_start = True
            yield sse_frame("message_start", {
                "type": "message_start",
                "message": {
                    "id": f"msg_{_uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                },
            })

        # finish
        if finish_reason is not None:
            if block_open:
                yield sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_open = False
            yield sse_frame("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": _STOP_MAP.get(finish_reason, "end_turn")},
                "usage": extract_usage(chunk_usage or last_usage),
            })
            yield sse_frame("message_stop", {"type": "message_stop"})
            return

        # text content
        text = getattr(delta, "content", None)
        if text:
            if not block_open or block_index < 0:
                block_index += 1
                block_open = True
                yield sse_frame("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield sse_frame("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        # tool calls
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tc_id = getattr(tc, "id", None)
                tc_func = getattr(tc, "function", None)
                if tc_id:
                    if block_open:
                        yield sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                    block_open = True
                    name = getattr(tc_func, "name", "") or ""
                    if tool_name_mapping:
                        name = tool_name_mapping.get(name, name)
                    yield sse_frame("content_block_start", {
                        "type": "content_block_start", "index": block_index,
                        "content_block": {"type": "tool_use", "id": tc_id, "name": name, "input": {}},
                    })
                args = getattr(tc_func, "arguments", None) if tc_func else None
                if args:
                    yield sse_frame("content_block_delta", {
                        "type": "content_block_delta", "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    })

    # stream ended without explicit finish_reason
    if block_open:
        yield sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
    if sent_start:
        yield sse_frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": extract_usage(last_usage),
        })
        yield sse_frame("message_stop", {"type": "message_stop"})
