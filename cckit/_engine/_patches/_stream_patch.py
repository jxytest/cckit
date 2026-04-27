"""Patch for LiteLLM's ``AnthropicStreamWrapper`` trigger-chunk argument loss.

LiteLLM's ``AnthropicStreamWrapper.__anext__`` (streaming_iterator.py:306-332)
discards the ``processed_chunk`` when a content-block type transition occurs
(``should_start_new_block == True``).  The comment says "the content_block_start
already carries the relevant information", but ``content_block_start.input``
is always ``{}``.  When a provider batches ``id + name + arguments`` in one
chunk (common with Chinese LLM gateways), the entire tool input is lost.

This module provides :func:`apply_stream_patch` which monkey-patches
``AnthropicAdapter.translate_completion_output_params_streaming`` to use a
fixed subclass that queues the ``processed_chunk`` after ``content_block_start``.

The patch is idempotent and safe to call multiple times.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_stream_patch() -> None:
    """Apply the monkey-patch once.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            AnthropicAdapter,
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError:
        logger.debug("LiteLLM adapter internals unavailable; skipping stream patch")
        return

    class PatchedStreamWrapper(AnthropicStreamWrapper):
        """Fixed version that does NOT discard trigger-chunk deltas."""

        async def __anext__(self):  # noqa: PLR0915
            try:
                # Always return queued chunks first
                if self.chunk_queue:
                    return self.chunk_queue.popleft()

                # Queue initial chunks if not sent yet
                if self.sent_first_chunk is False:
                    self.sent_first_chunk = True
                    from litellm._uuid import uuid
                    self.chunk_queue.append({
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    })
                    return self.chunk_queue.popleft()

                if self.sent_content_block_start is False:
                    self.sent_content_block_start = True
                    self.chunk_queue.append({
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    return self.chunk_queue.popleft()

                async for chunk in self.completion_stream:
                    if chunk == "None" or chunk is None:
                        raise Exception

                    should_start_new_block = self._should_start_new_content_block(chunk)
                    if should_start_new_block:
                        self._increment_content_block_index()

                    processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                        response=chunk,
                        current_content_block_index=self.current_content_block_index,
                    )

                    # Check if this is a usage chunk and we have a held stop_reason chunk
                    if (
                        self.holding_stop_reason_chunk is not None
                        and getattr(chunk, "usage", None) is not None
                    ):
                        merged_chunk = self.holding_stop_reason_chunk.copy()
                        if "delta" not in merged_chunk:
                            merged_chunk["delta"] = {}

                        uncached_input_tokens = chunk.usage.prompt_tokens or 0
                        if (
                            hasattr(chunk.usage, "prompt_tokens_details")
                            and chunk.usage.prompt_tokens_details
                        ):
                            cached_tokens = (
                                getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0)
                                or 0
                            )
                            uncached_input_tokens -= cached_tokens

                        from litellm.types.llms.anthropic import UsageDelta
                        usage_dict: UsageDelta = {
                            "input_tokens": uncached_input_tokens,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }
                        if (
                            hasattr(chunk.usage, "_cache_creation_input_tokens")
                            and chunk.usage._cache_creation_input_tokens > 0
                        ):
                            usage_dict["cache_creation_input_tokens"] = chunk.usage._cache_creation_input_tokens
                        if (
                            hasattr(chunk.usage, "_cache_read_input_tokens")
                            and chunk.usage._cache_read_input_tokens > 0
                        ):
                            usage_dict["cache_read_input_tokens"] = chunk.usage._cache_read_input_tokens
                        merged_chunk["usage"] = usage_dict

                        self.chunk_queue.append(merged_chunk)
                        self.queued_usage_chunk = True
                        self.holding_stop_reason_chunk = None
                        return self.chunk_queue.popleft()

                    if not self.queued_usage_chunk:
                        if should_start_new_block and not self.sent_content_block_finish:
                            # 1. Stop current content block
                            self.chunk_queue.append({
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            })
                            # 2. Start new content block
                            self.chunk_queue.append({
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            })
                            # ── FIX: also queue the trigger chunk's delta ──
                            if processed_chunk.get("type") == "content_block_delta":
                                self.chunk_queue.append(processed_chunk)
                            # ───────────────────────────────────────────────
                            self.sent_content_block_finish = False
                            return self.chunk_queue.popleft()

                        if (
                            processed_chunk["type"] == "message_delta"
                            and self.sent_content_block_finish is False
                        ):
                            self.chunk_queue.append({
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            })
                            self.sent_content_block_finish = True
                            if processed_chunk.get("delta", {}).get("stop_reason") is not None:
                                self.holding_stop_reason_chunk = processed_chunk
                            else:
                                self.chunk_queue.append(processed_chunk)
                            return self.chunk_queue.popleft()
                        elif self.holding_chunk is not None:
                            self.chunk_queue.append(self.holding_chunk)
                            self.chunk_queue.append(processed_chunk)
                            self.holding_chunk = None
                            return self.chunk_queue.popleft()
                        else:
                            self.chunk_queue.append(processed_chunk)
                            return self.chunk_queue.popleft()

                # Handle any remaining held chunks after stream ends
                if not self.queued_usage_chunk:
                    if self.holding_stop_reason_chunk is not None:
                        self.chunk_queue.append(self.holding_stop_reason_chunk)
                        self.holding_stop_reason_chunk = None
                    if self.holding_chunk is not None:
                        self.chunk_queue.append(self.holding_chunk)
                        self.holding_chunk = None

                if not self.sent_last_message:
                    self.sent_last_message = True
                    self.chunk_queue.append({"type": "message_stop"})

                if self.chunk_queue:
                    return self.chunk_queue.popleft()

                raise StopIteration

            except StopIteration:
                if self.chunk_queue:
                    return self.chunk_queue.popleft()
                if self.holding_stop_reason_chunk is not None:
                    return self.holding_stop_reason_chunk
                if not self.sent_last_message:
                    self.sent_last_message = True
                    return {"type": "message_stop"}
                raise StopAsyncIteration

    # ── apply the patch ──────────────────────────────────────────

    _original = AnthropicAdapter.translate_completion_output_params_streaming

    def _patched_translate(self: Any, completion_stream: Any, model: str,
                           tool_name_mapping: Any = None, **kw: Any) -> Any:
        wrapper = PatchedStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            tool_name_mapping=tool_name_mapping,
        )
        return wrapper.async_anthropic_sse_wrapper()

    AnthropicAdapter.translate_completion_output_params_streaming = _patched_translate  # type: ignore[assignment]
    _PATCHED = True
    logger.debug("Applied AnthropicStreamWrapper trigger-chunk fix")
