"""Tests for the tool_result single-image fix.

Guards the LiteLLM adapter bug where a ``tool_result`` carrying exactly one
image block is flattened into a plain data-URI string, so the model receives
text instead of pixels.  See ``cckit/_engine/_patches/tool_result_image.py``
for the upstream issue/PR references.

These tests pin behaviour against the pinned LiteLLM version — if the pin in
pyproject.toml moves and upstream has fixed the bug, ``test_end_to_end_*``
still passes (the patch becomes a no-op) and the patch can be deleted.
"""

from __future__ import annotations

import pytest

from cckit._engine._patches.tool_result_image import (
    _normalize_tool_message,
    apply_tool_result_image_patch,
)

_PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"


# ── unit: which messages get rewritten ────────────────────────────


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("plain text result", {"role": "tool", "content": "hello"}),
        ("http url", {"role": "tool", "content": "https://example.com/a.png"}),
        ("non-tool role", {"role": "user", "content": _PNG_URI}),
        ("non-image data uri", {"role": "tool", "content": "data:application/pdf;base64,AA"}),
        ("data uri without base64", {"role": "tool", "content": "data:image/svg+xml,<svg/>"}),
        ("content is None", {"role": "tool", "content": None}),
        ("no content key", {"role": "tool"}),
        (
            "already structured",
            {
                "role": "tool",
                "content": [{"type": "image_url", "image_url": {"url": _PNG_URI}}],
            },
        ),
    ],
)
def test_leaves_unrelated_messages_untouched(name: str, message: dict) -> None:
    before = repr(message)
    assert _normalize_tool_message(message) is False, name
    assert repr(message) == before, f"{name}: message was mutated"


@pytest.mark.parametrize("media", ["png", "jpeg", "gif", "webp"])
def test_rewrites_bare_image_data_uri(media: str) -> None:
    uri = f"data:image/{media};base64,AAAA"
    message = {"role": "tool", "tool_call_id": "call_1", "content": uri}

    assert _normalize_tool_message(message) is True
    assert message["content"] == [{"type": "image_url", "image_url": {"url": uri}}]
    # The tool_call_id must survive — dropping it orphans the tool result.
    assert message["tool_call_id"] == "call_1"


def test_is_idempotent() -> None:
    message = {"role": "tool", "content": _PNG_URI}

    assert _normalize_tool_message(message) is True
    once = message["content"]
    assert _normalize_tool_message(message) is False
    assert message["content"] == once


# ── integration: through the real LiteLLM adapter ─────────────────


def _anthropic_messages(image_block: dict) -> list[dict]:
    """A Read-tool exchange whose tool_result carries *image_block*."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "describe it"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/image.png"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": [image_block]},
            ],
        },
    ]


@pytest.fixture
def adapter():
    litellm_adapters = pytest.importorskip(
        "litellm.llms.anthropic.experimental_pass_through.adapters.transformation",
    )
    apply_tool_result_image_patch()
    return litellm_adapters.LiteLLMAnthropicMessagesAdapter()


def test_end_to_end_single_image_survives_translation(adapter) -> None:
    """The regression itself: one image block must reach the wire as an image."""
    messages = _anthropic_messages(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"},
        },
    )

    result = adapter.translate_anthropic_messages_to_openai(messages)

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert isinstance(content, list), "image was flattened to a bare string"
    assert [part["type"] for part in content] == ["image_url"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_end_to_end_text_tool_result_unaffected(adapter) -> None:
    messages = _anthropic_messages({"type": "text", "text": "file contents"})

    result = adapter.translate_anthropic_messages_to_openai(messages)

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "file contents"


def test_end_to_end_multi_block_tool_result_unaffected(adapter) -> None:
    """The multi-item branch was already correct; make sure we didn't disturb it."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "describe it"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [
                        {"type": "text", "text": "here:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo",
                            },
                        },
                    ],
                },
            ],
        },
    ]

    result = adapter.translate_anthropic_messages_to_openai(messages)

    content = [m for m in result if m.get("role") == "tool"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]


def test_patch_composes_with_deepseek_patch(adapter) -> None:
    """Both patches wrap the same adapter method; order must not matter."""
    from cckit._engine._patches.deepseek_reasoning import apply_deepseek_reasoning_patch

    apply_deepseek_reasoning_patch()
    apply_tool_result_image_patch()

    messages = _anthropic_messages(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"},
        },
    )

    result = adapter.translate_anthropic_messages_to_openai(messages)

    content = [m for m in result if m.get("role") == "tool"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
