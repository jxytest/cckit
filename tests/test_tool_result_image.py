"""Tests for the tool_result image relocation fix.

Guards the LiteLLM adapter bug where a ``tool_result`` carrying one image
block is flattened into a plain data-URI string, so the model receives text
instead of pixels — and the broader problem that images buried inside a
``role:"tool"`` message are ignored by some providers even when structured
correctly.  See ``cckit/_engine/_patches/tool_result_image.py``.

The invariant these tests protect: every ``tool`` message keeps its
``tool_call_id`` and stays in its original position relative to the
assistant message that requested it — relocation must not orphan a tool
call, which providers reject outright.
"""

from __future__ import annotations

import pytest

from cckit._engine._patches.tool_result_image import (
    _IMAGE_MOVED_PLACEHOLDER,
    _extract_image_url,
    _flatten_text_only_content,
    _relocate_tool_result_images,
    apply_tool_result_image_patch,
)

_PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"


# ── unit: which tool messages hold a relocatable image ────────────


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
            "mixed text + image stays put",
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "here:"},
                    {"type": "image_url", "image_url": {"url": _PNG_URI}},
                ],
            },
        ),
    ],
)
def test_ignores_messages_without_a_lone_image(name: str, message: dict) -> None:
    assert _extract_image_url(message) is None, name


@pytest.mark.parametrize("media", ["png", "jpeg", "gif", "webp"])
def test_detects_bare_data_uri(media: str) -> None:
    uri = f"data:image/{media};base64,AAAA"
    assert _extract_image_url({"role": "tool", "content": uri}) == uri


def test_detects_structured_single_image() -> None:
    """The in-tool structured form is also relocated: some providers ignore it."""
    msg = {
        "role": "tool",
        "content": [{"type": "image_url", "image_url": {"url": _PNG_URI}}],
    }
    assert _extract_image_url(msg) == _PNG_URI


# ── unit: relocation preserves conversation structure ─────────────


def _tool_call(call_id: str, name: str = "Read") -> dict:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_relocates_image_into_following_user_message() -> None:
    messages = [
        {"role": "user", "content": "describe it"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A")]},
        {"role": "tool", "tool_call_id": "call_A", "content": _PNG_URI},
    ]

    assert _relocate_tool_result_images(messages) == 1

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    # The tool message keeps its id and is no longer a giant data URI.
    assert messages[2]["tool_call_id"] == "call_A"
    assert messages[2]["content"] == _IMAGE_MOVED_PLACEHOLDER
    # The image rides in the appended user message.
    assert messages[3]["content"] == [
        {"type": "image_url", "image_url": {"url": _PNG_URI}},
    ]


def test_parallel_tool_calls_keep_pairing() -> None:
    """An image tool result must not split a contiguous run of tool messages.

    Inserting the user message between two ``tool`` messages would orphan the
    second tool_call, which providers reject.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_A"), _tool_call("call_B", "echo")],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": _PNG_URI},
        {"role": "tool", "tool_call_id": "call_B", "content": "TOKEN"},
    ]

    assert _relocate_tool_result_images(messages) == 1

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
        "call_A",
        "call_B",
    ]
    # The text result is untouched; only the image moved.
    assert messages[3]["content"] == "TOKEN"
    assert messages[4]["content"][0]["type"] == "image_url"


def test_multiple_images_share_one_user_message() -> None:
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A")]},
        {"role": "tool", "tool_call_id": "call_A", "content": _PNG_URI},
        {"role": "tool", "tool_call_id": "call_B", "content": _PNG_URI},
    ]

    assert _relocate_tool_result_images(messages) == 2

    assert [m["role"] for m in messages] == ["assistant", "tool", "tool", "user"]
    assert [p["type"] for p in messages[3]["content"]] == ["image_url", "image_url"]


def test_images_land_next_to_their_own_turn() -> None:
    """Two separate tool turns must not have their images pooled at the end."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A")]},
        {"role": "tool", "tool_call_id": "call_A", "content": _PNG_URI},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_B")]},
        {"role": "tool", "tool_call_id": "call_B", "content": _PNG_URI},
    ]

    assert _relocate_tool_result_images(messages) == 2

    assert [m["role"] for m in messages] == [
        "assistant", "tool", "user", "assistant", "tool", "user",
    ]


def test_is_idempotent() -> None:
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A")]},
        {"role": "tool", "tool_call_id": "call_A", "content": _PNG_URI},
    ]

    assert _relocate_tool_result_images(messages) == 1
    snapshot = [dict(m) for m in messages]
    assert _relocate_tool_result_images(messages) == 0
    assert messages == snapshot


def test_no_images_leaves_conversation_identical() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A")]},
        {"role": "tool", "tool_call_id": "call_A", "content": "file contents"},
    ]
    snapshot = [dict(m) for m in messages]

    assert _relocate_tool_result_images(messages) == 0
    assert messages == snapshot


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


_IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"},
}


@pytest.fixture
def adapter():
    litellm_adapters = pytest.importorskip(
        "litellm.llms.anthropic.experimental_pass_through.adapters.transformation",
    )
    apply_tool_result_image_patch()
    return litellm_adapters.LiteLLMAnthropicMessagesAdapter()


def test_end_to_end_single_image_reaches_the_model(adapter) -> None:
    """The regression itself: the image must leave as an image, not as text."""
    result = adapter.translate_anthropic_messages_to_openai(
        _anthropic_messages(_IMAGE_BLOCK),
    )

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == _IMAGE_MOVED_PLACEHOLDER

    image_parts = [
        part
        for m in result
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_end_to_end_text_tool_result_unaffected(adapter) -> None:
    result = adapter.translate_anthropic_messages_to_openai(
        _anthropic_messages({"type": "text", "text": "file contents"}),
    )

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "file contents"


def test_patch_composes_with_deepseek_patch(adapter) -> None:
    """Both patches wrap the same adapter method; order must not matter."""
    from cckit._engine._patches.deepseek_reasoning import apply_deepseek_reasoning_patch

    apply_deepseek_reasoning_patch()
    apply_tool_result_image_patch()

    result = adapter.translate_anthropic_messages_to_openai(
        _anthropic_messages(_IMAGE_BLOCK),
    )

    assert [m for m in result if m.get("role") == "tool"][0]["content"] == (
        _IMAGE_MOVED_PLACEHOLDER
    )
    assert any(
        part.get("type") == "image_url"
        for m in result
        if isinstance(m.get("content"), list)
        for part in m["content"]
    )


# ── text-only tool results must reach the model as a string ───────


def test_flattens_multi_text_tool_result() -> None:
    """A Task sub-agent returns description + agentId — two text blocks."""
    msg = {
        "role": "tool",
        "tool_call_id": "call_A",
        "content": [
            {"type": "text", "text": "the description"},
            {"type": "text", "text": "agentId: abc"},
        ],
    }

    assert _flatten_text_only_content(msg) is True
    assert msg["content"] == "the description\n\nagentId: abc"
    assert msg["tool_call_id"] == "call_A"


def test_flattens_single_text_tool_result() -> None:
    msg = {"role": "tool", "content": [{"type": "text", "text": "only"}]}

    assert _flatten_text_only_content(msg) is True
    assert msg["content"] == "only"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("already a string", {"role": "tool", "content": "plain"}),
        ("empty list", {"role": "tool", "content": []}),
        ("non-tool role", {"role": "user", "content": [{"type": "text", "text": "x"}]}),
        (
            "contains an image",
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "here:"},
                    {"type": "image_url", "image_url": {"url": _PNG_URI}},
                ],
            },
        ),
        (
            "image only",
            {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": _PNG_URI}}]},
        ),
    ],
)
def test_flatten_leaves_other_shapes_alone(name: str, message: dict) -> None:
    before = repr(message)
    assert _flatten_text_only_content(message) is False, name
    assert repr(message) == before, f"{name}: message was mutated"


def test_flatten_is_idempotent() -> None:
    msg = {"role": "tool", "content": [{"type": "text", "text": "a"}]}

    assert _flatten_text_only_content(msg) is True
    assert _flatten_text_only_content(msg) is False
    assert msg["content"] == "a"


def test_end_to_end_subagent_text_result_reaches_model(adapter) -> None:
    """The Task sub-agent regression: two text blocks must arrive as a string."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "look at it"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "Task", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [
                        {"type": "text", "text": "full description"},
                        {"type": "text", "text": "agentId: abc <usage>…</usage>"},
                    ],
                },
            ],
        },
    ]

    result = adapter.translate_anthropic_messages_to_openai(messages)

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert isinstance(content, str), "text result stayed a list; providers read it as empty"
    assert "full description" in content
    assert "agentId: abc" in content


def test_end_to_end_image_result_still_relocates(adapter) -> None:
    """Flattening must not undo the image relocation."""
    result = adapter.translate_anthropic_messages_to_openai(
        _anthropic_messages(_IMAGE_BLOCK),
    )

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == _IMAGE_MOVED_PLACEHOLDER
    assert any(
        part.get("type") == "image_url"
        for m in result
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        for part in m["content"]
    )
