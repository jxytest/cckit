"""Unit tests for the new claude-agent-sdk 0.2.122 capability integration.

Covers:
1. ThinkingConfig model + ModelConfig.thinking
2. Agent new params (permission_handler / output_format / max_budget_usd / betas / plugins)
3. RunContext / RunnerConfig session_store
4. _build_options wiring — all new fields reach ClaudeAgentOptions
5. Default-behavior equivalence — when new params are unset, opts has no new fields
6. effort top-level wiring (gap A fix) + disable_thinking direct-path (gap B fix)
7. FileSessionStore round-trip
8. RetryMiddleware rate-limit markers + backoff
"""

from __future__ import annotations

import asyncio
import tempfile

try:  # pytest is optional for these tests — they are plain sync functions
    import pytest
except ImportError:  # pragma: no cover - allows running without pytest
    pytest = None  # type: ignore[assignment]

from cckit import (
    Agent,
    FileSessionStore,
    ModelConfig,
    RunContext,
    RunnerConfig,
    Runner,
    SandboxOptions,
    ThinkingConfig,
)
from cckit._engine.model_bridge import PreparedModelEndpoint
from cckit._engine.state import RunState
from cckit.middleware.retry import RetryMiddleware


def _build_opts(agent, ctx, model=None, prepared=None, instruction=""):
    """Helper: invoke Runner._build_options with sane defaults."""
    runner = Runner(
        config=RunnerConfig(default_model=ModelConfig(model="anthropic/claude-sonnet-4-6"))
    )
    model = model or ModelConfig(model="anthropic/claude-sonnet-4-6")
    prepared = prepared or PreparedModelEndpoint(
        model="claude-sonnet-4-6", api_key="", base_url=""
    )
    return runner._build_options(
        agent, ctx, model, prepared, SandboxOptions(), None, instruction, RunState("t")
    )


# ---------------------------------------------------------------------------
# ThinkingConfig
# ---------------------------------------------------------------------------


def test_thinking_config_variants():
    assert ThinkingConfig.adaptive().to_sdk() == {"type": "adaptive"}
    assert ThinkingConfig.enabled(4096).to_sdk() == {
        "type": "enabled",
        "budget_tokens": 4096,
    }
    assert ThinkingConfig.disabled().to_sdk() == {"type": "disabled"}
    # display optional
    t = ThinkingConfig(type="adaptive", display="summarized")
    assert t.to_sdk() == {"type": "adaptive", "display": "summarized"}


def test_modelconfig_thinking_field_default_none():
    m = ModelConfig()
    assert m.thinking is None
    assert m.disable_thinking is False  # deprecated flag still present


# ---------------------------------------------------------------------------
# Agent new params
# ---------------------------------------------------------------------------


def test_agent_new_params_defaults_none():
    a = Agent(name="x")
    assert a.permission_handler is None
    assert a.output_format is None
    assert a.max_budget_usd is None
    assert a.betas is None
    assert a.plugins is None


def test_agent_new_params_set():
    async def _perm(tool_name, tool_input, context):
        return {"behavior": "allow"}

    a = Agent(
        name="x",
        permission_handler=_perm,
        output_format={"type": "json_schema", "schema": {"type": "object"}},
        max_budget_usd=1.5,
        betas=["context-1m-2025-08-07"],
        plugins=["/path/to/plugin", {"type": "local", "path": "/other"}],
    )
    assert a.permission_handler is _perm
    assert a.output_format["type"] == "json_schema"
    assert a.max_budget_usd == 1.5
    assert a.betas == ["context-1m-2025-08-07"]
    assert a.plugins[1] == {"type": "local", "path": "/other"}


# ---------------------------------------------------------------------------
# RunContext / RunnerConfig session_store
# ---------------------------------------------------------------------------


def test_session_store_defaults_none():
    assert RunContext(prompt="x").session_store is None
    assert RunnerConfig().session_store is None


def test_session_store_set():
    store = object()  # any object implementing the protocol
    ctx = RunContext(prompt="x", session_store=store)
    assert ctx.session_store is store
    cfg = RunnerConfig(session_store=store)
    assert cfg.session_store is store


# ---------------------------------------------------------------------------
# _build_options wiring
# ---------------------------------------------------------------------------


def test_build_options_wires_all_new_capabilities():
    async def _perm(tool_name, tool_input, context):
        return {"behavior": "allow"}

    agent = Agent(
        name="full",
        effort="high",
        permission_handler=_perm,
        output_format={"type": "json_schema", "schema": {"type": "object"}},
        max_budget_usd=2.0,
        betas=["context-1m-2025-08-07"],
        plugins=["/p"],
        model=ModelConfig(thinking=ThinkingConfig.adaptive()),
    )
    ctx = RunContext(prompt="x")
    opts = _build_opts(agent, ctx, model=agent.model_config)

    assert opts.effort == "high"  # gap A fix
    assert opts.thinking == {"type": "adaptive"}
    assert opts.can_use_tool is _perm
    assert opts.output_format == {"type": "json_schema", "schema": {"type": "object"}}
    assert opts.max_budget_usd == 2.0
    assert opts.betas == ["context-1m-2025-08-07"]
    assert opts.plugins == [{"type": "local", "path": "/p"}]


def test_build_options_effort_top_level_wired():
    """Gap A fix: agent.effort must reach the top-level opts, not just sub-agents."""
    agent = Agent(name="eff", effort="medium")
    opts = _build_opts(agent, RunContext(prompt="x"))
    assert opts.effort == "medium"


def test_build_options_thinking_from_disable_thinking_flag():
    """Gap B fix: disable_thinking=True sets opts.thinking on the direct path."""
    agent = Agent(name="dt", model=ModelConfig(disable_thinking=True))
    opts = _build_opts(agent, RunContext(prompt="x"), model=agent.model_config)
    assert opts.thinking == {"type": "disabled"}


def test_build_options_thinking_field_takes_precedence_over_flag():
    agent = Agent(
        name="t",
        model=ModelConfig(disable_thinking=True, thinking=ThinkingConfig.adaptive()),
    )
    opts = _build_opts(agent, RunContext(prompt="x"), model=agent.model_config)
    assert opts.thinking == {"type": "adaptive"}


def test_build_options_plugins_normalize_str_to_dict():
    agent = Agent(name="p", plugins=["/a", {"type": "local", "path": "/b"}])
    opts = _build_opts(agent, RunContext(prompt="x"))
    assert opts.plugins == [
        {"type": "local", "path": "/a"},
        {"type": "local", "path": "/b"},
    ]


def test_build_options_session_store_from_ctx():
    store = object()
    agent = Agent(name="s")
    opts = _build_opts(agent, RunContext(prompt="x", session_store=store))
    assert opts.session_store is store
    assert opts.session_store_flush == "eager"


def test_build_options_session_store_from_runner_config():
    """Runner-level default store is used when ctx doesn't supply one."""
    store = object()
    runner = Runner(config=RunnerConfig(session_store=store))
    agent = Agent(name="s")
    model = ModelConfig(model="anthropic/claude-sonnet-4-6")
    prepared = PreparedModelEndpoint(model="claude-sonnet-4-6", api_key="", base_url="")
    opts = runner._build_options(
        agent, RunContext(prompt="x"), model, prepared, SandboxOptions(),
        None, "", RunState("t"),
    )
    assert opts.session_store is store


# ---------------------------------------------------------------------------
# Default-behavior equivalence (the hard backward-compat contract)
# ---------------------------------------------------------------------------


def test_build_options_defaults_leave_new_fields_unset():
    """When no new capability is configured, opts must not set any new field."""
    agent = Agent(name="plain")
    opts = _build_opts(agent, RunContext(prompt="x"))

    # Every newly-wired field must be at its SDK default (None / unset).
    assert opts.effort is None
    assert opts.thinking is None
    assert opts.can_use_tool is None
    assert opts.output_format is None
    assert opts.max_budget_usd is None
    assert opts.betas == []
    assert opts.plugins == []
    assert opts.session_store is None


# ---------------------------------------------------------------------------
# FileSessionStore
# ---------------------------------------------------------------------------


def test_file_session_store_roundtrip():
    async def _run():
        from claude_agent_sdk import project_key_for_directory

        with tempfile.TemporaryDirectory() as d:
            store = FileSessionStore(directory=d)
            pk = project_key_for_directory(d)
            key = {"project_key": pk, "session_id": "sess-1"}
            await store.append(key, [
                {"type": "user", "uuid": "u1", "content": "hi"},
                {"type": "assistant", "uuid": "a1", "content": "yo"},
            ])
            # idempotent re-append of u1 must not duplicate
            await store.append(key, [{"type": "user", "uuid": "u1", "content": "dup"}])
            loaded = await store.load(key)
            assert loaded is not None and len(loaded) == 2
            assert loaded[0]["uuid"] == "u1" and loaded[0]["content"] == "hi"
            # unknown session -> None
            assert await store.load({"project_key": pk, "session_id": "nope"}) is None
            # list + delete
            listed = await store.list_sessions(pk)
            assert len(listed) == 1 and listed[0]["session_id"] == "sess-1"
            await store.delete(key)
            assert await store.load(key) is None

    asyncio.run(_run())


def test_file_session_store_passes_sdk_conformance():
    """FileSessionStore must satisfy the SDK's official 14-contract suite."""
    async def _run():
        from claude_agent_sdk.testing import run_session_store_conformance

        tmpdirs: list[str] = []

        def make_store():
            d = tempfile.mkdtemp(prefix="cckit-conf-")
            tmpdirs.append(d)
            return FileSessionStore(directory=d)

        try:
            await run_session_store_conformance(make_store)  # raises on failure
        finally:
            import shutil
            for d in tmpdirs:
                shutil.rmtree(d, ignore_errors=True)

    asyncio.run(_run())


def test_build_options_session_store_flush_configurable():
    """session_store_flush is configurable (default eager, can be batched)."""
    store = object()
    agent = Agent(name="s")
    # default -> eager
    opts = _build_opts(agent, RunContext(prompt="x", session_store=store))
    assert opts.session_store_flush == "eager"
    # explicit batched via ctx
    opts2 = _build_opts(
        agent, RunContext(prompt="x", session_store=store, session_store_flush="batched")
    )
    assert opts2.session_store_flush == "batched"


def test_build_options_session_store_flush_from_runner_config():
    """Runner-level session_store_flush is used when ctx doesn't supply one."""
    store = object()
    runner = Runner(config=RunnerConfig(session_store=store, session_store_flush="batched"))
    agent = Agent(name="s")
    model = ModelConfig(model="anthropic/claude-sonnet-4-6")
    prepared = PreparedModelEndpoint(model="claude-sonnet-4-6", api_key="", base_url="")
    opts = runner._build_options(
        agent, RunContext(prompt="x"), model, prepared, SandboxOptions(),
        None, "", RunState("t"),
    )
    assert opts.session_store is store
    assert opts.session_store_flush == "batched"


def test_retry_middleware_rate_limit_markers():
    mw = RetryMiddleware(rate_limit_base_delay=8.0)
    assert "overloaded" in mw._RATE_LIMIT_MARKERS
    assert "rate_limit" in mw._RATE_LIMIT_MARKERS
    assert mw.rate_limit_base_delay == 8.0
    # default back-compat
    assert RetryMiddleware().rate_limit_base_delay == 5.0


def test_retry_middleware_detects_rate_limit_status_codes():
    """HTTP 429/529 in an error message are detected as rate-limited."""
    mw = RetryMiddleware()
    assert mw._RATE_LIMIT_STATUS_RE.search("error 429 too many requests")
    assert mw._RATE_LIMIT_STATUS_RE.search("api_error_status=529")
    # port-like numbers must NOT match (word boundary): 5290 is a port, not 529
    assert not mw._RATE_LIMIT_STATUS_RE.search("connection to 10.0.0.5:5290 refused")
    # 4291 is not 429
    assert not mw._RATE_LIMIT_STATUS_RE.search("error code 4291")


def test_retry_middleware_permanent_not_in_rate_limit():
    """Permanent markers (invalid_api_key etc.) must still short-circuit."""
    mw = RetryMiddleware()
    err_lower = "invalid_api_key: 401"
    assert any(m in err_lower for m in mw._PERMANENT_MARKERS)


# ---------------------------------------------------------------------------
# ThinkingConfig validation (enabled requires positive budget_tokens)
# ---------------------------------------------------------------------------


def test_thinking_config_enabled_requires_budget():
    # valid
    t = ThinkingConfig.enabled(4096)
    assert t.to_sdk() == {"type": "enabled", "budget_tokens": 4096}
    # invalid: missing budget
    try:
        ThinkingConfig(type="enabled")
        raise AssertionError("expected ValueError for enabled without budget_tokens")
    except ValueError:
        pass
    # invalid: zero/negative budget
    try:
        ThinkingConfig(type="enabled", budget_tokens=0)
        raise AssertionError("expected ValueError for budget_tokens=0")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Regression: existing fields unaffected by the integration
# ---------------------------------------------------------------------------


def test_build_options_existing_fields_unchanged():
    """Core pre-existing opts fields must keep their values after the change.

    Guards against the _build_options / _resolve_model edits accidentally
    altering system_prompt, env isolation, sandbox=None, setting_sources,
    resume wiring, or the model passed to opts.
    """
    agent = Agent(name="reg", tools=["Read"], instruction="be helpful")
    ctx = RunContext(prompt="hi", user="alice")
    opts = _build_opts(agent, ctx, instruction=agent.resolve_instruction(ctx))

    # system_prompt: preset + appended instruction
    assert opts.system_prompt["type"] == "preset"
    assert opts.system_prompt["preset"] == "claude_code"
    assert opts.system_prompt["append"] == "be helpful"
    # sandbox stays None (rules go in settings JSON)
    assert opts.sandbox is None
    # setting_sources forced (Windows bug workaround)
    assert opts.setting_sources == ["user", "project", "local"]
    # tools / allowed_tools
    assert opts.tools == ["Read"]
    assert opts.allowed_tools == ["Read"]
    # user passthrough
    assert opts.user == "alice"
    # extra_args
    assert opts.extra_args == {"debug-to-stderr": None}
    # resume unset by default
    assert opts.resume is None
    assert opts.fork_session is False


def test_resolve_model_propagates_thinking_to_subagent_merge():
    """_resolve_model must merge `thinking` so sub-agents keep their config."""
    from cckit import Runner

    runner = Runner(
        config=RunnerConfig(
            default_model=ModelConfig(
                model="anthropic/claude-sonnet-4-6",
                thinking=ThinkingConfig.adaptive(),
            )
        )
    )
    # Agent with no own model → inherits runner default, thinking must survive
    agent = Agent(name="sub")
    merged = runner._resolve_model(agent, RunContext(prompt="x"), is_sub_agent=True)
    assert merged.thinking is not None
    assert merged.thinking.type == "adaptive"
    # Agent's own thinking overrides runner default
    agent2 = Agent(
        name="sub2",
        model=ModelConfig(model="anthropic/claude-sonnet-4-6", thinking=ThinkingConfig.disabled()),
    )
    merged2 = runner._resolve_model(agent2, RunContext(prompt="x"), is_sub_agent=True)
    assert merged2.thinking.type == "disabled"
