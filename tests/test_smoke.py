"""Smoke test for cckit — no API key required.

Verifies:
1. All public imports work
2. Agent instantiation (simple + gateway ModelConfig + sub-agents)
3. Type construction (ModelConfig, RunContext, RunnerConfig, etc.)
4. Middleware instantiation
5. WorkspaceManager lifecycle
6. SkillProvisioner validation
7. Exception hierarchy
8. RunnerConfig.from_env()
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from cckit import (  # noqa: F401 — test_imports verifies all public API symbols
    Agent,
    AgentExecutionError,
    AgentResult,
    CckitError,
    ConcurrencyMiddleware,
    GitConfig,
    GitLabAPIError,
    GitOperationError,
    LoggingMiddleware,
    Middleware,
    ModelConfig,
    RetryMiddleware,
    RunContext,
    Runner,
    RunnerConfig,
    SandboxOptions,
    SkillError,
    StreamResult,
    TaskStatus,
    WorkspaceConfig,
    WorkspaceError,
)
from cckit._engine._patches.deepseek_reasoning import patch_deepseek_reasoning
from cckit._engine.model_bridge import (
    LiteLLMAnthropicBridge,
    PreparedModelEndpoint,
    prepare_model_endpoint,
    resolve_model_transport,
)
from cckit._engine.state import RunState
from cckit.git.gitlab_client import GitLabClient
from cckit.sandbox.config import SandboxConfigBuilder
from cckit.sandbox.workspace import WorkspaceManager
from cckit.skill.provisioner import SkillProvisioner


def test_imports():
    """Verify all public API imports."""
    # All imports are already at the top, so this test just confirms they work
    assert Agent is not None
    assert Runner is not None
    assert RunContext is not None


def test_agent_simple():
    """Simple agent instantiation."""
    agent = Agent(
        name="test",
        instruction="You are a test agent.",
        tools=["Read", "Grep"],
    )
    assert agent.name == "test"
    assert agent.tools == ["Read", "Grep"]
    assert agent.model_config is None  # inherits from Runner
    assert repr(agent) == "<Agent name='test'>"


def test_agent_with_model():
    """Agent with ModelConfig."""
    agent = Agent(
        name="with-model",
        model=ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test"),
        instruction="Hello",
        tools=["Bash"],
    )
    assert agent.model_config is not None
    assert agent.model_config.model == "anthropic/claude-sonnet-4-6"
    assert agent.model_config.api_key == "sk-test"


def test_agent_with_sandbox():
    """Agent can carry its own sandbox policy."""
    sandbox = SandboxOptions(enabled=True, allowed_domains=["example.com"])
    agent = Agent(
        name="with-sandbox",
        instruction="Hello",
        tools=["Bash"],
        sandbox=sandbox,
    )
    assert agent.sandbox_config is sandbox
    assert agent.sandbox_config is not None
    assert agent.sandbox_config.enabled is True
    assert agent.sandbox_config.allowed_domains == ["example.com"]


def test_agent_with_string_model():
    """Agent with string model shorthand."""
    agent = Agent(name="string-model", model="claude-opus-4-6", tools=["Read"])
    assert agent.model_config is not None
    assert agent.model_config.model == "claude-opus-4-6"


def test_agent_with_gateway_model_config():
    """Agent can target a gateway-backed third-party model via ModelConfig."""
    agent = Agent(
        name="gateway-test",
        model=ModelConfig(
            model="openai/gemini-3-pro",
            base_url="http://localhost:4000",
            api_key="sk-gateway",
        ),
        tools=["Read"],
    )
    assert agent.model_config is not None
    assert agent.model_config.model == "openai/gemini-3-pro"
    assert agent.model_config.base_url == "http://localhost:4000"
    assert agent.model_config.api_key == "sk-gateway"


def test_model_config_normalizes_messages_endpoint():
    """ModelConfig strips pasted endpoint suffixes down to the API base."""
    cfg = ModelConfig(
        model="claude-sonnet-4-6",
        base_url="http://localhost:4000/v1/messages",
    )
    assert cfg.base_url == "http://localhost:4000"
    assert cfg.endpoint_protocol == "anthropic"


def test_model_config_normalizes_chat_completions_endpoint():
    """ModelConfig strips a pasted /chat/completions suffix."""
    cfg = ModelConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1/chat/completions",
    )
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.endpoint_protocol == "chat"


def test_model_config_normalizes_responses_endpoint():
    """ModelConfig strips a pasted /responses suffix."""
    cfg = ModelConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1/responses",
    )
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.endpoint_protocol == "responses"


def test_resolve_model_transport_defaults_unprefixed_models_to_chat():
    """Unprefixed models default to chat/completions."""
    transport = resolve_model_transport(
        ModelConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )
    )
    assert transport.protocol == "chat"
    assert transport.custom_llm_provider == "openai"
    assert transport.model == "gpt-4o-mini"
    assert transport.api_base == "https://api.openai.com/v1"


def test_resolve_model_transport_uses_prefix_before_base_url_hint():
    """Explicit model prefixes win over any full-path base URL hint."""
    transport = resolve_model_transport(
        ModelConfig(
            model="openai/gpt-4o-mini",
            base_url="https://api.openai.com/v1/chat/completions",
        )
    )
    assert transport.protocol == "responses"
    assert transport.custom_llm_provider == "openai"
    assert transport.model == "gpt-4o-mini"
    assert transport.api_base == "https://api.openai.com/v1"


def test_resolve_model_transport_uses_base_url_hint_for_unprefixed_models():
    """A full-path base URL selects the matching protocol when the model has no prefix."""
    anthropic_transport = resolve_model_transport(
        ModelConfig(
            model="claude-sonnet-4-6",
            base_url="https://gateway.example.com/v1/messages",
        )
    )
    assert anthropic_transport.protocol == "anthropic"
    assert anthropic_transport.custom_llm_provider == "anthropic"
    assert anthropic_transport.model == "claude-sonnet-4-6"
    assert anthropic_transport.api_base == "https://gateway.example.com"

    responses_transport = resolve_model_transport(
        ModelConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1/responses",
        )
    )
    assert responses_transport.protocol == "responses"
    assert responses_transport.custom_llm_provider == "openai"
    assert responses_transport.model == "responses/gpt-4o-mini"
    assert responses_transport.api_base == "https://api.openai.com/v1/responses"

    chat_transport = resolve_model_transport(
        ModelConfig(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
    )
    assert chat_transport.protocol == "chat"
    assert chat_transport.custom_llm_provider == "openai"
    assert chat_transport.model == "qwen-plus"
    assert chat_transport.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_prepare_model_endpoint_uses_litellm_bridge_for_plain_model(monkeypatch):
    """Unprefixed models still use the local bridge, defaulting to chat semantics."""
    async def fake_start(self):
        self.base_url = "http://127.0.0.1:41001"
        return self

    monkeypatch.setattr("cckit._engine.model_bridge.LiteLLMAnthropicBridge.start", fake_start)

    prepared = asyncio.run(
        prepare_model_endpoint(
            ModelConfig(
                model="claude-sonnet-4-6",
                api_key="sk-test",
            )
        )
    )
    assert prepared.model == "claude-sonnet-4-6"
    assert prepared.api_key == "cckit-bridge"
    assert prepared.base_url == "http://127.0.0.1:41001"
    assert prepared.bridge is not None
    assert prepared.bridge._primary.config.model == "claude-sonnet-4-6"
    assert prepared.bridge._primary.transport.protocol == "chat"
    assert prepared.bridge._primary.transport.custom_llm_provider == "openai"


def test_prepare_model_endpoint_missing_litellm_is_clear(monkeypatch):
    """Bridge-backed models should fail clearly if LiteLLM bridge deps are missing."""
    def fake_build_app(self):
        raise AgentExecutionError("cckit model execution requires LiteLLM bridge dependencies")

    monkeypatch.setattr(
        "cckit._engine.model_bridge.LiteLLMAnthropicBridge._build_app",
        fake_build_app,
    )

    with pytest.raises(AgentExecutionError, match="LiteLLM bridge dependencies"):
        asyncio.run(prepare_model_endpoint(ModelConfig(model="claude-sonnet-4-6")))


def test_prepare_model_endpoint_passes_anthropic_models_directly_to_sdk():
    """Anthropic-prefixed models bypass the local bridge and use direct SDK settings."""
    prepared = asyncio.run(
        prepare_model_endpoint(
            ModelConfig(
                model="anthropic/claude-sonnet-4-6",
                api_key="sk-anthropic",
                base_url="https://api.anthropic.example.com",
            )
        )
    )

    assert prepared.model == "claude-sonnet-4-6"
    assert prepared.api_key == "sk-anthropic"
    assert prepared.base_url == "https://api.anthropic.example.com"
    assert prepared.bridge is None


def test_prepare_model_endpoint_uses_base_url_hint_for_direct_anthropic_sdk_path():
    """Unprefixed models with an Anthropic endpoint hint should bypass the bridge."""
    prepared = asyncio.run(
        prepare_model_endpoint(
            ModelConfig(
                model="claude-sonnet-4-6",
                api_key="sk-anthropic",
                base_url="https://gateway.example.com/v1/messages",
            )
        )
    )

    assert prepared.model == "claude-sonnet-4-6"
    assert prepared.api_key == "sk-anthropic"
    assert prepared.base_url == "https://gateway.example.com"
    assert prepared.bridge is None


def test_prepare_model_endpoint_prefixed_model_uses_bridge(monkeypatch):
    """OpenAI-prefixed models should force Responses API routing through the bridge."""
    async def fake_start(self):
        self.base_url = "http://127.0.0.1:41001"
        return self

    monkeypatch.setattr("cckit._engine.model_bridge.LiteLLMAnthropicBridge.start", fake_start)

    prepared = asyncio.run(
        prepare_model_endpoint(
            ModelConfig(
                model="openai/gpt-4o-mini",
                api_key="sk-openai",
                base_url="https://api.openai.com/v1",
            )
        )
    )
    assert prepared.model == "openai/gpt-4o-mini"
    assert prepared.api_key == "cckit-bridge"
    assert prepared.base_url == "http://127.0.0.1:41001"
    assert prepared.bridge is not None
    assert prepared.bridge._primary.config.model == "openai/gpt-4o-mini"
    assert prepared.bridge._primary.transport.protocol == "responses"
    assert prepared.bridge._primary.transport.custom_llm_provider == "openai"


def test_bridge_chat_transport_drops_anthropic_only_fields():
    """Chat/completions transport must not forward Anthropic-only top-level params."""
    bridge = LiteLLMAnthropicBridge(
        ModelConfig(
            model="qwen-plus",
            api_key="sk-chat",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
    )

    kwargs = bridge._build_kwargs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"format": "json_schema"},
            "context_management": {"edits": []},
        }
    )

    assert "output_config" not in kwargs
    assert "context_management" not in kwargs
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["model"] == "qwen-plus"
    assert kwargs["api_base"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_bridge_chat_transport_clamps_max_tokens_to_model_config():
    """Chat/completions transport should cap Claude's larger default max_tokens."""
    bridge = LiteLLMAnthropicBridge(
        ModelConfig(
            model="qwen-plus",
            api_key="sk-chat",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            max_tokens=16384,
        )
    )

    kwargs = bridge._build_kwargs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32000,
        }
    )

    assert kwargs["max_tokens"] == 16384


def test_bridge_responses_transport_preserves_anthropic_only_fields():
    """Responses transport keeps Anthropic pass-through fields for LiteLLM to handle."""
    bridge = LiteLLMAnthropicBridge(
        ModelConfig(
            model="openai/gpt-4o-mini",
            api_key="sk-openai",
            base_url="https://api.openai.com/v1",
        )
    )

    kwargs = bridge._build_kwargs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "output_config": {"format": "json_schema"},
            "context_management": {"edits": []},
        }
    )

    assert kwargs["output_config"] == {"format": "json_schema"}
    assert kwargs["context_management"] == {"edits": []}
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["api_base"] == "https://api.openai.com/v1"


def test_bridge_responses_transport_does_not_clamp_max_tokens():
    """Responses transport should preserve the SDK-requested max_tokens."""
    bridge = LiteLLMAnthropicBridge(
        ModelConfig(
            model="openai/gpt-4o-mini",
            api_key="sk-openai",
            base_url="https://api.openai.com/v1",
            max_tokens=16384,
        )
    )

    kwargs = bridge._build_kwargs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32000,
        }
    )

    assert kwargs["max_tokens"] == 32000


def test_deepseek_v4_reasoning_patch_uses_non_empty_placeholder():
    """DeepSeek V4 gateways reject missing or empty reasoning_content in thinking mode."""
    payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "已深度思考\nanswer", "reasoning_content": ""},
            {"role": "assistant", "content": "answer", "reasoning_content": None},
        ]
    }

    patched = patch_deepseek_reasoning(payload, "deepseek-v4-pro")

    assistant_messages = [m for m in patched["messages"] if m["role"] == "assistant"]
    assert all(m["reasoning_content"] == "已深度思考" for m in assistant_messages)


def test_agent_with_sub_agents():
    """Agent with sub-agents."""
    child = Agent(name="child", description="A child agent", tools=["Read"])
    parent = Agent(
        name="parent",
        instruction="You are a parent agent.",
        tools=["Bash"],
        sub_agents=[child],
    )
    assert len(parent.sub_agents) == 1
    assert parent.sub_agents[0].name == "child"


def test_agent_dynamic_instruction():
    """Agent with callable instruction."""

    def build_instruction(ctx):
        return f"Language: {ctx.params.get('lang', 'Python')}"

    agent = Agent(name="dynamic", instruction=build_instruction, tools=["Read"])
    ctx = RunContext(params={"lang": "Rust"})
    result = agent.resolve_instruction(ctx)
    assert result == "Language: Rust"


def test_agent_subclass():
    """Agent subclass pattern."""

    class MyAgent(Agent):
        def __init__(self):
            super().__init__(
                name="custom",
                instruction=self._build,
                tools=["Read"],
            )

        def _build(self, ctx):
            return "Custom instruction"

    agent = MyAgent()
    assert agent.name == "custom"
    assert agent.resolve_instruction(RunContext()) == "Custom instruction"


def test_types():
    """Type construction."""
    ctx = RunContext(
        prompt="Test",
        model="claude-opus-4-6",
        params={"key": "value"},
        workspace=WorkspaceConfig(enabled=True),
        git=GitConfig(
            repo_url="https://example.com/repo.git",
            clone=True,
        ),
    )
    assert ctx.prompt == "Test"
    assert ctx.model == "claude-opus-4-6"
    assert ctx.git.clone is True
    assert ctx.git.repo_url == "https://example.com/repo.git"
    assert len(ctx.task_id) == 12

    result = AgentResult(
        task_id="abc123",
        agent_type="test",
        status=TaskStatus.COMPLETED,
    )
    assert result.status == "completed"
    assert result.output_text == ""

    model = ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
    assert model.max_tokens == 16384

    sandbox = SandboxOptions(enabled=True)
    assert sandbox.enabled is True
    assert sandbox.deny_read == ["~/"]


def test_runner_config_from_env(monkeypatch):
    """RunnerConfig.from_env()."""
    monkeypatch.setenv("PLATFORM_WORKSPACE_ROOT", "/tmp/cckit-env-workspaces")
    cfg = RunnerConfig.from_env()
    assert cfg.default_model.model  # should have a default
    assert cfg.workspace_root == Path("/tmp/cckit-env-workspaces")
    assert cfg.max_concurrent_agents > 0


def test_run_context_model_overrides_runner_default():
    """RunContext.model overrides only the model name for a single run."""
    runner = Runner(
        config=RunnerConfig(
            default_model=ModelConfig(
                model="anthropic/claude-sonnet-4-6",
                api_key="sk-test",
                base_url="https://proxy.test",
            ),
        )
    )
    agent = Agent(name="override-model", tools=["Read"])
    resolved = runner._resolve_model(agent, RunContext(model="claude-opus-4-6"))
    assert resolved.model == "claude-opus-4-6"
    assert resolved.api_key == "sk-test"
    assert resolved.base_url == "https://proxy.test"


def test_build_options_bridge_env_overrides_ctx_env():
    """Bridge transport settings must win over conflicting ctx.env Anthropic vars."""
    runner = Runner(
        config=RunnerConfig(
            default_model=ModelConfig(model="anthropic/claude-sonnet-4-6"),
        )
    )
    agent = Agent(name="bridge-env", tools=["Read"])
    ctx = RunContext(
        env={
            "ANTHROPIC_API_KEY": "sk-wrong",
            "ANTHROPIC_AUTH_TOKEN": "sk-wrong",
            "ANTHROPIC_BASE_URL": "https://wrong.example.com",
            "OTHER_ENV": "ok",
        }
    )
    model = ModelConfig(
        model="openai/gpt-4o-mini",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
    )
    prepared = PreparedModelEndpoint(
        model="openai/gpt-4o-mini",
        api_key="cckit-bridge",
        base_url="http://127.0.0.1:41001",
    )

    opts = runner._build_options(
        agent,
        ctx,
        model,
        prepared,
        SandboxOptions(),
        None,
        "",
        RunState("bridgeenv"),
    )

    assert opts.env["ANTHROPIC_API_KEY"] == "cckit-bridge"
    assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "cckit-bridge"
    assert opts.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:41001"
    assert opts.env["OTHER_ENV"] == "ok"


def test_middleware():
    """Middleware instantiation."""
    r = RetryMiddleware(max_retries=5, base_delay=0.5)
    assert r.max_retries == 5

    c = ConcurrencyMiddleware(max_concurrent=10)
    assert c._limit == 10

    _l = LoggingMiddleware()
    assert _l is not None


@pytest.mark.anyio
async def test_workspace():
    """WorkspaceManager lifecycle."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = WorkspaceManager(root=Path(tmp))
        ws = await mgr.create("smoke-test")
        assert ws.exists()
        assert ws.is_dir()

        # Suspend
        await mgr.suspend(ws)
        assert ws.exists()

        # Resume
        resumed, was_recreated = await mgr.resume(ws)
        assert resumed == ws
        assert not was_recreated

        # Cleanup
        await mgr.cleanup(ws)
        assert not ws.exists()


def test_skill_validation():
    """SkillProvisioner name validation."""
    provisioner = SkillProvisioner(skills_dir=Path("/tmp/nonexistent-skills"))

    # Valid names
    for name in ["my-skill", "a", "skill-123", "abc"]:
        provisioner._validate_skill_name(name)  # should not raise

    # Invalid names
    for name in ["", "../hack", "UPPERCASE", "has spaces", "-leading-dash"]:
        with pytest.raises(SkillError):
            provisioner._validate_skill_name(name)


def test_exceptions():
    """Exception hierarchy."""
    for exc_cls in [
        AgentExecutionError,
        WorkspaceError,
        GitOperationError,
        GitLabAPIError,
        SkillError,
    ]:
        exc = exc_cls("test message", detail="detail info")
        assert isinstance(exc, CckitError)
        assert str(exc) == "test message\ndetail info"
        assert exc.detail == "detail info"


def test_sandbox_config():
    """SandboxConfigBuilder."""
    # Disabled
    builder = SandboxConfigBuilder(enabled=False)
    assert builder.build() is None

    # Enabled with workspace
    builder = SandboxConfigBuilder(
        enabled=True,
        allowed_domains=["api.example.com"],
    )
    with tempfile.TemporaryDirectory() as ws_dir:
        ws_path = Path(ws_dir)
        cfg = json.loads(builder.build(workspace_dir=ws_path))
        sandbox = cfg["sandbox"]
        fs = sandbox["filesystem"]
        net = sandbox["network"]
        assert sandbox["enabled"] is True
        assert str(ws_path) in fs["allowRead"]
        assert str(ws_path) in fs["allowWrite"]
        assert "api.example.com" in net["allowedDomains"]


def test_gitlab_client():
    """GitLabClient credential requirements."""
    # Missing url
    with pytest.raises(GitLabAPIError):
        GitLabClient(url="", token="glpat-xxx")

    # Missing token
    with pytest.raises(GitLabAPIError):
        GitLabClient(url="https://gitlab.com", token="")

    # Valid construction
    client = GitLabClient(url="https://gitlab.com", token="glpat-xxx")
    assert client._url == "https://gitlab.com"
    assert client._token == "glpat-xxx"


# ---------------------------------------------------------------------------
# StreamResult tests
# ---------------------------------------------------------------------------


def test_stream_result_basic():
    """StreamResult wraps an async iterator and exposes .result."""
    from cckit.types import _ResultHolder

    async def _fake_events():
        yield "hello"
        yield " world"

    holder = _ResultHolder()
    holder.result = AgentResult(
        task_id="t1", agent_type="test", status=TaskStatus.COMPLETED,
    )
    stream = StreamResult(_fake_events(), holder)

    import asyncio

    async def _consume():
        texts = []
        async for message in stream:
            texts.append(message)
        return texts

    texts = asyncio.run(_consume())
    assert texts == ["hello", " world"]
    assert stream.result is not None
    assert stream.result.task_id == "t1"


def test_stream_result_identity():
    """StreamResult.result is the exact same object as the holder's result.

    This is the core guarantee: on_after mutations are visible to the caller.
    """
    from cckit.types import _ResultHolder

    async def _no_events():
        if False:
            yield None

    holder = _ResultHolder()
    result_obj = AgentResult(
        task_id="t2", agent_type="test", status=TaskStatus.COMPLETED,
    )
    holder.result = result_obj

    # Simulate on_after writing extra data
    result_obj.extra["mr_url"] = "https://gitlab.com/mr/42"

    stream = StreamResult(_no_events(), holder)

    import asyncio
    asyncio.run(_drain(stream))

    # The key assertion: same object identity
    assert stream.result is result_obj
    assert stream.result.extra["mr_url"] == "https://gitlab.com/mr/42"


def test_stream_result_none_before_consumption():
    """StreamResult.result is None before the holder has a result."""
    from cckit.types import _ResultHolder

    async def _no_events():
        if False:
            yield None

    holder = _ResultHolder()
    stream = StreamResult(_no_events(), holder)
    assert stream.result is None


async def _drain(stream):
    """Helper: consume all events from a StreamResult."""
    async for _ in stream:
        pass


# ---------------------------------------------------------------------------
# Git operations: public API tests
# ---------------------------------------------------------------------------


def test_git_run_git_is_public():
    """run_git is public and _run_git is a backward-compat alias."""
    from cckit.git import operations as git_ops

    assert hasattr(git_ops, "run_git")
    assert hasattr(git_ops, "_run_git")
    assert git_ops._run_git is git_ops.run_git


def test_git_status_and_diff_exist():
    """status() and diff() are importable from the git module."""
    from cckit.git import operations as git_ops

    assert callable(git_ops.status)
    assert callable(git_ops.diff)


def test_git_module_exports():
    """All public functions are exported from cckit.git."""
    from cckit import git

    for name in ["run_git", "clone", "create_branch", "add_all",
                  "commit", "push", "status", "diff"]:
        assert hasattr(git, name), f"cckit.git missing export: {name}"


# ---------------------------------------------------------------------------
# GitConfig tests
# ---------------------------------------------------------------------------


def test_git_config_basic():
    """GitConfig construction and defaults."""
    cfg = GitConfig(repo_url="https://gitlab.com/team/project.git", clone=True)
    assert cfg.repo_url == "https://gitlab.com/team/project.git"
    assert cfg.clone is True
    assert cfg.depth == 1
    assert cfg.branch == ""
    assert cfg.token == ""
    assert cfg.auth_env == {}


def test_git_config_build_git_env_with_token():
    """build_git_env() produces an askpass helper without leaking the token."""
    cfg = GitConfig(
        repo_url="https://gitlab.com/team/project.git",
        token="glpat-secret",
    )
    env = cfg.build_git_env()
    assert "GIT_ASKPASS" in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_PASSWORD" not in env
    assert "glpat-secret" not in env.values()


def test_git_config_build_git_env_empty():
    """build_git_env() returns empty dict when no credentials."""
    cfg = GitConfig(repo_url="https://gitlab.com/team/project.git")
    assert cfg.build_git_env() == {}


def test_git_config_auth_env_overrides_token():
    """auth_env can override/supplement token-derived values."""
    cfg = GitConfig(
        repo_url="https://gitlab.com/team/project.git",
        token="glpat-secret",
        auth_env={"GIT_ASKPASS": "/custom/askpass.sh"},
    )
    env = cfg.build_git_env()
    # auth_env takes precedence over token-derived GIT_ASKPASS
    assert env["GIT_ASKPASS"] == "/custom/askpass.sh"
    # token is still kept out of the subprocess environment
    assert "GIT_PASSWORD" not in env
    assert "glpat-secret" not in env.values()


def test_run_context_credential_isolation():
    """ctx.env and git credentials are completely separate namespaces."""
    ctx = RunContext(
        prompt="Test",
        env={"ANTHROPIC_API_KEY": "sk-safe", "MY_FLAG": "true"},
        git=GitConfig(
            repo_url="https://gitlab.com/team/project.git",
            token="glpat-secret-should-not-leak",
            clone=True,
        ),
    )
    # ctx.env has no git credentials
    assert "GIT_PASSWORD" not in ctx.env
    assert "glpat-secret-should-not-leak" not in ctx.env.values()

    # git env has no agent env
    git_env = ctx.git.build_git_env()
    assert "ANTHROPIC_API_KEY" not in git_env
    assert "MY_FLAG" not in git_env
    assert "GIT_PASSWORD" not in git_env
    assert "glpat-secret-should-not-leak" not in git_env.values()


def test_run_context_backward_compat():
    """Deprecated git_repo_url/git_branch still work via _resolved_git()."""
    ctx = RunContext(
        prompt="Test",
        git_repo_url="https://example.com/old-style.git",
        git_branch="develop",
    )
    resolved = ctx.resolved_git()
    assert resolved.repo_url == "https://example.com/old-style.git"
    assert resolved.branch == "develop"
    assert resolved.clone is True  # auto-upgraded when git_repo_url is set


def test_run_context_new_git_takes_precedence():
    """New git field takes precedence over deprecated fields."""
    ctx = RunContext(
        prompt="Test",
        git=GitConfig(repo_url="https://new.com/repo.git", branch="main", clone=True),
        git_repo_url="https://old.com/repo.git",  # should be ignored
        git_branch="old-branch",  # should be ignored
    )
    resolved = ctx.resolved_git()
    assert resolved.repo_url == "https://new.com/repo.git"
    assert resolved.branch == "main"
