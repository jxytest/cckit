"""Smoke test for cckit — no API key required.

Verifies:
1. All public imports work
2. Agent instantiation (simple + LiteLlm + sub-agents)
3. Type construction (ModelConfig, RunContext, RunnerConfig, etc.)
4. Middleware instantiation
5. WorkspaceManager lifecycle
6. SkillProvisioner validation
7. Exception hierarchy
8. RunnerConfig.from_env()
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cckit import (  # noqa: F401 — test_imports verifies all public API symbols
    Agent,
    AgentEvent,
    AgentEventType,
    AgentExecutionError,
    AgentResult,
    CckitError,
    ConcurrencyMiddleware,
    GitLabAPIError,
    GitOperationError,
    LiteLlm,
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
        model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-test"),
        instruction="Hello",
        tools=["Bash"],
    )
    assert agent.model_config is not None
    assert agent.model_config.model == "claude-sonnet-4-6"
    assert agent.model_config.api_key == "sk-test"


def test_agent_with_string_model():
    """Agent with string model shorthand."""
    agent = Agent(name="string-model", model="claude-opus-4-6", tools=["Read"])
    assert agent.model_config is not None
    assert agent.model_config.model == "claude-opus-4-6"


def test_agent_with_litellm():
    """Agent with LiteLlm bridge."""
    agent = Agent(
        name="litellm-test",
        model=LiteLlm(
            model="openai/gemini-3-pro",
            api_base="http://localhost:4000",
            api_key="sk-litellm",
        ),
        tools=["Read"],
    )
    assert agent.model_config is not None
    assert agent.model_config.model == "openai/gemini-3-pro"
    assert agent.model_config.base_url == "http://localhost:4000"
    assert agent.model_config.api_key == "sk-litellm"


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
        params={"key": "value"},
        workspace=WorkspaceConfig(enabled=True, git_clone=True),
        git_repo_url="https://example.com/repo.git",
    )
    assert ctx.prompt == "Test"
    assert ctx.workspace.git_clone is True
    assert len(ctx.task_id) == 12

    result = AgentResult(
        task_id="abc123",
        agent_type="test",
        status=TaskStatus.COMPLETED,
    )
    assert result.status == "completed"

    event = AgentEvent(
        event_type=AgentEventType.ASSISTANT_TEXT,
        text="Hello",
    )
    assert event.text == "Hello"

    model = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test")
    assert model.max_tokens == 16384

    sandbox = SandboxOptions(enabled=True)
    assert sandbox.workspace_root == Path("/tmp/cckit_workspaces")


def test_runner_config_from_env():
    """RunnerConfig.from_env()."""
    cfg = RunnerConfig.from_env()
    assert cfg.default_model.model  # should have a default
    assert cfg.max_concurrent_agents > 0


def test_middleware():
    """Middleware instantiation."""
    r = RetryMiddleware(max_retries=5, base_delay=0.5)
    assert r.max_retries == 5

    c = ConcurrencyMiddleware(max_concurrent=10)
    assert c._limit == 10

    _l = LoggingMiddleware()
    assert _l is not None


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
        resumed = await mgr.resume(ws)
        assert resumed == ws

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
        assert str(exc) == "test message"
        assert exc.detail == "detail info"


def test_sandbox_config():
    """SandboxConfigBuilder."""
    # Disabled
    builder = SandboxConfigBuilder(enabled=False)
    assert builder.build() == {}

    # Enabled with workspace
    builder = SandboxConfigBuilder(
        enabled=True,
        network_hosts=["api.example.com"],
    )
    with tempfile.TemporaryDirectory() as ws_dir:
        ws_path = Path(ws_dir)
        cfg = builder.build(workspace_dir=ws_path)
        assert cfg["enabled"] is True
        assert str(ws_path) in cfg["allowRead"]
        assert str(ws_path) in cfg["allowWrite"]
        assert "api.example.com" in cfg["networkAllowedHosts"]


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
        yield AgentEvent(event_type=AgentEventType.ASSISTANT_TEXT, text="hello")
        yield AgentEvent(event_type=AgentEventType.ASSISTANT_TEXT, text=" world")

    holder = _ResultHolder()
    holder.result = AgentResult(
        task_id="t1", agent_type="test", status=TaskStatus.COMPLETED,
    )
    stream = StreamResult(_fake_events(), holder)

    import asyncio

    async def _consume():
        texts = []
        async for event in stream:
            texts.append(event.text)
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
        return
        yield  # noqa: unreachable — makes it an async generator

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
        return
        yield  # noqa: unreachable

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
