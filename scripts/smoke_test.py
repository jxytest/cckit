#!/usr/bin/env python3
"""Smoke test — verify the agent core infrastructure wires up correctly.

Run from the repo root:
    cd new && python scripts/smoke_test.py

This script does NOT require a real ANTHROPIC_API_KEY.  It validates:
  1. All modules import without error
  2. Agent registry auto-registration works
  3. AgentConfig / ExecutionContext / AgentResult round-trip
  4. WorkspaceManager can create and clean up directories
  5. SandboxConfig produces the expected dict
  6. FixAgent.build_prompt() works
  7. Middleware classes instantiate and chain correctly
  8. required_params validation works
  9. datetime uses timezone-aware UTC
 10. on_error lifecycle hook
 11. GitLabClient credential requirements
 12. AgentExecutor clone semaphore + skill provisioner DI
 13. SkillProvisioner: scan, provision, security validation, cleanup
 14. AgentConfig.skills field
 15. skills + needs_workspace validation in executor
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import UTC
from pathlib import Path

# Ensure src/ is on the path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def test_imports() -> None:
    print("1. Testing imports...")
    from core import (  # noqa: F401
        AgentConfig,
        AgentEvent,
        AgentEventType,
        AgentExecutor,
        AgentMiddleware,
        AgentResult,
        BaseAgent,
        ConcurrencyMiddleware,
        ExecutionContext,
        LoggingMiddleware,
        RetryMiddleware,
        SubAgentConfig,
        TaskStatus,
        agent_registry,
    )
    from core.agent.streaming.collector import StreamCollector  # noqa: F401
    from core.config import (  # noqa: F401
        anthropic_settings,
        platform_settings,
        sandbox_settings,
    )
    from core.exceptions import (  # noqa: F401
        AgentExecutionError,
        AgentNotFoundError,
        CoreError,
        GitLabAPIError,
        GitOperationError,
        SandboxConfigError,
        SkillError,
        WorkspaceError,
    )
    from core.agent.skill import SkillProvisioner  # noqa: F401
    from core.models import CustomModel  # noqa: F401
    print("   [OK] All core modules imported successfully")


def test_registry() -> None:
    print("2. Testing agent registry...")
    # Trigger auto-registration
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    assert "fix" in agent_registry, "FixAgent should be registered"
    agent_cls = agent_registry.get("fix")
    agent = agent_cls()
    cfg = agent.config()
    assert cfg.agent_type == "fix"
    assert cfg.display_name == "Locator Fix Agent"
    assert len(cfg.sub_agents) == 1
    assert cfg.sub_agents[0].name == "log-analyzer"
    print(f"   [OK] FixAgent registered: {agent}")
    print(f"   [OK] Registry contains {len(agent_registry)} agent(s): "
          f"{list(agent_registry.list_agents().keys())}")


def test_schemas() -> None:
    print("3. Testing schemas...")
    from core.agent.schemas import (
        AgentConfig,
        AgentEvent,
        AgentEventType,
        AgentResult,
        ExecutionContext,
        TaskStatus,
    )

    ctx = ExecutionContext(
        extra_params={"test_file": "test_login.py", "error_log": "NoSuchElement"},
        git_repo_url="https://gitlab.com/team/repo.git",
    )
    assert ctx.task_id  # auto-generated
    assert ctx.workspace_dir is None

    result = AgentResult(
        task_id=ctx.task_id,
        agent_type="fix",
        status=TaskStatus.COMPLETED,
        result_text="Fixed the locator",
    )
    assert not result.is_error
    # Round-trip via JSON
    json_str = result.model_dump_json()
    restored = AgentResult.model_validate_json(json_str)
    assert restored.task_id == result.task_id
    print(f"   [OK] ExecutionContext task_id={ctx.task_id}")
    print(f"   [OK] AgentResult JSON round-trip OK ({len(json_str)} bytes)")

    # Verify timezone-aware datetime
    event = AgentEvent(
        event_type=AgentEventType.RESULT,
        task_id=ctx.task_id,
        text="test",
    )
    assert event.timestamp.tzinfo is not None, "timestamp should be timezone-aware"
    assert event.timestamp.tzinfo == UTC, "timestamp should be UTC"
    print("   [OK] AgentEvent.timestamp is timezone-aware UTC")

    # Verify max_turns on AgentConfig
    cfg = AgentConfig(agent_type="test")
    assert cfg.max_turns == 0, "max_turns defaults to 0 (use global setting)"
    print("   [OK] AgentConfig.max_turns defaults to 0")


async def test_workspace() -> None:
    print("4. Testing WorkspaceManager...")
    from core.agent.sandbox.workspace import WorkspaceManager

    with tempfile.TemporaryDirectory() as tmp:
        wm = WorkspaceManager(root=Path(tmp))
        ws = await wm.create("smoke_test")
        assert ws.exists()
        print(f"   [OK] Created workspace: {ws}")

        await wm.cleanup(ws)
        assert not ws.exists()
        print("   [OK] Cleaned up workspace")


def test_sandbox_config() -> None:
    print("5. Testing SandboxConfig...")
    from core.agent.sandbox.config import SandboxConfig

    # Disabled -> empty dict
    sc = SandboxConfig(enabled=False)
    assert sc.build() == {}

    # Enabled with workspace
    ws_path = Path(tempfile.gettempdir()) / "ws_test"
    sc_on = SandboxConfig(enabled=True, network_hosts=["gitlab.com"])
    result = sc_on.build(workspace_dir=ws_path)
    assert result["enabled"] is True
    assert str(ws_path) in result["allowRead"]
    assert str(ws_path) in result["allowWrite"]
    assert "gitlab.com" in result["networkAllowedHosts"]
    print(f"   [OK] Sandbox config (enabled): {result}")


def test_build_prompt() -> None:
    print("6. Testing FixAgent.build_prompt()...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("fix")
    ctx = ExecutionContext(
        extra_params={
            "test_file": "tests/test_login.py",
            "error_log": "NoSuchElementException: id=login-btn",
            "project_id": "42",
        },
    )
    prompt = agent.build_prompt(ctx)
    assert "test_login.py" in prompt
    assert "NoSuchElementException" in prompt
    print(f"   [OK] Prompt generated ({len(prompt)} chars)")


def test_middleware() -> None:
    print("7. Testing middleware...")
    from core.agent.middleware import (
        ConcurrencyMiddleware,
        LoggingMiddleware,
        RetryMiddleware,
    )

    # Verify instantiation with defaults
    retry = RetryMiddleware()
    assert retry.max_retries == 3
    assert retry.base_delay == 1.0

    retry_custom = RetryMiddleware(max_retries=5, base_delay=2.0, max_delay=60.0)
    assert retry_custom.max_retries == 5
    assert retry_custom.max_delay == 60.0

    concurrency = ConcurrencyMiddleware()
    assert concurrency._limit > 0

    concurrency_custom = ConcurrencyMiddleware(max_concurrent=10)
    assert concurrency_custom._limit == 10

    logging_mw = LoggingMiddleware()
    assert logging_mw is not None

    print("   [OK] RetryMiddleware(max_retries=3, base_delay=1.0)")
    print("   [OK] ConcurrencyMiddleware(limit from settings)")
    print("   [OK] LoggingMiddleware instantiated")


def test_required_params() -> None:
    print("8. Testing required_params...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    agent = agent_registry.get_instance("fix")
    params = agent.required_params()
    assert "test_file" in params, "FixAgent should require test_file"
    assert "error_log" in params, "FixAgent should require error_log"
    print(f"   [OK] FixAgent.required_params() = {params}")


def test_configurable_settings() -> None:
    print("9. Testing configurable SDK settings...")
    from core.config import anthropic_settings

    assert hasattr(anthropic_settings, "max_turns"), "max_turns should exist"
    assert hasattr(anthropic_settings, "permission_mode"), "permission_mode should exist"
    assert anthropic_settings.max_turns == 50
    assert anthropic_settings.permission_mode == "bypassPermissions"
    print(f"   [OK] max_turns={anthropic_settings.max_turns}")
    print(f"   [OK] permission_mode={anthropic_settings.permission_mode}")


def test_on_error_hook() -> None:
    print("10. Testing on_error lifecycle hook...")
    from core.agent.base import BaseAgent
    from core.agent.schemas import AgentConfig, ExecutionContext

    class TestAgent(BaseAgent):
        def config(self) -> AgentConfig:
            return AgentConfig(agent_type="test_error")

        def build_prompt(self, ctx: ExecutionContext) -> str:
            return "test"

    agent = TestAgent()
    # Verify on_error exists and is callable
    assert hasattr(agent, "on_error")
    assert callable(agent.on_error)
    print("   [OK] BaseAgent.on_error() hook is available")


def test_gitlab_client_requires_credentials() -> None:
    print("11. Testing GitLabClient requires explicit credentials...")
    from core.agent.git.gitlab_client import GitLabClient
    from core.exceptions import GitLabAPIError

    # Missing url
    try:
        GitLabClient(url="", token="glpat-xxx")
        raise AssertionError("Should have raised GitLabAPIError for empty url")
    except GitLabAPIError:
        pass

    # Missing token
    try:
        GitLabClient(url="https://gitlab.com", token="")
        raise AssertionError("Should have raised GitLabAPIError for empty token")
    except GitLabAPIError:
        pass

    # Valid construction
    client = GitLabClient(url="https://gitlab.com", token="glpat-xxx")
    assert client._url == "https://gitlab.com"
    assert client._token == "glpat-xxx"
    assert client._default_branch == "main"

    # No gitlab_settings in core.config
    import core.config as cfg_mod
    assert not hasattr(cfg_mod, "gitlab_settings"), "gitlab_settings should be removed"
    print("   [OK] GitLabClient requires url and token (no global fallback)")


def test_executor_clone_semaphore() -> None:
    print("12. Testing AgentExecutor clone semaphore...")
    from core.agent.executor import AgentExecutor

    # Default: reads from platform_settings.max_concurrent_agents
    executor = AgentExecutor()
    assert hasattr(executor, "_clone_semaphore")
    assert executor._clone_semaphore._value > 0

    # Custom limit
    executor_custom = AgentExecutor(max_concurrent_clones=3)
    assert executor_custom._clone_semaphore._value == 3

    # Verify skill_provisioner is injected by default
    assert hasattr(executor, "_skill_provisioner")
    print("   [OK] AgentExecutor._clone_semaphore works")
    print("   [OK] AgentExecutor._skill_provisioner injected")


async def test_skill_provisioner() -> None:
    print("13. Testing SkillProvisioner...")
    from core.agent.skill import SkillProvisioner
    from core.exceptions import SkillError

    with tempfile.TemporaryDirectory() as tmp:
        skills_src = Path(tmp) / "skills_source"
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()

        # -- list_available: empty when dir doesn't exist --
        provisioner = SkillProvisioner(skills_dir=skills_src)
        assert provisioner.list_available() == []
        print("   [OK] list_available() returns [] for missing dir")

        # -- create a valid skill --
        skills_src.mkdir()
        skill_dir = skills_src / "my-test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-test-skill\n"
            "description: A test skill\n---\n\nYou are a test expert.\n"
        )

        available = provisioner.list_available()
        assert "my-test-skill" in available
        print(f"   [OK] list_available() = {available}")

        # -- provision into workspace --
        deployed = await provisioner.provision(["my-test-skill"], workspace)
        assert deployed == ["my-test-skill"]
        target = workspace / ".claude" / "skills" / "my-test-skill" / "SKILL.md"
        assert target.is_file(), f"SKILL.md should exist at {target}"
        print(f"   [OK] Provisioned skill to {target.parent}")

        # -- provision clears pre-existing .claude/ --
        rogue_file = workspace / ".claude" / "settings.json"
        rogue_file.write_text("{}")
        await provisioner.provision(["my-test-skill"], workspace)
        assert not rogue_file.exists(), ".claude/ should be purged before provision"
        print("   [OK] Pre-existing .claude/ purged on provision")

        # -- validate_skill_name rejects bad names --
        for bad_name in ["../etc", "UPPER", "with spaces", "a" * 100]:
            try:
                await provisioner.provision([bad_name], workspace)
                raise AssertionError(f"Should have rejected skill name: {bad_name!r}")
            except SkillError:
                pass
        print("   [OK] Invalid skill names rejected (path traversal, uppercase, etc.)")

        # -- missing skill raises SkillError --
        try:
            await provisioner.provision(["nonexistent-skill"], workspace)
            raise AssertionError("Should have raised SkillError for missing skill")
        except SkillError:
            pass
        print("   [OK] Missing skill raises SkillError")

        # -- provision with empty list is a no-op --
        result = await provisioner.provision([], workspace)
        assert result == []
        print("   [OK] Empty skill list is a no-op")


def test_skills_on_agent_config() -> None:
    print("14. Testing skills field on AgentConfig...")
    from core.agent.schemas import AgentConfig

    # Default: empty
    cfg = AgentConfig(agent_type="test")
    assert cfg.skills == []
    print("   [OK] AgentConfig.skills defaults to []")

    # With skills declared
    cfg_with = AgentConfig(agent_type="test", skills=["playwright-testing", "api-testing"])
    assert cfg_with.skills == ["playwright-testing", "api-testing"]
    print(f"   [OK] AgentConfig.skills = {cfg_with.skills}")


def test_skills_validation_in_executor() -> None:
    print("15. Testing skills + needs_workspace validation...")
    from core.agent.base import BaseAgent
    from core.agent.executor import AgentExecutor
    from core.agent.schemas import AgentConfig, ExecutionContext

    class NoWorkspaceSkillAgent(BaseAgent):
        def config(self) -> AgentConfig:
            return AgentConfig(
                agent_type="bad_skill_agent",
                skills=["some-skill"],
                needs_workspace=False,  # conflict!
            )

        def build_prompt(self, ctx: ExecutionContext) -> str:
            return "test"

    agent = NoWorkspaceSkillAgent()
    ctx = ExecutionContext()
    missing = AgentExecutor._validate_context(agent, ctx)
    assert any("needs_workspace" in m for m in missing), (
        "Should flag needs_workspace conflict"
    )
    print("   [OK] skills + needs_workspace=False rejected by _validate_context")


async def main() -> None:
    print("=" * 60)
    print("Agent Core Infrastructure -- Smoke Test")
    print("=" * 60)
    print()

    test_imports()
    test_registry()
    test_schemas()
    await test_workspace()
    test_sandbox_config()
    test_build_prompt()
    test_middleware()
    test_required_params()
    test_configurable_settings()
    test_on_error_hook()
    test_gitlab_client_requires_credentials()
    test_executor_clone_semaphore()
    await test_skill_provisioner()
    test_skills_on_agent_config()
    test_skills_validation_in_executor()

    print()
    print("=" * 60)
    print("All smoke tests passed! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
