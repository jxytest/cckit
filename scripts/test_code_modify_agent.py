#!/usr/bin/env python3
"""Test script for CodeModifyAgent.

Run from the repo root:
    cd new && python scripts/test_code_modify_agent.py

This script does NOT require a real ANTHROPIC_API_KEY.  It validates:
  1. CodeModifyAgent is auto-registered in agent_registry
  2. AgentConfig fields are correct (agent_type, display_name, tools, sub_agents, etc.)
  3. required_params() declares the expected mandatory keys
  4. build_prompt() constructs a correct prompt for all param combinations
  5. on_error lifecycle hook is available and callable
  6. Agent can be instantiated multiple times (stateless)
  7. ExecutionContext round-trips correctly for CodeModifyAgent payloads
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASSED: list[str] = []
FAILED: list[str] = []


def ok(msg: str) -> None:
    print(f"   [OK] {msg}")
    PASSED.append(msg)


def fail(msg: str) -> None:
    print(f"   [FAIL] {msg}")
    FAILED.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------

def test_registration() -> None:
    section("1. Testing CodeModifyAgent registration...")
    import agents  # noqa: F401 — triggers auto-registration
    from core.agent.registry import agent_registry

    if "code_modify" in agent_registry:
        ok("agent_type 'code_modify' registered in agent_registry")
    else:
        fail(f"'code_modify' NOT found; registered agents: {list(agent_registry.list_agents())}")
        return

    agent_cls = agent_registry.get("code_modify")
    agent = agent_cls()
    if agent is not None:
        ok(f"Instantiated: {agent!r}")
    else:
        fail("get_instance returned None")


# ---------------------------------------------------------------------------
# 2. AgentConfig validation
# ---------------------------------------------------------------------------

def test_config() -> None:
    section("2. Testing CodeModifyAgent.config()...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    agent = agent_registry.get_instance("code_modify")
    cfg = agent.config()

    assert cfg.agent_type == "code_modify", f"Expected 'code_modify', got {cfg.agent_type!r}"
    ok(f"agent_type = {cfg.agent_type!r}")

    assert cfg.display_name, "display_name should not be empty"
    ok(f"display_name = {cfg.display_name!r}")

    assert cfg.system_prompt, "system_prompt should not be empty"
    ok(f"system_prompt length = {len(cfg.system_prompt)} chars")

    # Must have essential tools
    required_tools = {"Read", "Write", "Edit", "Glob", "Grep"}
    missing_tools = required_tools - set(cfg.allowed_tools)
    assert not missing_tools, f"Missing tools: {missing_tools}"
    ok(f"allowed_tools contains required set: {sorted(cfg.allowed_tools)}")

    # Must have at least one sub-agent
    assert len(cfg.sub_agents) >= 1, "Expected at least one sub_agent"
    sub = cfg.sub_agents[0]
    assert sub.name, "sub_agent name should not be empty"
    assert sub.description, "sub_agent description should not be empty"
    ok(f"sub_agents[0].name = {sub.name!r}")

    # needs_workspace must be True (we write files)
    assert cfg.needs_workspace is True, "needs_workspace should be True"
    ok("needs_workspace = True")

    # needs_git_clone must be True
    assert cfg.needs_git_clone is True, "needs_git_clone should be True"
    ok("needs_git_clone = True")


# ---------------------------------------------------------------------------
# 3. required_params
# ---------------------------------------------------------------------------

def test_required_params() -> None:
    section("3. Testing required_params()...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    agent = agent_registry.get_instance("code_modify")
    params = agent.required_params()

    assert isinstance(params, list), "required_params() should return a list"
    assert "modification_request" in params, "Should require 'modification_request'"
    assert "target_path" in params, "Should require 'target_path'"
    ok(f"required_params() = {params}")


# ---------------------------------------------------------------------------
# 4. build_prompt — various param combinations
# ---------------------------------------------------------------------------

def test_build_prompt_basic() -> None:
    section("4a. Testing build_prompt() — basic params...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("code_modify")
    ctx = ExecutionContext(
        extra_params={
            "modification_request": "Rename function `foo` to `bar` everywhere.",
            "target_path": "src/utils.py",
        },
    )
    prompt = agent.build_prompt(ctx)

    assert "Rename function" in prompt, "Prompt should contain modification_request text"
    assert "src/utils.py" in prompt, "Prompt should contain target_path"
    ok(f"Prompt generated ({len(prompt)} chars)")
    ok("modification_request present in prompt")
    ok("target_path present in prompt")


def test_build_prompt_with_context_hint() -> None:
    section("4b. Testing build_prompt() — with context_hint...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("code_modify")
    ctx = ExecutionContext(
        extra_params={
            "modification_request": "Add type hints to all public functions.",
            "target_path": "src/",
            "context_hint": "This project uses Python 3.12 and Pydantic v2.",
        },
    )
    prompt = agent.build_prompt(ctx)

    assert "Python 3.12" in prompt, "Prompt should contain context_hint text"
    ok("context_hint included in prompt")


def test_build_prompt_with_project_id() -> None:
    section("4c. Testing build_prompt() — with project_id (MCP tool hint)...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("code_modify")
    ctx = ExecutionContext(
        extra_params={
            "modification_request": "Remove all debug print statements.",
            "target_path": "src/",
            "project_id": "99",
        },
    )
    prompt = agent.build_prompt(ctx)

    assert "99" in prompt, "Prompt should contain project_id for MCP tool"
    assert "report_progress" in prompt, "Prompt should mention report_progress tool"
    ok("project_id and report_progress hint present in prompt")


def test_build_prompt_no_optional_fields() -> None:
    section("4d. Testing build_prompt() — only required params (no crash)...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("code_modify")
    ctx = ExecutionContext(
        extra_params={
            "modification_request": "Delete unused imports.",
            "target_path": "app/main.py",
        },
    )
    prompt = agent.build_prompt(ctx)

    assert prompt, "Prompt should not be empty"
    # Optional fields absent → no crash, no placeholder leakage
    assert "None" not in prompt, "Prompt should not contain literal 'None'"
    ok(f"No crash with only required params, prompt length={len(prompt)}")


# ---------------------------------------------------------------------------
# 5. on_error hook availability
# ---------------------------------------------------------------------------

def test_on_error_hook() -> None:
    section("5. Testing on_error lifecycle hook...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    agent = agent_registry.get_instance("code_modify")
    assert hasattr(agent, "on_error"), "BaseAgent.on_error should exist"
    assert callable(agent.on_error), "on_error should be callable"
    ok("on_error hook is available and callable")


# ---------------------------------------------------------------------------
# 6. Stateless — multiple instantiations
# ---------------------------------------------------------------------------

def test_multiple_instances() -> None:
    section("6. Testing multiple independent instances (stateless)...")
    import agents  # noqa: F401
    from core.agent.registry import agent_registry

    a1 = agent_registry.get_instance("code_modify")
    a2 = agent_registry.get_instance("code_modify")

    assert a1 is not a2, "Each call should return a fresh instance"
    assert a1.config().agent_type == a2.config().agent_type == "code_modify"
    ok("Two independent instances with identical config — no shared state")


# ---------------------------------------------------------------------------
# 7. ExecutionContext round-trip with CodeModifyAgent payload
# ---------------------------------------------------------------------------

def test_execution_context_round_trip() -> None:
    section("7. Testing ExecutionContext JSON round-trip for CodeModifyAgent payload...")
    from core.agent.schemas import ExecutionContext

    ctx = ExecutionContext(
        git_repo_url="https://gitlab.com/team/project.git",
        git_branch="feature/add-types",
        extra_params={
            "modification_request": "Add return type annotations to all functions.",
            "target_path": "src/core/",
            "context_hint": "Codebase uses Python 3.12.",
            "project_id": "42",
            "gitlab_url": "https://gitlab.example.com",
            "gitlab_token": "glpat-xxx",
            "gitlab_project_id": 99,
        },
    )
    json_str = ctx.model_dump_json()
    restored = ExecutionContext.model_validate_json(json_str)

    assert restored.task_id == ctx.task_id
    assert restored.git_repo_url == ctx.git_repo_url
    assert restored.extra_params["modification_request"] == ctx.extra_params["modification_request"]
    assert restored.extra_params["target_path"] == ctx.extra_params["target_path"]
    ok(f"Round-trip OK ({len(json_str)} bytes), task_id={ctx.task_id}")


# ---------------------------------------------------------------------------
# 8. _validate_context via AgentExecutor
# ---------------------------------------------------------------------------

def test_validate_context_missing_required_params() -> None:
    section("8. Testing _validate_context catches missing required_params...")
    import agents  # noqa: F401
    from core.agent.executor import AgentExecutor
    from core.agent.registry import agent_registry
    from core.agent.schemas import ExecutionContext

    agent = agent_registry.get_instance("code_modify")

    # Neither modification_request nor target_path provided
    ctx_empty = ExecutionContext()
    missing = AgentExecutor._validate_context(agent, ctx_empty)
    assert any("modification_request" in m for m in missing), (
        f"Should flag missing 'modification_request'; got: {missing}"
    )
    assert any("target_path" in m for m in missing), (
        f"Should flag missing 'target_path'; got: {missing}"
    )
    ok(f"Missing both params detected: {missing}")

    # Only modification_request provided
    ctx_partial = ExecutionContext(
        extra_params={"modification_request": "Do something"}
    )
    missing_partial = AgentExecutor._validate_context(agent, ctx_partial)
    assert any("target_path" in m for m in missing_partial), (
        f"Should flag missing 'target_path'; got: {missing_partial}"
    )
    ok("Missing target_path flagged when only modification_request supplied")

    # Both required params provided — no validation errors for these
    ctx_valid = ExecutionContext(
        extra_params={
            "modification_request": "Add docstrings.",
            "target_path": "src/",
        },
    )
    missing_valid = AgentExecutor._validate_context(agent, ctx_valid)
    param_errors = [m for m in missing_valid if "modification_request" in m or "target_path" in m]
    assert not param_errors, f"Should not flag params when both provided; got: {param_errors}"
    ok("No param errors when both required params supplied")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("CodeModifyAgent — Unit Tests (no API key required)")
    print("=" * 60)

    test_registration()
    test_config()
    test_required_params()
    test_build_prompt_basic()
    test_build_prompt_with_context_hint()
    test_build_prompt_with_project_id()
    test_build_prompt_no_optional_fields()
    test_on_error_hook()
    test_multiple_instances()
    test_execution_context_round_trip()
    test_validate_context_missing_required_params()

    print()
    print("=" * 60)
    if FAILED:
        print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"  FAIL: {f}")
        sys.exit(1)
    else:
        print(f"All {len(PASSED)} tests passed! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
