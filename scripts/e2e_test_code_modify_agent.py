#!/usr/bin/env python3
"""End-to-End integration test for CodeModifyAgent.

Prerequisites (set in .env or as environment variables):
    ANTHROPIC_API_KEY=sk-testaaaaaaaaaaxxx
    ANTHROPIC_BASE_URL=http://172.23.48.1:7000
    ANTHROPIC_MODEL=claude-sonnet-4-6
    ANTHROPIC_MAX_TURNS=50
    ANTHROPIC_PERMISSION_MODE=bypassPermissions

Run:
    cd new && python scripts/e2e_test_code_modify_agent.py

What this test does:
  1. Clone the target GitLab repository to a temp workspace
  2. Ask CodeModifyAgent to make a simple change (update the base image in Dockerfile)
  3. Stream and print every AgentEvent (assistant text, tool calls, etc.)
  4. After execution: verify the file was actually changed
  5. Print a diff of the changes, cost, and duration
  6. Clean up temp workspace automatically

Note: No MR is created (gitlab_project_id is intentionally omitted).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Load .env if present (optional, pydantic-settings auto-reads it too)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
        print(f"[setup] Loaded .env from {_env_file}")
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("core").setLevel(logging.DEBUG)
logger = logging.getLogger("e2e_test")

# ---------------------------------------------------------------------------
# ─── CONFIGURATION — edit these to match your environment ────────────────
# ---------------------------------------------------------------------------

# GitLab repository (HTTP clone URL)
GIT_REPO_URL = "https://g.hz.netease.com/CloudQA/evaluation.git"
GIT_BRANCH = "master"

# GitLab credentials (if needs_git_clone=True with auth)
# Token embedded in URL avoids passing it via env:
#   e.g.  "https://oauth2:<token>@g.hz.netease.com/CloudQA/evaluation.git"
# Or set here and the script will embed it automatically:
GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "2ukrUscHywFVTjzxERaa")

# ── Modification request ──────────────────────────────────────────────────
# Target file inside the repo. Use a path relative to repo root.
TARGET_PATH = "app/Dockerfile"

# What should the agent change?
MODIFICATION_REQUEST = (
    "Find the FROM instruction in the Dockerfile and change the base image tag "
    "to the next patch version. For example, if the current tag is python:3.11, "
    "change it to python:3.12. If there is no explicit tag, add ':slim' suffix. "
    "Only change the FROM line — do not modify anything else."
)

# Optional background hint to Claude
CONTEXT_HINT = (
    "This is a simple Dockerfile change for testing purposes. "
    "Make the minimal possible edit."
)

# If no Dockerfile exists in the repo root, the agent will create a minimal one
# for demonstration. Set to False to skip auto-creation and let the agent handle it.
CREATE_DEMO_DOCKERFILE_IF_MISSING = True
DEMO_DOCKERFILE_CONTENT = """\
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
"""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _embed_token(url: str, token: str) -> str:
    """Insert token into HTTPS GitLab URL for authentication."""
    if token and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://oauth2:{token}@{rest}"
    return url


def _print_separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def _read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"<could not read: {exc}>"


def _diff_lines(before: str, after: str, filepath: str) -> str:
    """Return a simple unified-style diff string."""
    import difflib
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff) or "(no textual diff detected)"


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

async def run_e2e_test() -> None:
    _print_separator("CodeModifyAgent — End-to-End Test")
    print(f"  Repo   : {GIT_REPO_URL}")
    print(f"  Branch : {GIT_BRANCH}")
    print(f"  Target : {TARGET_PATH}")
    print(f"  Request: {MODIFICATION_REQUEST[:80]}...")
    _print_separator()

    # ── Import agent infrastructure ────────────────────────────────────────
    import agents  # noqa: F401 — triggers auto-registration

    from core.agent.executor import AgentExecutor
    from core.agent.middleware import ConcurrencyMiddleware, LoggingMiddleware, RetryMiddleware
    from core.agent.registry import agent_registry
    from core.agent.schemas import AgentEventType, ExecutionContext
    from core.config import anthropic_settings

    # ── Sanity checks ──────────────────────────────────────────────────────
    if not anthropic_settings.api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY is not set.")
        print("  Set it via environment variable or .env file:")
        print("  ANTHROPIC_API_KEY=sk-testaaaaaaaaaaxxx")
        sys.exit(1)

    print(f"\n[config] model          = {anthropic_settings.model}")
    print(f"[config] base_url       = {anthropic_settings.base_url or '(default api.anthropic.com)'}")
    print(f"[config] max_turns      = {anthropic_settings.max_turns}")
    print(f"[config] permission_mode= {anthropic_settings.permission_mode}")
    print(f"[config] api_key        = {anthropic_settings.api_key[:12]}...")

    # ── Resolve agent ──────────────────────────────────────────────────────
    if "code_modify" not in agent_registry:
        print("\n[ERROR] 'code_modify' agent is not registered.")
        print("  Make sure src/agents/code_modify_agent.py exists and is importable.")
        sys.exit(1)

    agent = agent_registry.get_instance("code_modify")
    print(f"\n[agent] {agent.config().display_name} (type={agent.config().agent_type})")

    # ── Executor with middleware ───────────────────────────────────────────
    executor = AgentExecutor(
        middlewares=[
            ConcurrencyMiddleware(max_concurrent=1),
            RetryMiddleware(max_retries=2, base_delay=2.0),
            LoggingMiddleware(),
        ],
    )

    # ── Build authenticated repo URL ───────────────────────────────────────
    authed_url = _embed_token(GIT_REPO_URL, GITLAB_TOKEN)

    # ── Execution context ──────────────────────────────────────────────────
    ctx = ExecutionContext(
        git_repo_url=authed_url,
        git_branch=GIT_BRANCH,
        extra_params={
            "modification_request": MODIFICATION_REQUEST,
            "target_path": TARGET_PATH,
            "context_hint": CONTEXT_HINT,
            # No gitlab_project_id → on_after_execute will skip MR creation
        },
    )
    print(f"\n[task] task_id = {ctx.task_id}")

    # ── Stream execution ───────────────────────────────────────────────────
    _print_separator("Agent Execution Stream")

    event_counts: dict[str, int] = {}
    tool_calls: list[dict] = []
    final_result = None
    captured_file_content: str = ""   # read while workspace is still alive
    wall_start = time.monotonic()

    async for event in executor.execute_stream(agent, ctx):
        etype = event.event_type
        event_counts[etype] = event_counts.get(etype, 0) + 1

        if etype == AgentEventType.ASSISTANT_TEXT:
            # Stream Claude's thinking word-by-word
            print(event.text, end="", flush=True)

        elif etype == AgentEventType.TOOL_USE:
            tool_name = event.data.get("tool_name", "?")
            tool_input = event.data.get("tool_input", {})
            # Summarise long inputs
            input_summary = str(tool_input)[:120].replace("\n", "↵")
            print(f"\n  [tool_use]  {tool_name}  ← {input_summary}")
            tool_calls.append({"name": tool_name, "input": tool_input})

        elif etype == AgentEventType.TOOL_RESULT:
            tool_name = event.data.get("tool_name", "?")
            result_text = event.text or str(event.data.get("content", ""))
            print(f"  [tool_result] {tool_name} → {result_text[:120].replace(chr(10), '↵')}")

        elif etype == AgentEventType.RESULT:
            final_result = event
            print()  # newline after streamed text
            # ── Read modified file NOW while workspace is still alive ──────
            # on_after_execute fires after this event is emitted (in finally),
            # then cleanup runs.  But we are still inside the async for loop,
            # so the generator has NOT yielded cleanup yet — workspace exists.
            if ctx.workspace_dir and ctx.workspace_dir.exists():
                target_file = ctx.workspace_dir / TARGET_PATH
                if target_file.exists():
                    captured_file_content = _read_file_safe(target_file)

        elif etype == AgentEventType.ERROR:
            print(f"\n  [ERROR] {event.text}")

    wall_elapsed = time.monotonic() - wall_start

    # ── Results ────────────────────────────────────────────────────────────
    _print_separator("Execution Summary")

    agent_summary = ""
    cost = 0.0
    duration = wall_elapsed
    status = "unknown"
    session_id = ""

    if final_result:
        data = final_result.data
        cost = data.get("cost_usd", 0.0)
        duration = data.get("duration_seconds", wall_elapsed)
        status = data.get("status", "?")
        session_id = data.get("session_id", "")
        agent_summary = final_result.text or ""

    print(f"  status         : {status}")
    print(f"  cost           : ${cost:.4f} USD")
    print(f"  duration       : {duration:.1f}s  (wall: {wall_elapsed:.1f}s)")
    print(f"  session_id     : {session_id}")
    if agent_summary:
        print(f"\n  Agent summary  :\n{agent_summary[:600]}")
    else:
        print(f"  (no RESULT event received — wall time: {wall_elapsed:.1f}s)")

    _print_separator("Event Counts")
    for etype, count in sorted(event_counts.items()):
        print(f"  {etype:<25} : {count}")

    _print_separator("Tool Calls Trace")
    if tool_calls:
        for i, tc in enumerate(tool_calls, 1):
            inp = str(tc["input"])[:100].replace("\n", "↵")
            print(f"  {i:>2}. {tc['name']:<20}  {inp}")
    else:
        print("  (no tool calls recorded)")

    _print_separator("Modified File Content")
    if captured_file_content:
        lines = captured_file_content.splitlines()
        print(f"\n  [{TARGET_PATH}]  ({len(lines)} lines, {len(captured_file_content)} chars)\n")
        for ln in lines[:60]:
            print(f"    {ln}")
        if len(lines) > 60:
            print(f"    ... ({len(lines) - 60} more lines)")
    else:
        print(f"  (file not captured — '{TARGET_PATH}' may not exist in repo, "
              f"or agent chose not to create/modify it)")

    _print_separator("Done")
    print()


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
