"""Example: FixAgent using the cckit SDK.

Demonstrates:
- Agent definition with dynamic instruction
- Sub-agent composition
- Lifecycle callback (on_after) for creating GitLab MRs
- MCP tools integration
- Workspace with git clone

Run:
    python examples/fix_agent.py
"""

import asyncio

from cckit import Agent, Runner, RunContext, WorkspaceConfig
from cckit.git import operations as git_ops
from cckit.git.gitlab_client import GitLabClient
from cckit.tools.platform import get_platform_mcp_server


def build_fix_instruction(ctx):
    """Dynamic system prompt for the fix agent."""
    return (
        "You are a UI test repair expert.\n"
        "When given a failing test file and error log, you:\n"
        "1. Read the test file and identify the broken locator.\n"
        "2. Analyse the page source or screenshot to find the correct selector.\n"
        "3. Update the locator in the test code.\n"
        "4. Verify the fix is syntactically valid.\n"
        "Only modify the minimum code necessary."
    )


async def on_fix_complete(ctx, result):
    """Create a merge request with the fix."""
    if result.is_error or not ctx.workspace_dir:
        return

    gitlab_url = ctx.params.get("gitlab_url")
    gitlab_token = ctx.params.get("gitlab_token")
    project_id = ctx.params.get("gitlab_project_id")

    if not all([gitlab_url, gitlab_token, project_id]):
        return

    try:
        branch = f"fix/{ctx.task_id}"
        await git_ops.create_branch(branch, cwd=ctx.workspace_dir)
        await git_ops.add_all(cwd=ctx.workspace_dir)
        await git_ops.commit(
            f"fix: repair broken locator ({ctx.task_id})",
            cwd=ctx.workspace_dir,
        )
        await git_ops.push(
            "origin",
            branch,
            cwd=ctx.workspace_dir,
            set_upstream=True,
            extra_env=ctx.env or None,
        )

        client = GitLabClient(url=gitlab_url, token=gitlab_token)
        mr = await client.create_mr(
            project_id,
            source_branch=branch,
            title=f"fix: repair broken locator ({ctx.task_id})",
            description=(
                f"Automated fix by FixAgent.\n\n"
                f"- Test file: `{ctx.params.get('test_file', 'N/A')}`\n"
                f"- Error: `{ctx.params.get('error_log', 'N/A')[:200]}`"
            ),
        )
        result.extra["mr_url"] = mr.web_url
        result.extra["mr_id"] = mr.mr_iid
    except Exception as exc:
        result.extra["mr_error"] = str(exc)


# -- Sub-agents --

log_analyzer = Agent(
    name="log-analyzer",
    description="Analyze test execution logs to find root cause of locator failures",
    tools=["Read", "Grep"],
)

# -- Main agent --

fix_agent = Agent(
    name="fix",
    description="Locator Fix Agent",
    instruction=build_fix_instruction,
    tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
    sub_agents=[log_analyzer],
    required_params=["test_file", "error_log"],
    mcp_tools=["get_failure_info", "report_progress"],
    on_after=on_fix_complete,
)


async def main():
    runner = Runner(
        mcp_servers={"platform": get_platform_mcp_server},
    )

    ctx = RunContext(
        prompt="Fix the broken locator in the test file.",
        workspace=WorkspaceConfig(enabled=True, git_clone=True),
        git_repo_url="https://gitlab.com/team/ui-tests.git",
        git_branch="main",
        params={
            "test_file": "tests/test_login.py",
            "error_log": "NoSuchElementException: id=login-btn",
            "project_id": "42",
            "gitlab_project_id": 123,
            "gitlab_url": "https://gitlab.com",
            "gitlab_token": "glpat-xxx",
        },
    )

    async for event in runner.run_stream(fix_agent, ctx):
        if event.text:
            print(f"[{event.event_type}] {event.text[:200]}")
        else:
            print(f"[{event.event_type}] {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
