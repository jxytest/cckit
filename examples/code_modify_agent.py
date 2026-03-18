"""Example: CodeModifyAgent using the cckit SDK.

Demonstrates:
- Agent for code modification tasks
- Sub-agent for code review
- Workspace with git clone
- Lifecycle callback for capturing modified files and creating MRs

Run:
    python examples/code_modify_agent.py
"""

import asyncio

from cckit import Agent, Runner, RunContext, GitConfig, WorkspaceConfig
from cckit.git import operations as git_ops
from cckit.git.gitlab_client import GitLabClient
from cckit.tools.platform import get_platform_mcp_server


def build_modify_instruction(ctx):
    """Dynamic system prompt for the code modification agent."""
    target_path = ctx.params.get("target_path", "")
    return (
        "You are a precise code modification agent.\n"
        "When given a modification request and target path, you:\n"
        "1. Read and understand the current code.\n"
        "2. Analyse the modification request carefully.\n"
        "3. Apply the minimum necessary changes.\n"
        "4. Verify the changes compile/parse correctly.\n"
        f"\nTarget path: {target_path}"
    )


async def on_modify_complete(ctx, result):
    """Capture modified files and optionally create a MR."""
    if result.is_error or not ctx.workspace_dir:
        return

    # Capture modified files
    try:
        diff_output = await git_ops.diff(cwd=ctx.workspace_dir, name_only=True)
        result.extra["modified_files"] = [
            f for f in diff_output.strip().split("\n") if f
        ]
    except Exception:
        pass

    # Create MR if GitLab credentials are provided
    gitlab_url = ctx.params.get("gitlab_url")
    gitlab_token = ctx.params.get("gitlab_token")
    project_id = ctx.params.get("gitlab_project_id")

    if not all([gitlab_url, gitlab_token, project_id]):
        return

    try:
        branch = f"code-modify/{ctx.task_id}"
        git_env = ctx._resolved_git().build_git_env() or None
        await git_ops.create_branch(branch, cwd=ctx.workspace_dir)
        await git_ops.add_all(cwd=ctx.workspace_dir)
        await git_ops.commit(
            ctx.params.get("mr_title", f"code-modify: {ctx.task_id}"),
            cwd=ctx.workspace_dir,
        )
        await git_ops.push(
            "origin",
            branch,
            cwd=ctx.workspace_dir,
            extra_env=git_env,
        )

        client = GitLabClient(url=gitlab_url, token=gitlab_token)
        mr = await client.create_mr(
            project_id,
            source_branch=branch,
            title=ctx.params.get("mr_title", f"Code modification ({ctx.task_id})"),
            description=(
                f"Automated code modification.\n\n"
                f"**Request:** {ctx.params.get('modification_request', 'N/A')[:500]}"
            ),
        )
        result.extra["mr_url"] = mr.web_url
        result.extra["mr_id"] = mr.mr_iid
    except Exception as exc:
        result.extra["mr_error"] = str(exc)


# -- Sub-agents --

code_reviewer = Agent(
    name="code-reviewer",
    description="Review code changes for correctness and best practices",
    tools=["Read", "Grep", "Glob"],
)

# -- Main agent --

code_modify_agent = Agent(
    name="code-modify",
    description="Code Modification Agent",
    instruction=build_modify_instruction,
    tools=["Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep"],
    sub_agents=[code_reviewer],
    required_params=["modification_request", "target_path"],
    mcp_tools=["report_progress"],
    on_after=on_modify_complete,
)


async def main():
    runner = Runner(
        mcp_servers={"platform": get_platform_mcp_server},
    )

    ctx = RunContext(
        prompt="Apply the following code modification.",
        workspace=WorkspaceConfig(enabled=True),
        git=GitConfig(
            repo_url="https://gitlab.com/team/project.git",
            token="glpat-xxxx",  # securely isolated from Agent subprocess
            clone=True,
        ),
        params={
            "modification_request": "Change the base image tag to python:3.12",
            "target_path": "Dockerfile",
        },
    )

    async for event in runner.run_stream(code_modify_agent, ctx):
        if event.text:
            print(f"[{event.event_type}] {event.text[:200]}")
        else:
            print(f"[{event.event_type}] {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
