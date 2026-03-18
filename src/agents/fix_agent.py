"""FixAgent — repairs broken UI locators.

This is a reference implementation showing how to define a business agent
in ~20 lines.  It demonstrates:

* ``config()`` for static declaration
* ``build_prompt()`` for dynamic prompt construction
* ``required_params()`` for pre-execution validation
* ``on_after_execute()`` lifecycle hook (create an MR after the fix)

GitLab credentials are passed per-task via ``extra_params`` so that
different projects can use their own tokens concurrently.
"""

from __future__ import annotations

import logging

from core.agent.base import BaseAgent
from core.agent.registry import agent_registry
from core.agent.schemas import AgentConfig, AgentResult, ExecutionContext, SubAgentConfig

logger = logging.getLogger(__name__)


@agent_registry.register
class FixAgent(BaseAgent):
    """Agent that fixes broken UI test locators."""

    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="fix",
            display_name="Locator Fix Agent",
            system_prompt=(
                "You are a UI test repair expert. "
                "When given a failing test file and error log, you:\n"
                "1. Read the test file and identify the broken locator.\n"
                "2. Analyse the page source or screenshot to find the correct selector.\n"
                "3. Update the locator in the test code.\n"
                "4. Verify the fix is syntactically valid.\n"
                "Only modify the minimum code necessary."
            ),
            allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            mcp_tool_names=["get_failure_info", "report_progress"],
            needs_workspace=True,
            needs_git_clone=True,
            sub_agents=[
                SubAgentConfig(
                    name="log-analyzer",
                    description=(
                        "Analyze test execution logs to identify the root cause "
                        "of a locator failure."
                    ),
                    tools=["Read", "Grep"],
                ),
            ],
        )

    def required_params(self) -> list[str]:
        return ["test_file", "error_log"]

    def build_prompt(self, ctx: ExecutionContext) -> str:
        test_file = ctx.extra_params.get("test_file", "unknown")
        error_log = ctx.extra_params.get("error_log", "")
        project_id = ctx.extra_params.get("project_id", "")

        return (
            f"The following UI test has a broken locator.\n\n"
            f"**Test file**: `{test_file}`\n\n"
            f"**Error log**:\n```\n{error_log}\n```\n\n"
            f"Please fix the locator so the test passes. "
            f"Project ID for the platform MCP tool is `{project_id}`."
        )

    async def on_after_execute(
        self,
        ctx: ExecutionContext,
        result: AgentResult,
    ) -> None:
        """Create a merge request with the fix if the agent succeeded.

        GitLab credentials are read from ``ctx.extra_params``:

        - ``gitlab_url``: GitLab instance URL
        - ``gitlab_token``: project/personal access token
        - ``gitlab_project_id``: numeric project ID
        - ``gitlab_default_branch`` (optional): defaults to ``"main"``
        """
        if result.is_error:
            logger.warning("FixAgent failed for task %s — skipping MR", ctx.task_id)
            return

        gitlab_url = ctx.extra_params.get("gitlab_url")
        gitlab_token = ctx.extra_params.get("gitlab_token")
        project_id = ctx.extra_params.get("gitlab_project_id")
        default_branch = ctx.extra_params.get("gitlab_default_branch", "main")

        if not project_id or not gitlab_url or not gitlab_token:
            logger.info(
                "FixAgent task %s — missing GitLab credentials, skipping MR",
                ctx.task_id,
            )
            return

        if not ctx.workspace_dir:
            return

        try:
            from core.agent.git import operations as git_ops
            from core.agent.git.gitlab_client import GitLabClient

            branch_name = f"fix/{ctx.task_id}"
            await git_ops.create_branch(branch_name, cwd=ctx.workspace_dir)
            await git_ops.add_all(cwd=ctx.workspace_dir)
            commit_sha = await git_ops.commit(
                f"fix: repair broken locator (task {ctx.task_id})",
                cwd=ctx.workspace_dir,
            )
            await git_ops.push(
                "origin",
                branch_name,
                cwd=ctx.workspace_dir,
                extra_env=ctx.env or None,
            )

            client = GitLabClient(
                url=gitlab_url,
                token=gitlab_token,
                default_branch=default_branch,
            )
            mr = await client.create_mr(
                project_id,
                source_branch=branch_name,
                title=f"fix: repair broken locator (task {ctx.task_id})",
                description=f"Auto-generated by FixAgent.\n\n{result.result_text[:500]}",
            )
            logger.info("Created MR: %s", mr.web_url)
            result.extra["mr_url"] = mr.web_url
            result.extra["mr_id"] = mr.mr_iid
            result.extra["commit_sha"] = commit_sha
        except Exception as exc:
            logger.exception("Failed to create MR for task %s", ctx.task_id)
            result.extra["mr_error"] = str(exc)
