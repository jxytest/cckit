"""CodeModifyAgent — modifies code according to user instructions.

This agent accepts a natural-language modification instruction
(``modification_request``) and a target file or directory (``target_path``),
then applies the requested changes to the codebase.

It demonstrates:

* ``config()`` for static declaration
* ``build_prompt()`` for dynamic prompt construction
* ``required_params()`` for pre-execution validation
* ``on_after_execute()`` lifecycle hook (optional: create GitLab MR)
* Sub-agent for code review after modification

Credentials are passed per-task via ``extra_params`` so that
different projects can use their own tokens concurrently.
"""

from __future__ import annotations

import logging

from core.agent.base import BaseAgent
from core.agent.registry import agent_registry
from core.agent.schemas import AgentConfig, AgentResult, ExecutionContext, SubAgentConfig

logger = logging.getLogger(__name__)


@agent_registry.register
class CodeModifyAgent(BaseAgent):
    """Agent that modifies code files according to user-specified instructions."""

    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="code_modify",
            display_name="Code Modification Agent",
            system_prompt=(
                "You are an expert software engineer tasked with modifying code "
                "according to precise user instructions. When given a modification "
                "request and a target file or directory, you:\n"
                "1. Read and fully understand the existing code structure.\n"
                "2. Analyse the modification request carefully.\n"
                "3. Apply the requested changes in the most minimal and correct way.\n"
                "4. Verify the modification is syntactically valid and logically sound.\n"
                "5. Report a concise summary of all changes made.\n"
                "Only modify what is explicitly requested. Do not refactor unrelated "
                "code or introduce new dependencies unless explicitly asked."
            ),
            allowed_tools=["Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep"],
            mcp_tool_names=["report_progress"],
            needs_workspace=True,
            needs_git_clone=True,
            sub_agents=[
                SubAgentConfig(
                    name="code-reviewer",
                    description=(
                        "Review the modified code to ensure correctness, style "
                        "consistency, and absence of regressions."
                    ),
                    tools=["Read", "Grep", "Glob"],
                ),
            ],
        )

    def required_params(self) -> list[str]:
        """Declare mandatory keys in ``extra_params``."""
        return ["modification_request", "target_path"]

    def build_prompt(self, ctx: ExecutionContext) -> str:
        modification_request = ctx.extra_params.get("modification_request", "")
        target_path = ctx.extra_params.get("target_path", "")
        context_hint = ctx.extra_params.get("context_hint", "")
        project_id = ctx.extra_params.get("project_id", "")

        lines = [
            "Please modify the code according to the following instruction.\n",
            f"**Modification request**:\n{modification_request}\n",
            f"**Target file / directory**: `{target_path}`\n",
        ]

        if context_hint:
            lines.append(f"**Additional context**:\n{context_hint}\n")

        lines.append(
            "After completing all changes, provide a brief summary listing "
            "each file modified and what was changed."
        )

        if project_id:
            lines.append(
                f"\nUse the `report_progress` MCP tool with project_id=`{project_id}` "
                "to report status milestones."
            )

        return "\n".join(lines)

    async def on_after_execute(
        self,
        ctx: ExecutionContext,
        result: AgentResult,
    ) -> None:
        """Capture modified file content and optionally create a GitLab MR.

        File content is always captured into ``result.extra["modified_files"]``
        (a dict of relative-path -> new-content) so callers can inspect changes
        even after the workspace has been cleaned up.

        GitLab credentials are read from ``ctx.extra_params``:

        - ``gitlab_url``: GitLab instance URL
        - ``gitlab_token``: project/personal access token
        - ``gitlab_project_id``: numeric project ID
        - ``gitlab_default_branch`` (optional): defaults to "main"
        - ``mr_title`` (optional): custom MR title
        """
        # ── Capture modified file content before workspace is cleaned up ──
        if ctx.workspace_dir and ctx.workspace_dir.exists() and not result.is_error:
            target_path = ctx.extra_params.get("target_path", "")
            if target_path:
                target = ctx.workspace_dir / target_path
                if target.exists():
                    try:
                        result.extra["modified_files"] = {
                            target_path: target.read_text(encoding="utf-8"),
                        }
                    except Exception as read_exc:
                        result.extra["modified_files_error"] = str(read_exc)

        # ── Optionally create a GitLab MR with the modifications ──
        if result.is_error:
            logger.warning(
                "CodeModifyAgent failed for task %s — skipping MR", ctx.task_id
            )
            return

        gitlab_url = ctx.extra_params.get("gitlab_url")
        gitlab_token = ctx.extra_params.get("gitlab_token")
        project_id = ctx.extra_params.get("gitlab_project_id")
        default_branch = ctx.extra_params.get("gitlab_default_branch", "main")
        mr_title = ctx.extra_params.get(
            "mr_title",
            f"feat: code modification (task {ctx.task_id})",
        )

        if not (project_id and gitlab_url and gitlab_token):
            logger.info(
                "CodeModifyAgent task %s — missing GitLab credentials, skipping MR",
                ctx.task_id,
            )
            return

        if not ctx.workspace_dir:
            return

        try:
            from core.agent.git import operations as git_ops
            from core.agent.git.gitlab_client import GitLabClient

            branch_name = f"code-modify/{ctx.task_id}"
            await git_ops.create_branch(branch_name, cwd=ctx.workspace_dir)
            await git_ops.add_all(cwd=ctx.workspace_dir)
            commit_sha = await git_ops.commit(
                f"feat: apply code modification (task {ctx.task_id})",
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
                title=mr_title,
                description=(
                    f"Auto-generated by CodeModifyAgent.\n\n"
                    f"**Modification request**: "
                    f"{ctx.extra_params.get('modification_request', '')}\n\n"
                    f"**Summary**:\n{result.result_text[:500]}"
                ),
            )
            logger.info("Created MR: %s", mr.web_url)
            result.extra["mr_url"] = mr.web_url
            result.extra["mr_id"] = mr.mr_iid
            result.extra["commit_sha"] = commit_sha
        except Exception as exc:
            logger.exception("Failed to create MR for task %s", ctx.task_id)
            result.extra["mr_error"] = str(exc)
