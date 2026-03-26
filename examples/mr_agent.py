"""MR Agent — 执行完成后自动 commit → push → 创建 GitLab MR。

调用示例::

    python mr_agent.py "把所有 print 替换为 logging"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from dotenv import load_dotenv

from cckit import Agent, GitConfig, ModelConfig, RunContext, Runner, RunnerConfig, WorkspaceConfig
from cckit.git import operations as git_ops
from cckit.git.gitlab_client import GitLabClient
from cckit.types import AgentResult

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 配置区（也可全部走环境变量） ──────────────────────────────────────────

GITLAB_URL   = os.getenv("GITLAB_URL",   "https://xxx")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "2ukxxxxxa")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-6xxxY")

# ── on_after 回调 ──────────────────────────────────────────────────────────

def _find_repo_dir(workspace_dir: Path) -> Path:
    if (workspace_dir / ".git").exists():
        return workspace_dir
    for sub in sorted(workspace_dir.iterdir()):
        if sub.is_dir() and (sub / ".git").exists():
            return sub
    return workspace_dir


async def create_mr(ctx: RunContext, result: AgentResult) -> None:
    """Agent 执行完成后：commit → push → 创建 GitLab MR，并将结果写入 result.extra。"""
    if result.is_error or not ctx.workspace_dir:
        return

    repo_dir      = _find_repo_dir(ctx.workspace_dir)
    project_id    = ctx.params["project_id"]
    target_branch = ctx.params.get("target_branch", "main")
    new_branch    = f"ai-fix/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    title         = ctx.params.get("mr_title") or f"[AI] {ctx.prompt[:60]}"

    # 配置 git 用户（忽略失败）
    for args in [("config", "user.email", "mr-agent@cckit.local"),
                 ("config", "user.name",  "MR Agent")]:
        with suppress(Exception):
            await git_ops.run_git(*args, cwd=repo_dir)

    if not await git_ops.status(cwd=repo_dir, short=True):
        logger.info("没有检测到文件改动，跳过 MR 创建")
        return

    await git_ops.create_branch(new_branch, cwd=repo_dir)
    await git_ops.add_all(cwd=repo_dir)
    commit_sha = await git_ops.commit(f"feat: {title}", cwd=repo_dir)
    await git_ops.push("origin", new_branch, cwd=repo_dir,
                       extra_env=ctx.resolved_git().build_git_env())

    mr = await GitLabClient(
        url=GITLAB_URL,
        token=GITLAB_TOKEN,
        default_branch=target_branch,
    ).create_mr(
        project_id,
        source_branch=new_branch,
        target_branch=target_branch,
        title=title,
        description=(
            f"由 MR Agent 自动生成。\n\n"
            f"**需求**：\n{ctx.prompt}\n\n"
            f"**Agent 摘要**：\n{result.output_text[:2000]}"
        ),
    )

    result.extra.update(mr_url=mr.web_url, mr_iid=mr.mr_iid,
                        mr_branch=new_branch, commit_sha=commit_sha)
    logger.info("✅ MR 已创建: %s", mr.web_url)


# ── Agent 定义（声明式，可复用） ───────────────────────────────────────────

def _instruction(ctx: RunContext) -> str:
    lang = ctx.params.get("lang", "")
    return (
        f"你是一位经验丰富的软件工程师。{'你精通 ' + lang + '。' if lang else ''}\n"
        "根据用户需求修改代码仓库中的文件。\n\n"
        "工作原则：\n"
        "1. 先浏览仓库结构，了解项目架构\n"
        "2. 精确、最小化地修改与需求相关的文件\n"
        "3. 保持原有代码风格\n"
        "4. 完成后简要总结所做修改。向用户问好时，不要提及自己是claude"
    )


mr_agent = Agent(
    name="mr_agent",
    instruction=_instruction,
    tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob"],
    required_params=["project_id"],
    on_after=create_mr,
)

# ── Runner ─────────────────────────────────────────────────────────────────

runner = Runner(
    config=RunnerConfig(
        default_model=ModelConfig(
            model="claude-haiku-4-5-20251001",
            api_key=ANTHROPIC_KEY,
            base_url="https://aigc-api.fuxi.netease.com",
        ),
        log_level="INFO",
    ),
    preflight_check=True,
)

# ── 入口 ───────────────────────────────────────────────────────────────────

async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else input("请输入修改需求: ")

    ctx = RunContext(
        prompt=prompt,
        workspace=WorkspaceConfig(enabled=True),
        git=GitConfig(
            repo_url=os.getenv("GIT_REPO_URL", "https://xxx/xx.git"),
            branch=os.getenv("GIT_TARGET_BRANCH", "master"),
            token=GITLAB_TOKEN,
            clone=True,
        ),
        params={
            "project_id":    os.getenv("GITLAB_PROJECT_ID", "CloudQA/xx"),
            "target_branch": os.getenv("GIT_TARGET_BRANCH", "master"),
        },
    )

    stream = runner.run_stream(mr_agent, ctx)
    async for message in stream:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage) and message.is_error and message.result:
            print(message.result, end="", flush=True)

    result = stream.result
    print()
    if mr_url := result.extra.get("mr_url"):
        print(f"✅ MR 已创建: {mr_url}")
    elif result.is_error:
        print(f"❌ 失败: {result.error_message}")
    else:
        print("✅ 完成（无文件改动）")


if __name__ == "__main__":
    asyncio.run(main())
