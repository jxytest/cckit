"""Example: workspace-only sandbox — agent can ONLY read/write its workspace.

Sandbox rules:
  WRITE  → only workspace directory (auto-injected by cckit)
  READ   → everything except ~/ (home dir blocked to protect ~/.ssh etc.)
  NETWORK→ all outbound blocked (empty allowed_domains)
  ESCAPE → dangerouslyDisableSandbox blocked (allow_unsandboxed_commands=False)

Requires macOS, Linux, or WSL2.  Native Windows does not support OS-level sandbox.

Run:
    python examples/sandbox_workspace_only.py
"""

import asyncio

from claude_agent_sdk import AssistantMessage, TextBlock

from cckit import Agent, RunContext, Runner, RunnerConfig, SandboxOptions, WorkspaceConfig

agent = Agent(
    name="sandbox-demo",
    instruction="You are a sandboxed assistant. Only read/write inside your workspace.",
    tools=["Bash", "Read", "Write", "Edit"],
)


async def main() -> None:
    runner = Runner(
        config=RunnerConfig(
            sandbox=SandboxOptions(
                enabled=True,
                deny_read=["~/"],       # block home dir reads
                allowed_domains=[],     # block all outbound network
                # allow_unsandboxed_commands defaults to False —
                # the Agent cannot use dangerouslyDisableSandbox to escape.
            ),
        ),
    )

    ctx = RunContext(
        prompt=(
            "1. Create hello.txt with 'Hello from sandbox!' in your workspace.\n"
            "2. Read it back.\n"
            "3. Try writing to /tmp/escape_test.txt (should fail — outside workspace).\n"
            "4. Try reading ~/.ssh/ (should fail — home dir blocked)."
        ),
        workspace=WorkspaceConfig(enabled=True),
    )

    async for message in runner.run_stream(agent, ctx):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
