"""cckit — Claude Agent Kit.

Build multi-agent AI systems on top of Claude CLI.

Quick start::

    from cckit import Agent, Runner, RunContext

    agent = Agent(
        name="assistant",
        instruction="You are a helpful assistant.",
        tools=["Bash", "Read", "Write"],
    )

    runner = Runner()
    ctx = RunContext(prompt="Hello!")

    import asyncio

    async def main():
        async for message in runner.run_stream(agent, ctx):
            print(message)

    asyncio.run(main())
"""

__version__ = "0.1.0"

# Core classes
from cckit._cli import check_api_connectivity
from cckit.agent import Agent

# Exceptions
from cckit.exceptions import (
    AgentExecutionError,
    CckitError,
    ConnectivityError,
    GitLabAPIError,
    GitOperationError,
    HookError,
    SkillError,
    WorkspaceError,
)

# Middleware
from cckit.middleware import (
    ConcurrencyMiddleware,
    LoggingMiddleware,
    Middleware,
    RetryMiddleware,
)
from cckit.runner import Runner

# Data types
from cckit.types import (
    AgentResult,
    ContextConfig,
    GitConfig,
    ModelConfig,
    RunContext,
    RunnerConfig,
    SandboxOptions,
    StreamResult,
    TaskBudgetConfig,
    TaskStatus,
    WorkspaceConfig,
)

__all__ = [
    # Core
    "Agent",
    "Runner",
    "check_api_connectivity",
    # Types
    "AgentResult",
    "ContextConfig",
    "GitConfig",
    "ModelConfig",
    "RunContext",
    "RunnerConfig",
    "SandboxOptions",
    "StreamResult",
    "TaskBudgetConfig",
    "TaskStatus",
    "WorkspaceConfig",
    # Exceptions
    "CckitError",
    "AgentExecutionError",
    "ConnectivityError",
    "HookError",
    "WorkspaceError",
    "GitOperationError",
    "GitLabAPIError",
    "SkillError",
    # Middleware
    "Middleware",
    "RetryMiddleware",
    "ConcurrencyMiddleware",
    "LoggingMiddleware",
]
