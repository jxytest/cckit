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
        async for event in runner.run_stream(agent, ctx):
            print(event.text, end="", flush=True)

    asyncio.run(main())
"""

__version__ = "0.1.0"

# Core classes
from cckit.agent import Agent

# Exceptions
from cckit.exceptions import (
    AgentExecutionError,
    CckitError,
    GitLabAPIError,
    GitOperationError,
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
    AgentEvent,
    AgentEventType,
    AgentResult,
    LiteLlm,
    ModelConfig,
    RunContext,
    RunnerConfig,
    SandboxOptions,
    StreamResult,
    TaskStatus,
    WorkspaceConfig,
)

__all__ = [
    # Core
    "Agent",
    "Runner",
    # Types
    "AgentEvent",
    "AgentEventType",
    "AgentResult",
    "LiteLlm",
    "ModelConfig",
    "RunContext",
    "RunnerConfig",
    "SandboxOptions",
    "StreamResult",
    "TaskStatus",
    "WorkspaceConfig",
    # Exceptions
    "CckitError",
    "AgentExecutionError",
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
