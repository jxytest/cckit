"""clab — Claude Agent Builder.

Build multi-agent AI systems on top of Claude CLI.

Quick start::

    from clab import Agent, Runner, RunContext

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
from clab.agent import Agent

# Exceptions
from clab.exceptions import (
    AgentExecutionError,
    ClabError,
    GitLabAPIError,
    GitOperationError,
    SkillError,
    WorkspaceError,
)

# Middleware
from clab.middleware import (
    ConcurrencyMiddleware,
    LoggingMiddleware,
    Middleware,
    RetryMiddleware,
)
from clab.runner import Runner

# Data types
from clab.types import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    LiteLlm,
    ModelConfig,
    RunContext,
    RunnerConfig,
    SandboxOptions,
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
    "TaskStatus",
    "WorkspaceConfig",
    # Exceptions
    "ClabError",
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
