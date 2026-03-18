"""Core infrastructure — re-exports key classes for convenience."""

from core.agent.base import BaseAgent
from core.agent.executor import AgentExecutor
from core.agent.middleware import (
    AgentMiddleware,
    ConcurrencyMiddleware,
    LoggingMiddleware,
    RetryMiddleware,
)
from core.agent.registry import AgentRegistry, agent_registry
from core.agent.schemas import (
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentResult,
    ExecutionContext,
    SubAgentConfig,
    TaskStatus,
)
from core.exceptions import (
    AgentExecutionError,
    AgentNotFoundError,
    CoreError,
    GitLabAPIError,
    GitOperationError,
    SandboxConfigError,
    SkillError,
    WorkspaceError,
)

__all__ = [
    # Agent
    "BaseAgent",
    "AgentExecutor",
    "AgentRegistry",
    "agent_registry",
    # Middleware
    "AgentMiddleware",
    "RetryMiddleware",
    "ConcurrencyMiddleware",
    "LoggingMiddleware",
    # Schemas
    "AgentConfig",
    "AgentEvent",
    "AgentEventType",
    "AgentResult",
    "ExecutionContext",
    "SubAgentConfig",
    "TaskStatus",
    # Exceptions
    "CoreError",
    "AgentNotFoundError",
    "AgentExecutionError",
    "WorkspaceError",
    "GitOperationError",
    "GitLabAPIError",
    "SandboxConfigError",
    "SkillError",
]
