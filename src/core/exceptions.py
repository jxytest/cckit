"""Exception hierarchy for the core infrastructure.

All exceptions inherit from ``CoreError`` so callers can catch broadly or
narrowly.
"""

from __future__ import annotations


class CoreError(Exception):
    """Root exception for the core package."""

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


# -- Agent ------------------------------------------------------------------

class AgentNotFoundError(CoreError):
    """Raised when a requested agent type is not registered."""


class AgentExecutionError(CoreError):
    """Raised when agent execution fails (SDK error, timeout, etc.)."""


# -- Workspace / Sandbox ----------------------------------------------------

class WorkspaceError(CoreError):
    """Raised on workspace creation / cleanup failures."""


class SandboxConfigError(CoreError):
    """Raised when sandbox configuration is invalid."""


# -- Git / GitLab -----------------------------------------------------------

class GitOperationError(CoreError):
    """Raised when a git CLI command fails."""


class GitLabAPIError(CoreError):
    """Raised when a GitLab API call fails."""


# -- Skill -----------------------------------------------------------------

class SkillError(CoreError):
    """Raised when skill provisioning fails (not found, invalid, copy error)."""
