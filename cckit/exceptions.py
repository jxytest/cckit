"""Exception hierarchy for cckit.

All exceptions inherit from ``CckitError`` so callers can catch broadly or
narrowly.
"""

from __future__ import annotations


class CckitError(Exception):
    """Root exception for the cckit package."""

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


# -- Agent ------------------------------------------------------------------

class AgentExecutionError(CckitError):
    """Raised when agent execution fails (SDK error, timeout, etc.)."""


# -- Workspace / Sandbox ----------------------------------------------------

class WorkspaceError(CckitError):
    """Raised on workspace creation / cleanup failures."""


# -- Git / GitLab -----------------------------------------------------------

class GitOperationError(CckitError):
    """Raised when a git CLI command fails."""


class GitLabAPIError(CckitError):
    """Raised when a GitLab API call fails."""


# -- Skill -----------------------------------------------------------------

class SkillError(CckitError):
    """Raised when skill provisioning fails (not found, invalid, copy error)."""
