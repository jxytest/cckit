"""Exception hierarchy for clab.

All exceptions inherit from ``ClabError`` so callers can catch broadly or
narrowly.
"""

from __future__ import annotations


class ClabError(Exception):
    """Root exception for the clab package."""

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


# -- Agent ------------------------------------------------------------------

class AgentExecutionError(ClabError):
    """Raised when agent execution fails (SDK error, timeout, etc.)."""


# -- Workspace / Sandbox ----------------------------------------------------

class WorkspaceError(ClabError):
    """Raised on workspace creation / cleanup failures."""


# -- Git / GitLab -----------------------------------------------------------

class GitOperationError(ClabError):
    """Raised when a git CLI command fails."""


class GitLabAPIError(ClabError):
    """Raised when a GitLab API call fails."""


# -- Skill -----------------------------------------------------------------

class SkillError(ClabError):
    """Raised when skill provisioning fails (not found, invalid, copy error)."""
