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

    def __str__(self) -> str:
        msg = super().__str__()
        if self.detail:
            return f"{msg}\n{self.detail}"
        return msg


# -- Agent ------------------------------------------------------------------

class AgentExecutionError(CckitError):
    """Raised when agent execution fails (SDK error, timeout, etc.)."""


class ConnectivityError(CckitError):
    """Raised when API connectivity preflight check fails.

    Possible causes: invalid API key, network unreachable, proxy misconfigured.
    """


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
