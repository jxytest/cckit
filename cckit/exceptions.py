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
        return super().__str__()


# -- Agent ------------------------------------------------------------------

class AgentExecutionError(CckitError):
    """Raised when agent execution fails (SDK error, timeout, etc.)."""


class HookError(CckitError):
    """Raised when a lifecycle hook (after_execute / error_execute) raises.

    Wraps the original exception so the hook name and original traceback
    are preserved while still propagating to the caller.

    Attributes
    ----------
    hook_name:
        Name of the hook that failed (e.g. ``"after_execute"``).
    original:
        The original exception raised by the hook.
    """

    def __init__(self, hook_name: str, original: BaseException) -> None:
        self.hook_name = hook_name
        self.original = original
        super().__init__(
            f"{hook_name} hook failed: {original}",
            detail=repr(original),
        )


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
