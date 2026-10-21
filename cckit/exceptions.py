"""Exception hierarchy for cckit.

All exceptions inherit from ``CckitError`` so callers can catch broadly or
narrowly.
"""

from __future__ import annotations

from typing import Any


class CckitError(Exception):
    """Root exception for the cckit package.

    Attributes
    ----------
    detail:
        Legacy free-form technical detail.  ``str(exc)`` appends it when it
        is not already part of the message, so existing log output is
        unchanged.
    code:
        Stable machine-readable failure code (for example ``"max_turns"``).
        Callers should branch on this instead of matching message text.
        ``None`` when the exception was not classified.
    details:
        Structured metadata about the failure (session id, turn count, SDK
        subtype, ...).  Empty when no metadata is available.
    """

    def __init__(
        self,
        message: str = "",
        *,
        detail: str = "",
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail
        self.code = code
        self.details: dict[str, Any] = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        message = super().__str__().strip()
        detail = self.detail.strip()

        if detail and detail != message and detail not in message:
            if message:
                return f"{message}\n{detail}"
            return detail

        return message


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
