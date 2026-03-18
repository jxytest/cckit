"""GitLab API client for merge-request operations.

Wraps ``python-gitlab`` to provide a thin async-friendly interface.
Heavy calls are offloaded to a thread executor.

Credentials **must** be injected per-instance.  The client requires these
explicitly — there is no global fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field

from cckit._models import CustomModel
from cckit.exceptions import GitLabAPIError

logger = logging.getLogger(__name__)


class MRResult(CustomModel):
    """Simplified merge-request result."""

    mr_id: int = 0
    mr_iid: int = 0
    web_url: str = ""
    title: str = ""
    source_branch: str = ""
    target_branch: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class GitLabClient:
    """Thin async wrapper around python-gitlab.

    Credentials **must** be injected per-instance.  This avoids global
    singleton pollution and ensures concurrent tasks with different
    projects/tokens don't interfere.

    Parameters
    ----------
    url:
        GitLab instance URL (e.g. ``https://gitlab.example.com``).
    token:
        Personal access token or project token.
    default_branch:
        Target branch for MRs when not specified (default ``"main"``).
    """

    def __init__(
        self,
        url: str,
        token: str,
        default_branch: str = "main",
    ) -> None:
        if not url:
            raise GitLabAPIError("GitLab URL is required")
        if not token:
            raise GitLabAPIError("GitLab token is required")
        self._url = url
        self._token = token
        self._default_branch = default_branch
        self._gl: Any = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    def _connect(self) -> Any:
        """Create the ``gitlab.Gitlab`` instance (sync, called in executor)."""
        if self._gl is None:
            try:
                import gitlab  # noqa: WPS433
            except ImportError as exc:
                raise GitLabAPIError(
                    "python-gitlab is not installed",
                    detail=str(exc),
                ) from exc
            self._gl = gitlab.Gitlab(self._url, private_token=self._token)
            self._gl.auth()
        return self._gl

    # ------------------------------------------------------------------
    # Merge Request
    # ------------------------------------------------------------------

    async def create_mr(
        self,
        project_id: int | str,
        *,
        source_branch: str,
        target_branch: str = "",
        title: str,
        description: str = "",
    ) -> MRResult:
        """Create a merge request and return the result."""
        target = target_branch or self._default_branch

        def _create() -> MRResult:
            gl = self._connect()
            project = gl.projects.get(project_id)
            mr = project.mergerequests.create({
                "source_branch": source_branch,
                "target_branch": target,
                "title": title,
                "description": description,
            })
            return MRResult(
                mr_id=mr.id,
                mr_iid=mr.iid,
                web_url=mr.web_url,
                title=mr.title,
                source_branch=mr.source_branch,
                target_branch=mr.target_branch,
            )

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _create)
        except Exception as exc:
            if isinstance(exc, GitLabAPIError):
                raise
            raise GitLabAPIError(
                f"Failed to create MR on project {project_id}",
                detail=str(exc),
            ) from exc
