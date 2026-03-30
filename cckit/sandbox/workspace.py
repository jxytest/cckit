"""Workspace lifecycle management.

Creates isolated temporary directories for agent execution and handles
cleanup.  Git cloning is handled by ``cckit.git.operations`` — this module
only manages the filesystem.

Lifecycle states::

    create()  →  [agent execution]  →  cleanup()     (normal one-shot)
    create()  →  [agent execution]  →  suspend()      (preserves for resume)
    resume()  →  [agent execution]  →  cleanup()      (continued session)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from cckit.exceptions import WorkspaceError

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Create and clean up agent workspaces.

    Parameters
    ----------
    root:
        Root directory for workspace creation.  Required — no global fallback.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, task_id: str) -> Path:
        """Create an empty workspace directory and return its path."""
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            workspace = Path(
                tempfile.mkdtemp(prefix=f"agent_{task_id}_", dir=self._root)
            )
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to create workspace for task {task_id}",
                detail=str(exc),
            ) from exc

        logger.info("Created workspace: %s", workspace)
        return workspace

    async def cleanup(self, workspace: Path) -> None:
        """Remove a workspace directory tree."""
        if not workspace.exists():
            return
        shutil.rmtree(workspace, ignore_errors=True)
        logger.info("Cleaned up workspace: %s", workspace)

    async def suspend(self, workspace: Path) -> None:
        """Mark a workspace as suspended — skip cleanup so it can be resumed.

        The workspace directory is kept intact on disk.  A follow-up
        execution can pass the same path via ``RunContext.workspace_dir``
        and call :meth:`resume` to reuse it.
        """
        logger.info("Suspended workspace (preserved for resume): %s", workspace)

    async def resume(
        self, workspace: Path, *, recreate: bool = False
    ) -> tuple[Path, bool]:
        """Validate and return an existing workspace for a resumed session.

        Parameters
        ----------
        workspace:
            The workspace directory to resume.
        recreate:
            If ``True`` and the directory no longer exists, recreate it at
            the same path instead of raising.  The caller is responsible for
            re-cloning / re-provisioning as needed.

        Returns
        -------
        tuple[Path, bool]
            ``(workspace_path, was_recreated)`` — the second element is
            ``True`` when the directory was missing and had to be recreated.

        Raises
        ------
        WorkspaceError
            If the directory does not exist and *recreate* is ``False``, or
            if the path exists but is not a directory.
        """
        if not workspace.exists():
            if recreate:
                try:
                    workspace.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise WorkspaceError(
                        f"Failed to recreate workspace: {workspace}",
                        detail=str(exc),
                    ) from exc
                logger.warning(
                    "Workspace recreated at original path (was missing): %s",
                    workspace,
                )
                return workspace, True
            raise WorkspaceError(
                f"Cannot resume — workspace not found: {workspace}",
            )
        if not workspace.is_dir():
            raise WorkspaceError(
                f"Cannot resume — path is not a directory: {workspace}",
            )
        logger.info("Resumed workspace: %s", workspace)
        return workspace, False
