"""Workspace lifecycle management.

Creates isolated temporary directories for agent execution and handles
cleanup.  Git cloning is handled by ``core.git.operations`` — this module
only manages the filesystem.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from core.config import sandbox_settings
from core.exceptions import WorkspaceError

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Create and clean up agent workspaces."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or sandbox_settings.workspace_root

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

        def _rm() -> None:
            shutil.rmtree(workspace, ignore_errors=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _rm)
        logger.info("Cleaned up workspace: %s", workspace)
