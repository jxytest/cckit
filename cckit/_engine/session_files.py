"""Claude Code session file persistence (~/.claude/projects/).

Read/restore helpers that let a caller move a session's JSONL transcript
between hosts so ``--resume`` keeps working.  All functions are best-effort:
they log and return a falsy value instead of raising.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_session(session_id: str, workspace_dir: Path) -> str | None:
    """Read a Claude Code session JSONL from ~/.claude/projects/."""
    try:
        from claude_agent_sdk._internal.sessions import (
            _canonicalize_path,
            _find_project_dir,
        )
        project_dir = _find_project_dir(_canonicalize_path(str(workspace_dir)))
        if project_dir is None:
            return None
        session_file = project_dir / f'{session_id}.jsonl'
        if session_file.exists():
            return session_file.read_text(encoding='utf-8')
    except Exception:
        logger.debug('Failed to read session JSONL for %s', session_id, exc_info=True)
    return None


def restore_session(session_id: str, workspace_dir: Path, content: str) -> None:
    """Restore a persisted session JSONL so --resume works."""
    try:
        from claude_agent_sdk._internal.sessions import (
            _canonicalize_path,
            _get_project_dir,
        )
        project_dir = _get_project_dir(_canonicalize_path(str(workspace_dir)))
        project_dir.mkdir(parents=True, exist_ok=True)
        session_file = project_dir / f'{session_id}.jsonl'
        if not session_file.exists():
            session_file.write_text(content, encoding='utf-8')
            logger.info('Restored session JSONL for %s to %s', session_id, session_file)
    except Exception:
        logger.warning('Failed to restore session JSONL for %s', session_id, exc_info=True)


def read_session_dir(workspace_dir: Path) -> dict[str, bytes] | None:
    """Read all session files from the project directory (recursively).

    Returns ``{relative_posix_path: content}`` for every ``.jsonl``
    and ``.meta.json`` file (including subagent files in
    subdirectories), or *None* when the directory does not exist or
    contains no files.
    """
    try:
        from claude_agent_sdk._internal.sessions import (
            _canonicalize_path,
            _find_project_dir,
        )
        project_dir = _find_project_dir(_canonicalize_path(str(workspace_dir)))
        if project_dir is None:
            return None
        files: dict[str, bytes] = {}
        for entry in project_dir.rglob('*'):
            if entry.is_file() and entry.suffix in ('.jsonl', '.json'):
                # Use forward-slash relative path as key to preserve
                # subdirectory structure (e.g. subagent sessions).
                rel = entry.relative_to(project_dir).as_posix()
                files[rel] = entry.read_bytes()
        return files or None
    except Exception:
        logger.debug('Failed to read session dir for %s', workspace_dir, exc_info=True)
    return None


def restore_session_dir(workspace_dir: Path, files: dict[str, bytes]) -> None:
    """Restore multiple session files to the project directory.

    Keys may contain forward-slash separated relative paths for
    subagent files stored in subdirectories.  Parent directories are
    created automatically.  Skips files that already exist
    (idempotent).
    """
    try:
        from claude_agent_sdk._internal.sessions import (
            _canonicalize_path,
            _get_project_dir,
        )
        project_dir = _get_project_dir(_canonicalize_path(str(workspace_dir)))
        project_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            target = project_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
        logger.info(
            'Restored %d session file(s) to %s', len(files), project_dir,
        )
    except Exception:
        logger.warning(
            'Failed to restore session dir for %s', workspace_dir, exc_info=True,
        )
