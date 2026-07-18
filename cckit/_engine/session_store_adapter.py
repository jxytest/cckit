"""File-backed ``SessionStore`` adapter for the Claude Agent SDK.

Implements the SDK's :class:`SessionStore` protocol on top of a local
directory tree. Each ``(project_key, session_id)`` maps to a JSONL file under
a caller-supplied root (default ``~/.claude/projects/`` — the same layout the
SDK's built-in persistence uses, so a ``FileSessionStore`` pointed at that
root is a true mirror).

For cross-machine resume, implement a remote ``SessionStore`` (Redis, object
storage, DB) instead — this adapter is process/disk-local.

The SDK passes ``project_key`` already sanitized (a directory *name*, not a
path — see ``claude_agent_sdk.project_key_for_directory``). It is used
verbatim as a path segment under the root; it is NOT re-canonicalized.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default root mirrors the SDK's built-in persistence location so that, out of
# the box, FileSessionStore reads/writes the same files ``Runner.read_session``
# operates on. Override via the ``directory`` constructor param for isolation
# (e.g. point at a tempdir in tests).
_DEFAULT_ROOT = Path.home() / ".claude" / "projects"


class FileSessionStore:
    """Persist session transcripts as JSONL files under a directory tree.

    Layout::

        <root>/<project_key>/<session_id>.jsonl
        <root>/<project_key>/<subpath>/<session_id>.jsonl   (subagent transcripts)

    ``project_key`` is treated as an opaque sanitized name (per the SDK
    contract) and used verbatim as a path segment.

    Parameters
    ----------
    directory:
        Storage root. Defaults to ``~/.claude/projects/`` so the store mirrors
        the SDK's built-in persistence. Pass a fresh tempdir per instance for
        test isolation (the SDK conformance suite creates a new store per
        contract and expects a clean slate).
    """

    def __init__(self, *, directory: str | Path | None = None) -> None:
        self._root = Path(directory).expanduser() if directory else _DEFAULT_ROOT

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _session_file(self, key: dict[str, Any]) -> Path:
        """Resolve the JSONL file for a SessionKey.

        ``project_key`` is an opaque sanitized name (not a path); use it
        verbatim. ``subpath`` (when present) is a forward-slash-relative
        path under the project dir (e.g. ``subagents/agent-<id>``).
        """
        project_dir = self._root / key["project_key"]
        subpath = key.get("subpath")
        if subpath:
            # subpath is POSIX-style; Path handles forward slashes on Windows
            return project_dir / subpath / f'{key["session_id"]}.jsonl'
        return project_dir / f'{key["session_id"]}.jsonl'

    # ------------------------------------------------------------------
    # Required protocol methods
    # ------------------------------------------------------------------

    async def append(self, key: dict[str, Any], entries: list[Any]) -> None:
        """Append a batch of transcript entries to the session JSONL file.

        Entries carrying a ``uuid`` are deduplicated against existing lines so
        retried batches (the SDK retries failed appends up to 3 times) do not
        produce duplicates. Dedup state is held in-memory for the duration of
        this call only — see the concurrency note below.
        """
        if not entries:
            return
        path = self._session_file(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_uuids = self._read_existing_uuids(path)
        new_lines: list[str] = []
        for entry in entries:
            uid = entry.get("uuid") if isinstance(entry, dict) else None
            if isinstance(uid, str) and uid in existing_uuids:
                continue  # idempotent: already persisted
            if isinstance(uid, str):
                existing_uuids.add(uid)
            try:
                new_lines.append(json.dumps(entry, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                logger.debug("Skipping non-serializable session entry", exc_info=True)

        if not new_lines:
            return

        # Append-mode writes are atomic at the line level on POSIX; the SDK
        # guarantees append calls for a given session are serialized within a
        # process, so the read-then-write dedup is safe for the common case.
        # Cross-process concurrency may produce duplicates of uuid-less
        # entries (titles/tags), which the protocol explicitly allows.
        try:
            with path.open("a", encoding="utf-8") as fh:
                for line in new_lines:
                    fh.write(line + "\n")
        except OSError:
            logger.warning("Failed to append session entries to %s", path, exc_info=True)

    async def load(self, key: dict[str, Any]) -> list[Any] | None:
        """Load a full session for resume. Returns ``None`` if never written."""
        path = self._session_file(key)
        if not path.exists():
            return None
        entries: list[Any] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            logger.debug("Failed to load session file %s", path, exc_info=True)
            return None
        return entries or None

    # ------------------------------------------------------------------
    # Optional protocol methods (on-disk implementations)
    # ------------------------------------------------------------------

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]:
        """List top-level session IDs + mtimes under a project directory.

        Only top-level ``<session_id>.jsonl`` files are returned; subagent
        transcripts live in subdirectories and are excluded.
        """
        project_dir = self._root / project_key
        if not project_dir.exists():
            return []
        results: list[dict[str, Any]] = []
        for entry in project_dir.iterdir():
            if not entry.is_file() or entry.suffix != ".jsonl":
                continue
            try:
                mtime_ms = int(entry.stat().st_mtime_ns // 1_000_000)
            except OSError:
                mtime_ms = 0
            results.append({"session_id": entry.stem, "mtime": mtime_ms})
        return results

    async def delete(self, key: dict[str, Any]) -> None:
        """Delete a session file. Cascades to subpath files when no subpath given.

        Deleting a main-transcript key (no ``subpath``) must remove the top-level
        transcript AND every subagent transcript in any subdirectory (the SDK's
        ``subagents/agent-<id>`` layout is two levels deep), so subagent
        transcripts are not orphaned.
        """
        project_dir = self._root / key["project_key"]
        subpath = key.get("subpath")
        target_name = f'{key["session_id"]}.jsonl'
        if subpath:
            self._safe_remove(project_dir / subpath / target_name)
            return
        # Cascade: remove the main transcript + every <session_id>.jsonl in
        # any subdirectory (subagent transcripts live at arbitrary depth).
        self._safe_remove(project_dir / target_name)
        if project_dir.exists():
            for match in project_dir.rglob(target_name):
                if match.is_file():
                    self._safe_remove(match)

    async def list_subkeys(self, key: dict[str, Any]) -> list[str]:
        """List subpath directories containing a transcript for this session.

        Returns full POSIX-style subpaths (e.g. ``subagents/agent-<id>``) so
        the SDK's ``list_subagents_from_store`` (which filters on the
        ``subagents/`` prefix) discovers subagent transcripts. The main
        transcript (at the project-dir root) is excluded — it is not a subkey.
        Results are sorted for deterministic order.
        """
        project_dir = self._root / key["project_key"]
        if not project_dir.exists():
            return []
        target = f'{key["session_id"]}.jsonl'
        subkeys: list[str] = []
        for match in project_dir.rglob(target):
            if not match.is_file():
                continue
            try:
                rel = match.parent.relative_to(project_dir)
            except ValueError:
                continue
            # The main transcript lives directly under project_dir (rel == ".");
            # it is not a subkey, skip it.
            if rel == Path("."):
                continue
            subkeys.append(rel.as_posix())
        return sorted(subkeys)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_existing_uuids(path: Path) -> set[str]:
        """Read existing entry uuids from ``path`` for dedup. Empty if unreadable."""
        uuids: set[str] = set()
        if not path.exists():
            return uuids
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = obj.get("uuid") if isinstance(obj, dict) else None
                if isinstance(uid, str):
                    uuids.add(uid)
        except OSError:
            logger.debug("Failed to read existing session file %s", path, exc_info=True)
        return uuids

    @staticmethod
    def _safe_remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove %s", path, exc_info=True)
