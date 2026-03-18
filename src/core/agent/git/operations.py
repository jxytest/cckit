"""Async wrappers around common git CLI operations.

All functions shell out to ``git`` via ``asyncio.create_subprocess_exec``
so they work in both sandbox and non-sandbox environments.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.exceptions import GitOperationError

logger = logging.getLogger(__name__)


async def _run_git(
    *args: str,
    cwd: Path | str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run a git command and return stdout; raise on non-zero exit.

    Parameters
    ----------
    extra_env:
        Additional environment variables merged on top of ``os.environ``.
        Use this to inject per-task credentials (e.g. ``GIT_ASKPASS``,
        ``GIT_TOKEN``) so that concurrent clones with different tokens
        don't interfere.
    """
    import os

    cmd = ["git", *args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)

    env: dict[str, str] | None = None
    if extra_env:
        env = {**os.environ, **extra_env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise GitOperationError(
            f"git {args[0]} failed (exit {proc.returncode})",
            detail=stderr.decode(errors="replace"),
        )
    return stdout.decode(errors="replace").strip()


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

async def clone(
    repo_url: str,
    target: Path,
    *,
    branch: str = "",
    depth: int = 1,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Clone a repository.  Returns *target*.

    Parameters
    ----------
    extra_env:
        Per-task environment variables for credential isolation.
        Example: inject a helper script via ``GIT_ASKPASS`` so each
        concurrent clone authenticates with its own token.
    """
    args = ["clone", f"--depth={depth}"]
    if branch:
        args.extend(["--branch", branch])
    args.extend([repo_url, str(target)])
    await _run_git(*args, extra_env=extra_env)
    logger.info("Cloned %s → %s", repo_url, target)
    return target


async def create_branch(name: str, *, cwd: Path) -> None:
    """Create and checkout a new branch."""
    await _run_git("checkout", "-b", name, cwd=cwd)
    logger.info("Created branch %s in %s", name, cwd)


async def add_all(*, cwd: Path) -> None:
    """Stage all changes."""
    await _run_git("add", "-A", cwd=cwd)


async def commit(message: str, *, cwd: Path) -> str:
    """Commit staged changes.  Returns the commit SHA."""
    await _run_git("commit", "-m", message, cwd=cwd)
    sha = await _run_git("rev-parse", "HEAD", cwd=cwd)
    logger.info("Committed %s in %s", sha[:8], cwd)
    return sha


async def push(
    remote: str = "origin",
    branch: str = "",
    *,
    cwd: Path,
    set_upstream: bool = True,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Push to remote.

    Parameters
    ----------
    extra_env:
        Per-task environment variables for credential isolation.
    """
    args = ["push"]
    if set_upstream:
        args.append("--set-upstream")
    args.append(remote)
    if branch:
        args.append(branch)
    await _run_git(*args, cwd=cwd, extra_env=extra_env)
    logger.info("Pushed to %s/%s from %s", remote, branch, cwd)
