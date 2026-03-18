"""Async wrappers around common git CLI operations.

All functions shell out to ``git`` via ``asyncio.create_subprocess_exec``
so they work in both sandbox and non-sandbox environments.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from cckit.exceptions import GitOperationError

logger = logging.getLogger(__name__)


async def run_git(
    *args: str,
    cwd: Path | str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run an arbitrary git command and return stdout; raise on non-zero exit.

    This is the public low-level API for running any ``git`` sub-command.
    Higher-level helpers like :func:`clone`, :func:`status`, :func:`diff`
    are built on top of it.

    Parameters
    ----------
    extra_env:
        Additional environment variables merged on top of ``os.environ``.
        Use this to inject per-task credentials (e.g. ``GIT_ASKPASS``,
        ``GIT_TOKEN``) so that concurrent clones with different tokens
        don't interfere.

    Example::

        sha = await run_git("rev-parse", "HEAD", cwd=repo_dir)
        await run_git("config", "user.email", "bot@example.com", cwd=repo_dir)
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


# Backward-compatible alias for internal callers
_run_git = run_git


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
    await run_git(*args, extra_env=extra_env)
    logger.info("Cloned %s → %s", repo_url, target)
    return target


async def create_branch(name: str, *, cwd: Path) -> None:
    """Create and checkout a new branch."""
    await run_git("checkout", "-b", name, cwd=cwd)
    logger.info("Created branch %s in %s", name, cwd)


async def add_all(*, cwd: Path) -> None:
    """Stage all changes."""
    await run_git("add", "-A", cwd=cwd)


async def commit(message: str, *, cwd: Path) -> str:
    """Commit staged changes.  Returns the commit SHA."""
    await run_git("commit", "-m", message, cwd=cwd)
    sha = await run_git("rev-parse", "HEAD", cwd=cwd)
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
    await run_git(*args, cwd=cwd, extra_env=extra_env)
    logger.info("Pushed to %s/%s from %s", remote, branch, cwd)


async def status(*, cwd: Path, short: bool = False) -> str:
    """Return ``git status`` output.

    Parameters
    ----------
    short:
        If ``True``, use ``--short`` (porcelain-like) format.
    """
    args = ["status"]
    if short:
        args.append("--short")
    return await run_git(*args, cwd=cwd)


async def diff(
    *,
    cwd: Path,
    name_only: bool = False,
    staged: bool = False,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Return ``git diff`` output.

    Parameters
    ----------
    name_only:
        Only show changed file names (``--name-only``).
    staged:
        Show staged (cached) changes instead of working-tree changes.
    extra_env:
        Per-task environment variables for credential isolation.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if name_only:
        args.append("--name-only")
    return await run_git(*args, cwd=cwd, extra_env=extra_env)
