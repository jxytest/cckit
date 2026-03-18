"""Skill provisioner — deploy skills from a source directory into a sandbox
workspace so that ``claude-agent-sdk`` can discover them via
``setting_sources=["project"]``.

Security measures:
    * Skill names are validated against ``^[a-z0-9][a-z0-9-]{0,62}$``.
    * Resolved real paths must reside under the configured *skills_dir*.
    * Symbolic links are rejected.
    * Any pre-existing ``.claude/`` content in the workspace (e.g. from
      ``git clone``) is removed before provisioning to prevent injection.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

from clab.exceptions import SkillError

logger = logging.getLogger(__name__)

# Matches the official Agent Skills ``name`` constraint:
# lowercase letters, digits, and hyphens; 1–64 characters; must start
# with a letter or digit.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

# SDK discovers skills at {cwd}/.claude/skills/<skill-name>/SKILL.md
_CLAUDE_SKILLS_REL = Path(".claude") / "skills"


class SkillProvisioner:
    """Copy platform-managed skills into a sandbox workspace.

    Parameters
    ----------
    skills_dir:
        Absolute path to the directory containing skill sub-directories.
        Required — no global config fallback.
    """

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def provision(
        self,
        skill_names: list[str],
        workspace_dir: Path,
    ) -> list[str]:
        """Deploy requested skills into *workspace_dir*/.claude/skills/.

        Steps:
            1. Remove any pre-existing ``.claude/`` in the workspace
               (defends against git-cloned repos injecting settings/skills).
            2. Create ``workspace/.claude/skills/``.
            3. Copy each requested skill directory.

        Returns the list of successfully provisioned skill names.

        Raises
        ------
        SkillError
            If a requested skill does not exist, fails validation, or
            cannot be copied.
        """
        if not skill_names:
            return []

        # Step 1 — purge any existing .claude/ from the workspace
        claude_dir = workspace_dir / ".claude"
        if claude_dir.exists():
            logger.info(
                "Removing pre-existing .claude/ from workspace %s",
                workspace_dir,
            )
            await self._rmtree(claude_dir)

        # Step 2 — create target directory
        target_base = workspace_dir / _CLAUDE_SKILLS_REL
        target_base.mkdir(parents=True, exist_ok=True)

        # Step 3 — copy each skill
        deployed: list[str] = []
        for name in skill_names:
            self._validate_skill_name(name)
            source = self._resolve_skill_source(name)
            self._validate_skill_dir(source, name)

            target = target_base / name
            await self._copytree(source, target)
            deployed.append(name)
            logger.info("Provisioned skill %r → %s", name, target)

        return deployed

    def list_available(self) -> list[str]:
        """Return the names of all skills found in the source directory.

        Returns an empty list (with a warning) if the source directory
        does not exist.
        """
        if not self._skills_dir.exists():
            logger.warning(
                "Skills directory does not exist: %s", self._skills_dir
            )
            return []

        return sorted(
            d.name
            for d in self._skills_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and (d / "SKILL.md").is_file()
        )

    # ------------------------------------------------------------------
    # Validation (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_skill_name(name: str) -> None:
        """Reject names that could cause path traversal or are invalid."""
        if not _SKILL_NAME_RE.match(name):
            raise SkillError(
                f"Invalid skill name {name!r}: must match "
                f"{_SKILL_NAME_RE.pattern}",
            )

    def _resolve_skill_source(self, name: str) -> Path:
        """Resolve the skill directory path and guard against traversal."""
        candidate = (self._skills_dir / name).resolve()

        # Ensure the resolved path is still under skills_dir
        try:
            candidate.relative_to(self._skills_dir.resolve())
        except ValueError:
            raise SkillError(
                f"Skill {name!r} resolves outside the skills directory: "
                f"{candidate}",
            ) from None

        return candidate

    @staticmethod
    def _validate_skill_dir(source: Path, name: str) -> None:
        """Verify the skill directory is valid and safe."""
        if not source.exists():
            raise SkillError(f"Skill {name!r} not found: {source}")

        if source.is_symlink():
            raise SkillError(
                f"Skill {name!r} is a symbolic link (rejected for security): "
                f"{source}",
            )

        if not source.is_dir():
            raise SkillError(
                f"Skill {name!r} is not a directory: {source}",
            )

        skill_md = source / "SKILL.md"
        if not skill_md.is_file():
            raise SkillError(
                f"Skill {name!r} is missing SKILL.md: expected {skill_md}",
            )

    # ------------------------------------------------------------------
    # Async filesystem helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    async def _copytree(src: Path, dst: Path) -> None:
        """Copy a directory tree in a thread pool."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: shutil.copytree(src, dst, dirs_exist_ok=True)
            )
        except OSError as exc:
            raise SkillError(
                f"Failed to copy skill from {src} to {dst}",
                detail=str(exc),
            ) from exc

    @staticmethod
    async def _rmtree(path: Path) -> None:
        """Remove a directory tree in a thread pool."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: shutil.rmtree(path, ignore_errors=True)
        )
