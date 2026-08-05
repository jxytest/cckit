"""Skill planning — collect, provision, and repair an agent tree's skills."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cckit.skill.provisioner import SkillProvisioner

if TYPE_CHECKING:
    from cckit.agent import Agent


def collect_all_skills(agent: Agent) -> list[str]:
    """Collect deduplicated skill names from *agent* and its sub-agents."""
    seen: set[str] = set()
    result: list[str] = []
    for name in agent.skills:
        if name not in seen:
            seen.add(name)
            result.append(name)
    for sub in agent.sub_agents:
        for name in sub.skills:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


async def provision_agent_skills(
        agent: Agent,
        workspace_dir: Path,
        default_provisioner: SkillProvisioner,
) -> None:
    """Provision all declared skills (top-level + sub-agents) into the
    workspace's ``.claude/skills/``. No-op when the agent declares none.

    Collects ALL skills across the agent tree first, then provisions
    once — ``SkillProvisioner.provision()`` purges ``.claude/`` on each
    call, so multiple calls would wipe earlier results.
    """
    all_skills = collect_all_skills(agent)
    if not all_skills:
        return
    # Use the agent-level skills_dir if specified; otherwise fall back to
    # the first sub-agent that declares one.
    effective_dir = agent.skills_dir
    if not effective_dir:
        for sub in agent.sub_agents:
            if sub.skills_dir:
                effective_dir = sub.skills_dir
                break
    provisioner = (
        SkillProvisioner(skills_dir=effective_dir)
        if effective_dir
        else default_provisioner
    )
    await provisioner.provision(all_skills, workspace_dir)


def needs_skill_repair(agent: Agent, workspace_dir: Path) -> bool:
    """True when the agent declares skills but any is missing from the
    workspace's ``.claude/skills/``.

    This detects a resumed turn whose workspace dir survived (so cckit
    did not recreate it and skipped provisioning) yet the skills are
    absent — typically because the host wiped the tmpdir between turns
    and recreated an empty dir before cckit ran.
    """
    skills = collect_all_skills(agent)
    if not skills:
        return False
    base = workspace_dir / ".claude" / "skills"
    if not base.is_dir():
        return True
    return not all(
        (base / name / "SKILL.md").is_file() for name in skills
    )
