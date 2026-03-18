"""Skill provisioning — deploy skills into sandbox workspaces.

Usage::

    from core.agent.skill import SkillProvisioner

    provisioner = SkillProvisioner(skills_dir=Path("/opt/agent-platform/skills"))
    deployed = await provisioner.provision(["playwright-testing"], workspace_dir)
"""

from core.agent.skill.provisioner import SkillProvisioner

__all__ = ["SkillProvisioner"]
