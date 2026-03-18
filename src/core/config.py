"""Pydantic-based settings split by domain.

Core infrastructure settings are defined here.  Business-specific settings
(GitLab, database, etc.) belong in the upper business layer, not in core/.

Usage:
    from core.config import anthropic_settings, sandbox_settings
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Core infrastructure settings
# ---------------------------------------------------------------------------


class AnthropicSettings(BaseSettings):
    """Anthropic / Claude SDK configuration."""

    model_config = {"env_prefix": "ANTHROPIC_", "case_sensitive": False}

    api_key: str = ""
    base_url: str = ""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 16384
    max_turns: int = 50
    permission_mode: str = "bypassPermissions"
    timeout_seconds: int = 300


class SandboxSettings(BaseSettings):
    """Sandbox / workspace isolation configuration."""

    model_config = {"env_prefix": "SANDBOX_", "case_sensitive": False}

    enabled: bool = False
    workspace_root: Path = Path("/tmp/agent_workspaces")
    network_allowed_hosts: list[str] = []
    max_file_size_mb: int = 50


class PlatformSettings(BaseSettings):
    """Platform-level configuration."""

    model_config = {"env_prefix": "PLATFORM_", "case_sensitive": False}

    debug: bool = False
    log_level: str = "INFO"
    max_concurrent_agents: int = 5
    skills_dir: Path = Path("/opt/agent-platform/skills")


# ---------------------------------------------------------------------------
# Singletons — import these directly
# ---------------------------------------------------------------------------

anthropic_settings = AnthropicSettings()
sandbox_settings = SandboxSettings()
platform_settings = PlatformSettings()
