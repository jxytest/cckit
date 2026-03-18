"""Sandbox configuration builder.

Translates high-level ``SandboxSettings`` into the dict format expected by
``claude-agent-sdk``'s ``ClaudeAgentOptions.sandbox``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import sandbox_settings

logger = logging.getLogger(__name__)


class SandboxConfig:
    """Build the ``sandbox`` dict consumed by the SDK."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        allow_read: list[str] | None = None,
        allow_write: list[str] | None = None,
        network_hosts: list[str] | None = None,
    ) -> None:
        self.enabled = enabled if enabled is not None else sandbox_settings.enabled
        self.allow_read = allow_read or []
        self.allow_write = allow_write or []
        self.network_hosts = network_hosts or list(sandbox_settings.network_allowed_hosts)

    def build(self, workspace_dir: Path | None = None) -> dict[str, Any]:
        """Return a dict suitable for ``ClaudeAgentOptions(sandbox=...)``.

        If *workspace_dir* is given it is automatically added to both
        ``allowRead`` and ``allowWrite``.
        """
        if not self.enabled:
            return {}

        allow_read = list(self.allow_read)
        allow_write = list(self.allow_write)

        if workspace_dir:
            ws = str(workspace_dir)
            if ws not in allow_read:
                allow_read.append(ws)
            if ws not in allow_write:
                allow_write.append(ws)

        cfg: dict[str, Any] = {"enabled": True}

        if allow_read:
            cfg["allowRead"] = allow_read
        if allow_write:
            cfg["allowWrite"] = allow_write
        if self.network_hosts:
            cfg["networkAllowedHosts"] = self.network_hosts

        logger.debug("Sandbox config: %s", cfg)
        return cfg
