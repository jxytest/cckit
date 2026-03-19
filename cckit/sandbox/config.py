"""Sandbox configuration builder.

Translates high-level ``SandboxOptions`` into the two injection points
expected by ``claude-agent-sdk``:

- ``ClaudeAgentOptions.sandbox``  — ``SandboxSettings`` TypedDict
  Controls bash process isolation behaviour (macOS Seatbelt / Linux bubblewrap).

- ``ClaudeAgentOptions.settings`` — JSON string with ``sandbox.filesystem``
  and ``sandbox.network`` keys.
  Controls OS-level path allow/deny rules and domain restrictions.
  These are merged by the SDK's ``_build_settings_value()`` with any
  existing ``settings`` JSON before being passed to the CLI via ``--settings``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxConfigBuilder:
    """Build the two sandbox config blobs consumed by the SDK.

    All parameters must be passed explicitly — no global config fallback.

    Parameters
    ----------
    enabled:
        Master switch.  When ``False``, ``build()`` returns ``(None, None)``.
    workspace_root:
        Root directory under which per-task workspaces are created.
        Added to ``allowWrite`` automatically so the agent can write to its
        own workspace; kept out of ``allowRead`` overrides (it is readable by
        default).
    allow_write:
        Extra paths the sandbox may write to beyond ``workspace_root``.
    deny_write:
        Paths explicitly blocked from writes.
    allow_read:
        Paths re-allowed for reading inside a ``deny_read`` zone.
    deny_read:
        Paths blocked from reading.  Defaults to ``["~/"]`` to prevent the
        agent from accessing the home directory (e.g. ``~/.ssh``).
    allowed_domains:
        Outbound network domains allowed for Bash subprocesses.
        Empty list means *allow all*.
    auto_allow_bash:
        ``autoAllowBashIfSandboxed`` — skip per-command approval prompts
        when running inside the sandbox.
    excluded_commands:
        Commands that run *outside* the sandbox (e.g. ``["git", "docker"]``).
    allow_unsandboxed_commands:
        ``allowUnsandboxedCommands`` — whether ``dangerouslyDisableSandbox``
        is honoured.  Set to ``False`` for stricter enforcement.
    enable_weaker_nested_sandbox:
        ``enableWeakerNestedSandbox`` — weaker isolation for unprivileged
        Docker environments (Linux only).  Reduces security.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        workspace_root: Path = Path("/tmp/cckit_workspaces"),
        allow_write: list[str] | None = None,
        deny_write: list[str] | None = None,
        allow_read: list[str] | None = None,
        deny_read: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        auto_allow_bash: bool = True,
        excluded_commands: list[str] | None = None,
        allow_unsandboxed_commands: bool = True,
        enable_weaker_nested_sandbox: bool = False,
    ) -> None:
        self.enabled = enabled
        self.workspace_root = workspace_root
        self.allow_write = allow_write or []
        self.deny_write = deny_write or []
        self.allow_read = allow_read or []
        self.deny_read = deny_read if deny_read is not None else ["~/"]
        self.allowed_domains = allowed_domains or []
        self.auto_allow_bash = auto_allow_bash
        self.excluded_commands = excluded_commands or []
        self.allow_unsandboxed_commands = allow_unsandboxed_commands
        self.enable_weaker_nested_sandbox = enable_weaker_nested_sandbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self, workspace_dir: Path | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Return ``(sandbox_dict, settings_json)``.

        Parameters
        ----------
        workspace_dir:
            The concrete per-task workspace directory created by
            ``WorkspaceManager``.  When provided it is added to
            ``sandbox.filesystem.allowWrite`` so the agent can write there.

        Returns
        -------
        sandbox_dict:
            Dict suitable for ``SandboxSettings(**sandbox_dict)`` and passed
            to ``ClaudeAgentOptions.sandbox``.  ``None`` when disabled.
        settings_json:
            JSON string with ``sandbox.filesystem`` / ``sandbox.network``
            keys, passed to ``ClaudeAgentOptions.settings``.  ``None`` when
            there are no filesystem / network rules to inject.
        """
        if not self.enabled:
            return None, None

        # ── SandboxSettings dict (bash process behaviour) ──────────────
        sandbox: dict[str, Any] = {
            "enabled": True,
            "autoAllowBashIfSandboxed": self.auto_allow_bash,
            "allowUnsandboxedCommands": self.allow_unsandboxed_commands,
        }
        if self.excluded_commands:
            sandbox["excludedCommands"] = self.excluded_commands
        if self.enable_weaker_nested_sandbox:
            sandbox["enableWeakerNestedSandbox"] = True

        logger.debug("SandboxSettings dict: %s", sandbox)

        # ── settings JSON (OS-level filesystem & network rules) ─────────
        # Build allowWrite: workspace_dir first, then any extra paths
        allow_write: list[str] = []
        if workspace_dir:
            allow_write.append(str(workspace_dir))
        allow_write.extend(self.allow_write)

        filesystem: dict[str, Any] = {}
        if allow_write:
            filesystem["allowWrite"] = allow_write
        if self.deny_write:
            filesystem["denyWrite"] = self.deny_write
        if self.allow_read:
            filesystem["allowRead"] = self.allow_read
        if self.deny_read:
            filesystem["denyRead"] = self.deny_read

        network: dict[str, Any] = {}
        if self.allowed_domains:
            network["allowedDomains"] = self.allowed_domains

        settings_obj: dict[str, Any] = {}
        if filesystem:
            settings_obj.setdefault("sandbox", {})["filesystem"] = filesystem
        if network:
            settings_obj.setdefault("sandbox", {})["network"] = network

        # ── permissions.deny (Claude built-in tool rules) ───────────────
        # sandbox.filesystem.denyRead only restricts Bash subprocesses (OS-level).
        # To also block Claude's built-in Read/Glob/Grep tools from accessing the
        # same paths, we mirror deny_read into permissions.deny rules using the
        # ~/path syntax documented in the Claude Code permissions reference.
        permission_deny: list[str] = []
        for path in self.deny_read:
            # Normalise: "~/" → "~/" prefix for the rule specifier
            specifier = path.rstrip("/")   # e.g. "~/" → "~", "~/.ssh" → "~/.ssh"
            if specifier == "~":
                # deny entire home directory: Read(~/**) covers all files inside
                permission_deny.append("Read(~/**)")
                permission_deny.append("Edit(~/**)")
            else:
                permission_deny.append(f"Read({specifier}/**)")
                permission_deny.append(f"Edit({specifier}/**)")
        for path in self.deny_write:
            specifier = path.rstrip("/")
            if specifier == "~":
                permission_deny.append("Edit(~/**)")
            else:
                permission_deny.append(f"Edit({specifier}/**)")

        if permission_deny:
            settings_obj.setdefault("permissions", {})["deny"] = permission_deny

        settings_json = json.dumps(settings_obj) if settings_obj else None
        logger.debug("Sandbox settings JSON: %s", settings_json)

        return sandbox, settings_json
