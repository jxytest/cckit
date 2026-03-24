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
        self.deny_write = deny_write if deny_write is not None else ["~/"]
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
            ``sandbox.filesystem.allowWrite`` **and** ``allowRead``
            so the agent can only read/write inside its own workspace.
        Returns
        -------
        sandbox_dict:
            Dict suitable for ``SandboxSettings(**sandbox_dict)`` and passed
            to ``ClaudeAgentOptions.sandbox``.  ``None`` when disabled.
        settings_json:
            JSON string with ``sandbox``, ``permissions`` keys, passed to
            ``ClaudeAgentOptions.settings``.  ``None`` when there are no
            rules to inject.
        """
        if not self.enabled:
            return None, None

        sandbox = self._build_sandbox_settings()
        allow_read, allow_write, deny_read, deny_write = self._collect_paths(workspace_dir)

        settings_obj: dict[str, Any] = {}
        self._attach_filesystem_and_network(
            settings_obj,
            allow_read=allow_read,
            allow_write=allow_write,
            deny_read=deny_read,
            deny_write=deny_write,
        )
        self._attach_permissions(
            settings_obj,
            allow_read=allow_read,
            allow_write=allow_write,
            deny_read=deny_read,
            deny_write=deny_write,
        )

        settings_json = json.dumps(settings_obj) if settings_obj else None
        logger.info("Sandbox settings JSON: %s", settings_json)
        logger.info("Sandbox JSON: %s", sandbox)
        return sandbox, settings_json

    def _build_sandbox_settings(self) -> dict[str, Any]:
        """Build ``SandboxSettings`` dict for ``ClaudeAgentOptions.sandbox``."""
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
        return sandbox

    def _collect_paths(
            self, workspace_dir: Path | None
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Collect effective allow/deny paths with workspace overrides."""
        ws = str(workspace_dir) if workspace_dir else None
        allow_read = ([ws] if ws else []) + list(self.allow_read)
        allow_write = ([ws] if ws else []) + list(self.allow_write)
        deny_read = self.deny_read
        deny_write = self.deny_write
        return allow_read, allow_write, deny_read, deny_write

    def _attach_filesystem_and_network(
            self,
            settings_obj: dict[str, Any],
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> None:
        """Attach ``sandbox.filesystem`` and ``sandbox.network`` sections."""
        filesystem: dict[str, Any] = {}
        if allow_write:
            filesystem["allowWrite"] = allow_write
        if deny_write:
            filesystem["denyWrite"] = deny_write
        if allow_read:
            filesystem["allowRead"] = allow_read
        if deny_read:
            filesystem["denyRead"] = deny_read

        network: dict[str, Any] = {}
        if self.allowed_domains:
            network["allowedDomains"] = self.allowed_domains

        if filesystem:
            settings_obj.setdefault("sandbox", {})["filesystem"] = filesystem
        if network:
            settings_obj.setdefault("sandbox", {})["network"] = network

    def _attach_permissions(
            self,
            settings_obj: dict[str, Any],
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> None:
        """Attach permission allow/deny rules for built-in tools."""
        read_tools = ("Read",)
        write_tools = ("Read", "Edit", "Write")
        write_only_tools = ("Edit", "Write")
        raw_allow = (
                self._permission_rules(allow_read, read_tools)
                + self._permission_rules(allow_write, write_tools)
        )
        permission_allow = list(dict.fromkeys(raw_allow))
        filtered_deny_read, filtered_deny_write = self._filter_conflicting_denies(
            allow_read=allow_read, allow_write=allow_write,
            deny_read=deny_read, deny_write=deny_write,
        )
        raw_deny = (
                self._permission_rules(filtered_deny_read, write_tools)  # 禁止所有访问
                + self._permission_rules(filtered_deny_write, write_only_tools)  # 只禁止写入
        )
        permission_deny = list(dict.fromkeys(raw_deny))
        if permission_allow:
            settings_obj.setdefault("permissions", {})["allow"] = permission_allow
        if permission_deny:
            settings_obj.setdefault("permissions", {})["deny"] = permission_deny

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a path: strip trailing slash and expand ``~``."""
        stripped = path.rstrip("/")
        if stripped == "~" or stripped.startswith("~/"):
            stripped = str(Path(stripped).expanduser())
        return stripped

    @staticmethod
    def _filter_conflicting_denies(
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> tuple[list[str], list[str]]:
        """Remove deny paths that would shadow allow paths."""
        allow_resolved = {
            SandboxConfigBuilder._normalize_path(p)
            for p in allow_read + allow_write
        }

        def is_covered_by_allow(deny_path: str) -> bool:
            deny_norm = SandboxConfigBuilder._normalize_path(deny_path)
            for allow_path in allow_resolved:
                if allow_path == deny_norm or allow_path.startswith(deny_norm + "/"):
                    return True
            return False

        filtered_deny_read = [p for p in deny_read if not is_covered_by_allow(p)]
        filtered_deny_write = [p for p in deny_write if not is_covered_by_allow(p)]
        return filtered_deny_read, filtered_deny_write

    @staticmethod
    def _permission_rules(
            paths: list[str], tools: tuple[str, ...]
    ) -> list[str]:
        """Generate permission rules for *paths* × *tools*.
        Each path is normalised (trailing ``/`` stripped) and combined with
        every tool name to produce rules like ``Edit(/tmp/work/**)``.
        """
        rules: list[str] = []
        for path in paths:
            specifier = path.rstrip("/")
            for tool in tools:
                rule = f"{tool}({specifier}/**)"
                if rule not in rules:
                    rules.append(rule)
        return rules
