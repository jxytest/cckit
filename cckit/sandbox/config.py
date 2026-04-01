"""Sandbox configuration builder.

Translates high-level ``SandboxOptions`` into a single unified settings JSON
string expected by ``ClaudeAgentOptions.settings``.

The settings JSON contains a single ``sandbox`` section that merges:
- Behaviour flags (``enabled``, ``autoAllowBashIfSandboxed``, etc.)
- ``filesystem`` rules (``allowWrite``, ``denyWrite``, ``denyRead``, ``allowRead``)
- ``network`` rules (``allowedDomains``, ``deniedDomains``)

And a ``permissions`` section for built-in tool (Read/Edit/Write) access control.

**Important:** ``ClaudeAgentOptions.sandbox`` must be set to ``None`` when using
the unified settings JSON approach, because the SDK's ``_build_settings_value()``
would otherwise overwrite the ``sandbox`` section in settings JSON with the
``SandboxSettings`` TypedDict, losing filesystem/network rules.

Sandbox-runtime schema reference
--------------------------------
- ``network.allowedDomains`` (required array) — empty = block all outbound traffic
- ``network.deniedDomains`` (required array) — takes precedence over allowedDomains
- ``filesystem.denyRead`` (required array) — paths denied for reading
- ``filesystem.allowRead`` (optional array) — re-allow within denied regions
- ``filesystem.allowWrite`` (required array) — paths allowed for writing (default: deny all)
- ``filesystem.denyWrite`` (required array) — overrides within allowWrite
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Claude Code's internal harness directory.  Must remain readable even
# when ``deny_read`` covers ``~/``, because Claude Code stores tool-results,
# session-memory, plans, and scratchpad files under this path.
#
# Claude Code's permission system evaluates deny rules (step 3) *before*
# the internal-path allow check (step 7) in ``checkReadPermissionForTool``.
# A blanket ``Read(~/**)`` permission deny rule therefore blocks reads to
# ``~/.claude/projects/.../tool-results/`` — breaking sub-agent output
# retrieval and other internal operations.
#
# The OS-level sandbox (``filesystem.denyRead`` / ``filesystem.allowRead``)
# uses "most-specific-wins" semantics, so ``allowRead: ["~/.claude/"]``
# correctly carves out an exception within ``denyRead: ["~/"]``.
# But the permissions layer has no such override — deny is absolute.
#
# Solution: keep ``~/`` in OS sandbox denyRead + add ``~/.claude/`` to
# OS sandbox allowRead (most-specific-wins handles it).  For the
# permissions layer, replace the broad ``~/`` deny with granular
# sensitive-directory denies that skip ``~/.claude/``.
_CLAUDE_HARNESS_DIR = "~/.claude/"

# Well-known sensitive directories under ``~/`` that should remain
# denied for reading when the original intent is to block ``~/``.
# ``~/.claude/`` is intentionally absent — it must stay readable.
_SENSITIVE_HOME_DIRS = [
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config",
    "~/.local",
    "~/.kube",
    "~/.docker",
    "~/.npmrc",
    "~/.netrc",
    "~/.bash_history",
    "~/.zsh_history",
]


class SandboxConfigBuilder:
    """Build the unified sandbox settings JSON consumed by the SDK.

    All parameters must be passed explicitly — no global config fallback.

    Parameters
    ----------
    enabled:
        Master switch.  When ``False``, ``build()`` returns ``None``.
    allow_write:
        Extra paths the sandbox may write to beyond the task workspace.
    deny_write:
        Paths explicitly blocked from writes.  Defaults to ``[]`` because
        writes are already deny-all by default in sandbox-runtime.
    allow_read:
        Paths re-allowed for reading inside a ``deny_read`` zone.
    deny_read:
        Paths blocked from reading.  Defaults to ``["~/"]`` to prevent the
        agent from accessing the home directory (e.g. ``~/.ssh``).
    allowed_domains:
        Outbound network domains allowed for Bash subprocesses.
        Empty list means *block all* (sandbox-runtime allow-only semantics).
    denied_domains:
        Domains explicitly blocked even if matched by ``allowed_domains``.
        Takes precedence over ``allowed_domains``.
    auto_allow_bash:
        ``autoAllowBashIfSandboxed`` — skip per-command approval prompts
        when running inside the sandbox.
    excluded_commands:
        Commands that run *outside* the sandbox (e.g. ``["git", "docker"]``).
    allow_unsandboxed_commands:
        ``allowUnsandboxedCommands`` — whether ``dangerouslyDisableSandbox``
        is honoured.  Defaults to ``False`` for stricter enforcement to
        prevent agents from trivially escaping sandbox restrictions.
    enable_weaker_nested_sandbox:
        ``enableWeakerNestedSandbox`` — weaker isolation for unprivileged
        Docker environments (Linux only).  Reduces security.
    """

    def __init__(
            self,
            *,
            enabled: bool = False,
            allow_write: list[str] | None = None,
            deny_write: list[str] | None = None,
            allow_read: list[str] | None = None,
            deny_read: list[str] | None = None,
            allowed_domains: list[str] | None = None,
            denied_domains: list[str] | None = None,
            auto_allow_bash: bool = True,
            excluded_commands: list[str] | None = None,
            allow_unsandboxed_commands: bool = False,
            enable_weaker_nested_sandbox: bool = False,
    ) -> None:
        self.enabled = enabled
        self.allow_write = allow_write or []
        # deny_write defaults to [] — writes are already deny-all by default
        self.deny_write = deny_write or []
        self.allow_read = allow_read or []
        self.deny_read = deny_read if deny_read is not None else ["~/"]
        self.allowed_domains = allowed_domains or []
        self.denied_domains = denied_domains or []
        self.auto_allow_bash = auto_allow_bash
        self.excluded_commands = excluded_commands or []
        self.allow_unsandboxed_commands = allow_unsandboxed_commands
        self.enable_weaker_nested_sandbox = enable_weaker_nested_sandbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
            self, workspace_dir: Path | None = None
    ) -> str | None:
        """Return a unified settings JSON string, or ``None`` when disabled.

        Parameters
        ----------
        workspace_dir:
            The concrete per-task workspace directory created by
            ``WorkspaceManager``.  When provided it is added to
            ``sandbox.filesystem.allowWrite`` **and** ``allowRead``
            so the agent can only read/write inside its own workspace.

        Returns
        -------
        settings_json:
            JSON string with a unified ``sandbox`` section (behaviour flags +
            filesystem + network) and ``permissions`` section, passed to
            ``ClaudeAgentOptions.settings``.  ``None`` when disabled.

        Notes
        -----
        When using this output, set ``ClaudeAgentOptions.sandbox = None``
        to avoid the SDK overwriting the ``sandbox`` section in settings JSON.
        """
        if not self.enabled:
            return None

        allow_read, allow_write, deny_read, deny_write = self._collect_paths(workspace_dir)

        # For the permissions layer, replace broad deny_read entries that
        # cover ~/.claude/ with granular sensitive-dir denies.  The OS
        # sandbox keeps the original deny_read (most-specific-wins handles
        # the ~/.claude/ allowRead carve-out).
        perm_deny_read = self._permissions_deny_read(deny_read)

        settings_obj: dict[str, Any] = {}

        # Build unified sandbox section: behaviour flags + filesystem + network
        self._attach_unified_sandbox(
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
            deny_read=perm_deny_read,
            deny_write=deny_write,
        )

        settings_json = json.dumps(settings_obj)
        logger.info("Sandbox settings JSON: %s", settings_json)
        print("Sandbox settings JSON:", settings_json)
        return settings_json

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _attach_unified_sandbox(
            self,
            settings_obj: dict[str, Any],
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> None:
        """Build the unified ``sandbox`` section with behaviour + filesystem + network.

        All three sub-sections live as siblings under ``settings.sandbox`` so
        the SDK does not overwrite any of them.
        """
        # -- behaviour flags --
        sandbox: dict[str, Any] = {
            "enabled": True,
            "autoAllowBashIfSandboxed": self.auto_allow_bash,
            "allowUnsandboxedCommands": self.allow_unsandboxed_commands,
        }
        if self.excluded_commands:
            sandbox["excludedCommands"] = self.excluded_commands
        if self.enable_weaker_nested_sandbox:
            sandbox["enableWeakerNestedSandbox"] = True

        # -- filesystem --
        # Always emit filesystem section; sandbox-runtime requires
        # allowWrite as an array (can be empty).  denyWrite is only
        # meaningful when non-empty (writes are deny-all by default).
        filesystem: dict[str, Any] = {
            "allowWrite": allow_write,
        }
        if deny_write:
            filesystem["denyWrite"] = deny_write
        if allow_read:
            filesystem["allowRead"] = allow_read
        if deny_read:
            filesystem["denyRead"] = deny_read
        sandbox["filesystem"] = filesystem

        # -- network --
        # Always emit network section; sandbox-runtime requires
        # allowedDomains and deniedDomains as arrays (can be empty).
        # Empty allowedDomains = block all outbound traffic.
        network: dict[str, Any] = {
            "allowedDomains": self.allowed_domains,
            "deniedDomains": self.denied_domains,
        }
        sandbox["network"] = network

        settings_obj["sandbox"] = sandbox

    def _collect_paths(
            self, workspace_dir: Path | None
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Collect effective allow/deny paths with workspace overrides.

        When any ``deny_read`` entry covers ``~/.claude/``, the harness
        directory is automatically added to ``allow_read`` so the OS
        sandbox layer (which uses most-specific-wins semantics) permits
        reads to Claude Code's internal files (tool-results, session-memory,
        plans, scratchpad).
        """
        ws = str(workspace_dir) if workspace_dir else None
        allow_read = ([ws] if ws else []) + list(self.allow_read)
        allow_write = ([ws] if ws else []) + list(self.allow_write)
        deny_read = list(self.deny_read)
        deny_write = list(self.deny_write)

        # Ensure ~/.claude/ is in allowRead when deny_read covers it.
        # The OS sandbox uses most-specific-wins, so this carve-out works.
        if self._any_deny_covers_harness(deny_read):
            if _CLAUDE_HARNESS_DIR not in allow_read:
                allow_read.append(_CLAUDE_HARNESS_DIR)

        return allow_read, allow_write, deny_read, deny_write

    @classmethod
    def _permissions_deny_read(cls, deny_read: list[str]) -> list[str]:
        """Return deny_read list adjusted for the permissions layer.

        Claude Code's permission system checks deny rules *before* its
        internal-path allow list (step 3 vs step 7), and deny is absolute
        — there is no "most-specific-wins" override.  A blanket
        ``Read(~/**)`` deny therefore blocks reads to
        ``~/.claude/projects/.../tool-results/``.

        This method replaces any deny_read entry that covers ``~/.claude/``
        with an explicit list of well-known sensitive directories,
        preserving the security intent without blocking the harness.
        """
        if not cls._any_deny_covers_harness(deny_read):
            return deny_read

        result: list[str] = []
        for entry in deny_read:
            norm = str(Path(cls._normalize_path(entry)))
            harness_str = str(Path(_CLAUDE_HARNESS_DIR).expanduser()).rstrip("/\\")
            if harness_str == norm or harness_str.startswith(norm + "/"):
                result.extend(_SENSITIVE_HOME_DIRS)
            else:
                result.append(entry)
        return result

    @staticmethod
    def _any_deny_covers_harness(deny_read: list[str]) -> bool:
        """Check whether any deny_read entry is a parent of ``~/.claude/``."""
        harness = Path(_CLAUDE_HARNESS_DIR).expanduser()
        harness_str = str(harness).rstrip("/\\")
        for entry in deny_read:
            norm = str(Path(SandboxConfigBuilder._normalize_path(entry)))
            if harness_str == norm or harness_str.startswith(norm + "/"):
                return True
        return False

    def _attach_permissions(
            self,
            settings_obj: dict[str, Any],
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> None:
        """Attach permission allow/deny rules for built-in tools.

        Mapping between sandbox filesystem rules and permission tool rules:

        - ``allow_read``  → ``Read(path/**)``  (allow)
        - ``allow_write`` → ``Read(path/**)``, ``Edit(path/**)``, ``Write(path/**)``  (allow)
        - ``deny_read``   → ``Read(path/**)``, ``Edit(path/**)``, ``Write(path/**)``  (deny)
          If a path is denied for reading, it is implicitly denied for all
          tool access — reading, editing, and writing.  This mirrors the
          principle that ``sandbox.filesystem.denyRead`` blocks the OS-level
          Bash sandbox entirely, so the permissions layer should be equally
          restrictive for built-in tools.
        - ``deny_write``  → ``Edit(path/**)``, ``Write(path/**)``  (deny)
        """
        read_tools = ("Read",)
        all_tools = ("Read", "Edit", "Write")
        write_only_tools = ("Edit", "Write")
        raw_allow = (
                self._permission_rules(allow_read, read_tools)
                + self._permission_rules(allow_write, all_tools)
        )
        permission_allow = list(dict.fromkeys(raw_allow))
        filtered_deny_read, filtered_deny_write = self._filter_conflicting_denies(
            allow_read=allow_read, allow_write=allow_write,
            deny_read=deny_read, deny_write=deny_write,
        )
        raw_deny = (
                self._permission_rules(filtered_deny_read, all_tools)
                + self._permission_rules(
                    filtered_deny_write,
                    write_only_tools,
                )
        )
        permission_deny = list(dict.fromkeys(raw_deny))
        if permission_allow:
            settings_obj.setdefault("permissions", {})["allow"] = permission_allow
        if permission_deny:
            settings_obj.setdefault("permissions", {})["deny"] = permission_deny

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a path: strip trailing separators and expand ``~``."""
        stripped = path.rstrip("/\\")
        if not stripped:
            stripped = path
        if stripped == "~" or stripped.startswith("~/") or stripped.startswith("~\\"):
            stripped = str(Path(stripped.replace("\\", "/")).expanduser())
        return stripped

    @staticmethod
    def _to_permission_path_specifier(path: str) -> str:
        """Convert a filesystem path into Claude Code's permission path syntax."""
        specifier = SandboxConfigBuilder._normalize_path(path)

        if specifier == "~" or specifier.startswith("~/"):
            return specifier

        if re.match(r"^[A-Za-z]:([\\/]|$)", specifier):
            drive = specifier[0].lower()
            remainder = specifier[2:].replace("\\", "/").lstrip("/")
            return f"//{drive}" + (f"/{remainder}" if remainder else "")

        if specifier.startswith("/"):
            return f"/{specifier}"

        return specifier.replace("\\", "/")

    @staticmethod
    def _filter_conflicting_denies(
            *,
            allow_read: list[str],
            allow_write: list[str],
            deny_read: list[str],
            deny_write: list[str],
    ) -> tuple[list[str], list[str]]:
        """Remove deny paths that exactly match an allow path.

        Only exact matches are removed.  A child allow (e.g. ``/data/public``)
        does NOT remove a parent deny (e.g. ``/data``), because
        sandbox-runtime uses "most-specific wins" semantics — the parent
        deny still applies to paths outside the allowed child.
        """
        allow_resolved = {
            SandboxConfigBuilder._normalize_path(p)
            for p in allow_read + allow_write
        }

        def is_exact_allow(deny_path: str) -> bool:
            deny_norm = SandboxConfigBuilder._normalize_path(deny_path)
            return deny_norm in allow_resolved

        filtered_deny_read = [p for p in deny_read if not is_exact_allow(p)]
        filtered_deny_write = [p for p in deny_write if not is_exact_allow(p)]
        return filtered_deny_read, filtered_deny_write

    @staticmethod
    def _permission_rules(
            paths: list[str], tools: tuple[str, ...]
    ) -> list[str]:
        """Generate permission rules for *paths* × *tools*.
        Each path is converted into Claude Code's permission syntax and
        combined with every tool name to produce rules like ``Edit(//tmp/work/**)``.
        """
        rules: list[str] = []
        for path in paths:
            specifier = SandboxConfigBuilder._to_permission_path_specifier(path)
            for tool in tools:
                rule = f"{tool}({specifier}/**)"
                if rule not in rules:
                    rules.append(rule)
        return rules
