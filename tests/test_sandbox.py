"""Tests for sandbox configuration — validates cckit's sandbox config builder
against sandbox-runtime semantics.

Tests cover:
1. SandboxConfigBuilder basic construction and build() return types
2. Filesystem rules: allow-only writes, deny-then-allow reads
3. Network rules: allow-only domains with denied_domains precedence
4. Permission rule generation (Read/Edit/Write tools)
5. Conflict resolution between allow and deny paths
6. Workspace directory auto-injection
7. Path normalisation (trailing slash, ~ expansion)
8. SandboxOptions Pydantic model defaults
9. Integration with RunnerConfig.from_env()
10. Unified sandbox section (behaviour flags + filesystem + network merged)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cckit.sandbox.config import SandboxConfigBuilder
from cckit.types import RunnerConfig, SandboxOptions


def _parse(settings_json: str) -> dict:
    """Parse settings JSON and return the parsed dict."""
    assert settings_json is not None
    return json.loads(settings_json)


# =====================================================================
# 1. Basic construction & disabled state
# =====================================================================


class TestSandboxConfigDisabled:
    """When sandbox is disabled, build() should return None."""

    def test_disabled_returns_none(self):
        builder = SandboxConfigBuilder(enabled=False)
        assert builder.build() is None

    def test_disabled_with_workspace_still_returns_none(self):
        builder = SandboxConfigBuilder(enabled=False)
        with tempfile.TemporaryDirectory() as ws:
            assert builder.build(workspace_dir=Path(ws)) is None

    def test_disabled_ignores_all_config(self):
        """All settings are ignored when disabled — build() never processes them."""
        builder = SandboxConfigBuilder(
            enabled=False,
            allow_write=["/tmp/data"],
            deny_read=["/secret"],
            allowed_domains=["github.com"],
        )
        assert builder.build() is None


# =====================================================================
# 2. Enabled state — return type validation
# =====================================================================


class TestSandboxConfigEnabled:
    """When enabled, build() returns a JSON string."""

    def test_returns_json_string(self):
        builder = SandboxConfigBuilder(enabled=True)
        result = builder.build()
        assert isinstance(result, str)

    def test_json_has_sandbox_section(self):
        builder = SandboxConfigBuilder(enabled=True)
        parsed = _parse(builder.build())
        assert "sandbox" in parsed

    def test_sandbox_section_has_behaviour_flags(self):
        """Behaviour flags (SandboxSettings) are inside sandbox section."""
        builder = SandboxConfigBuilder(enabled=True)
        parsed = _parse(builder.build())
        sandbox = parsed["sandbox"]
        assert sandbox["enabled"] is True
        assert "autoAllowBashIfSandboxed" in sandbox
        assert "allowUnsandboxedCommands" in sandbox

    def test_sandbox_section_has_filesystem(self):
        """filesystem rules are inside sandbox section (not overwritten)."""
        builder = SandboxConfigBuilder(enabled=True)
        parsed = _parse(builder.build())
        assert "filesystem" in parsed["sandbox"]

    def test_sandbox_section_has_network(self):
        """network rules are inside sandbox section (not overwritten)."""
        builder = SandboxConfigBuilder(enabled=True)
        parsed = _parse(builder.build())
        assert "network" in parsed["sandbox"]

    def test_sandbox_settings_defaults(self):
        builder = SandboxConfigBuilder(enabled=True)
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["autoAllowBashIfSandboxed"] is True
        assert sandbox["allowUnsandboxedCommands"] is False
        assert "excludedCommands" not in sandbox
        assert "enableWeakerNestedSandbox" not in sandbox

    def test_sandbox_settings_custom_values(self):
        builder = SandboxConfigBuilder(
            enabled=True,
            auto_allow_bash=False,
            allow_unsandboxed_commands=False,
            excluded_commands=["git", "docker"],
            enable_weaker_nested_sandbox=True,
        )
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["autoAllowBashIfSandboxed"] is False
        assert sandbox["allowUnsandboxedCommands"] is False
        assert sandbox["excludedCommands"] == ["git", "docker"]
        assert sandbox["enableWeakerNestedSandbox"] is True


# =====================================================================
# 3. Unified sandbox section (the critical fix)
# =====================================================================


class TestUnifiedSandboxSection:
    """Behaviour flags, filesystem, and network must all be in ONE sandbox dict.

    This verifies the fix for the SDK overwrite bug: if we set
    ClaudeAgentOptions.sandbox, the SDK replaces settings_obj["sandbox"]
    entirely, losing filesystem/network rules. Instead we merge everything
    into settings JSON and leave ClaudeAgentOptions.sandbox=None.
    """

    def test_behaviour_and_filesystem_coexist(self):
        """enabled/autoAllowBashIfSandboxed and filesystem are siblings."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=["/secret"])
        sandbox = _parse(builder.build())["sandbox"]
        # behaviour flags
        assert sandbox["enabled"] is True
        # filesystem rules
        assert "/secret" in sandbox["filesystem"]["denyRead"]

    def test_behaviour_and_network_coexist(self):
        """enabled/autoAllowBashIfSandboxed and network are siblings."""
        builder = SandboxConfigBuilder(
            enabled=True, allowed_domains=["github.com"], deny_read=[],
        )
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["enabled"] is True
        assert "github.com" in sandbox["network"]["allowedDomains"]

    def test_all_three_coexist(self):
        """behaviour + filesystem + network in a single sandbox dict."""
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["/secret"],
            allowed_domains=["github.com"],
            excluded_commands=["docker"],
        )
        with tempfile.TemporaryDirectory() as ws:
            sandbox = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]
            # behaviour
            assert sandbox["enabled"] is True
            assert sandbox["excludedCommands"] == ["docker"]
            # filesystem
            assert str(Path(ws)) in sandbox["filesystem"]["allowWrite"]
            # network
            assert "github.com" in sandbox["network"]["allowedDomains"]


# =====================================================================
# 4. Filesystem: WRITE rules (allow-only semantics)
# =====================================================================


class TestFilesystemWriteRules:
    """Write uses allow-only semantics: default deny all, allowWrite permits."""

    def test_empty_allow_write_means_deny_all(self):
        """Empty allowWrite = no paths are writable (sandbox-runtime semantics)."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert fs["allowWrite"] == []

    def test_workspace_added_to_allow_write(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        with tempfile.TemporaryDirectory() as ws:
            fs = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]["filesystem"]
            assert str(Path(ws)) in fs["allowWrite"]

    def test_extra_allow_write_paths(self):
        builder = SandboxConfigBuilder(
            enabled=True, allow_write=["/opt/data", "/var/log"], deny_read=[],
        )
        with tempfile.TemporaryDirectory() as ws:
            fs = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]["filesystem"]
            assert str(Path(ws)) in fs["allowWrite"]
            assert "/opt/data" in fs["allowWrite"]
            assert "/var/log" in fs["allowWrite"]

    def test_deny_write_blocks_within_allowed(self):
        """denyWrite blocks specific paths within allowed regions."""
        builder = SandboxConfigBuilder(
            enabled=True,
            allow_write=["/opt"],
            deny_write=["/opt/secrets"],
            deny_read=[],
        )
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert "/opt" in fs["allowWrite"]
        assert "/opt/secrets" in fs["denyWrite"]

    def test_deny_write_defaults_to_empty(self):
        """deny_write should default to empty (writes are already deny-all)."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        assert builder.deny_write == []
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert "denyWrite" not in fs


# =====================================================================
# 5. Filesystem: READ rules (deny-then-allow semantics)
# =====================================================================


class TestFilesystemReadRules:
    """Read uses deny-then-allow: default allow all, denyRead blocks."""

    def test_deny_read_defaults_to_home(self):
        builder = SandboxConfigBuilder(enabled=True)
        assert builder.deny_read == ["~/"]

    def test_deny_read_appears_in_settings(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=["/secret"])
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert "/secret" in fs["denyRead"]

    def test_allow_read_re_allows_within_deny(self):
        """allowRead takes precedence over denyRead (most-specific wins)."""
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["/data"],
            allow_read=["/data/public"],
        )
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert "/data" in fs["denyRead"]
        assert "/data/public" in fs["allowRead"]

    def test_workspace_added_to_allow_read(self):
        builder = SandboxConfigBuilder(enabled=True)
        with tempfile.TemporaryDirectory() as ws:
            fs = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]["filesystem"]
            assert str(Path(ws)) in fs["allowRead"]

    def test_no_deny_read_when_empty(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        fs = _parse(builder.build())["sandbox"]["filesystem"]
        assert "denyRead" not in fs


# =====================================================================
# 6. Network rules (allow-only semantics)
# =====================================================================


class TestNetworkRules:
    """Network uses allow-only: empty allowedDomains = block all."""

    def test_empty_allowed_domains_blocks_all(self):
        """Empty allowedDomains means block all outbound traffic."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        net = _parse(builder.build())["sandbox"]["network"]
        assert net["allowedDomains"] == []

    def test_allowed_domains_permit_access(self):
        builder = SandboxConfigBuilder(
            enabled=True,
            allowed_domains=["github.com", "*.npmjs.org"],
            deny_read=[],
        )
        net = _parse(builder.build())["sandbox"]["network"]
        assert "github.com" in net["allowedDomains"]
        assert "*.npmjs.org" in net["allowedDomains"]

    def test_denied_domains_present(self):
        builder = SandboxConfigBuilder(
            enabled=True,
            allowed_domains=["github.com"],
            denied_domains=["evil.github.com"],
            deny_read=[],
        )
        net = _parse(builder.build())["sandbox"]["network"]
        assert "evil.github.com" in net["deniedDomains"]

    def test_network_always_emitted(self):
        """Network section is always present even when empty (allow-only semantics)."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        sandbox = _parse(builder.build())["sandbox"]
        assert "network" in sandbox


# =====================================================================
# 7. Permission rule generation
# =====================================================================


class TestPermissionRules:
    """Permission rules control Claude Code built-in tools (Read/Edit/Write)."""

    def test_allow_write_generates_read_edit_write_rules(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        with tempfile.TemporaryDirectory() as ws:
            parsed = _parse(builder.build(workspace_dir=Path(ws)))
            perms = parsed["permissions"]
            ws_str = str(Path(ws))
            ws_rule = SandboxConfigBuilder._to_permission_path_specifier(ws_str)
            assert f"Read({ws_rule}/**)" in perms["allow"]
            assert f"Edit({ws_rule}/**)" in perms["allow"]
            assert f"Write({ws_rule}/**)" in perms["allow"]

    def test_allow_read_generates_read_only_rules(self):
        builder = SandboxConfigBuilder(
            enabled=True,
            allow_read=["/data/public"],
            deny_read=["/data"],
        )
        parsed = _parse(builder.build())
        allow_rules = parsed["permissions"]["allow"]
        assert "Read(//data/public/**)" in allow_rules
        assert "Edit(//data/public/**)" not in allow_rules
        assert "Write(//data/public/**)" not in allow_rules

    def test_deny_read_generates_all_tool_deny(self):
        """deny_read should block all tools (Read/Edit/Write).

        If a path is denied for reading, it should also be denied for
        editing and writing — the principle of least privilege.  The OS-level
        sandbox.filesystem.denyRead blocks Bash entirely; the permissions
        layer should be equally restrictive for built-in tools.
        """
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["/secret"],
            allow_write=["/secret/writable"],
        )
        parsed = _parse(builder.build())
        deny_rules = parsed["permissions"]["deny"]
        assert "Read(//secret/**)" in deny_rules
        assert "Edit(//secret/**)" in deny_rules
        assert "Write(//secret/**)" in deny_rules

    def test_deny_write_generates_edit_write_deny_only(self):
        """deny_write should only block Edit and Write tools."""
        builder = SandboxConfigBuilder(
            enabled=True,
            allow_write=["/opt"],
            deny_write=["/opt/secrets"],
            deny_read=[],
        )
        parsed = _parse(builder.build())
        deny_rules = parsed["permissions"]["deny"]
        assert "Edit(//opt/secrets/**)" in deny_rules
        assert "Write(//opt/secrets/**)" in deny_rules
        assert "Read(//opt/secrets/**)" not in deny_rules

    def test_no_duplicate_permission_rules(self):
        """Duplicate rules (e.g., workspace in both allow_read and allow_write) are deduped."""
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        with tempfile.TemporaryDirectory() as ws:
            parsed = _parse(builder.build(workspace_dir=Path(ws)))
            allow_rules = parsed["permissions"]["allow"]
            ws_str = str(Path(ws))
            ws_rule = SandboxConfigBuilder._to_permission_path_specifier(ws_str)
            assert allow_rules.count(f"Read({ws_rule}/**)") == 1


# =====================================================================
# 8. Conflict resolution
# =====================================================================


class TestConflictResolution:
    """Deny entries that exactly match an allow entry are removed."""

    def test_exact_conflict_removed(self):
        """If deny_read path == allow_read path, deny is removed."""
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["/data"],
            allow_read=["/data"],
        )
        parsed = _parse(builder.build())
        assert "deny" not in parsed.get("permissions", {}) or \
            "Read(//data/**)" not in parsed["permissions"].get("deny", [])

    def test_parent_deny_not_removed_when_child_allowed(self):
        """deny_read on parent should NOT be removed if only child is in allow_read.

        sandbox-runtime uses "most-specific wins" — the parent deny still applies
        to paths outside the allowed child.
        """
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["/data"],
            allow_read=["/data/public"],
        )
        parsed = _parse(builder.build())
        deny_rules = parsed["permissions"].get("deny", [])
        assert "Read(//data/**)" in deny_rules

    def test_write_conflict_uses_write_allows(self):
        """deny_write conflict checked against allow_write only."""
        builder = SandboxConfigBuilder(
            enabled=True,
            allow_write=["/opt"],
            deny_write=["/opt"],
            deny_read=[],
        )
        parsed = _parse(builder.build())
        deny_rules = parsed["permissions"].get("deny", [])
        assert "Edit(//opt/**)" not in deny_rules


# =====================================================================
# 9. Path normalisation
# =====================================================================


class TestPathNormalisation:
    """Paths are normalised: trailing slashes stripped, ~ expanded."""

    def test_trailing_slash_stripped(self):
        result = SandboxConfigBuilder._normalize_path("/tmp/data/")
        assert result == "/tmp/data"

    def test_tilde_expanded(self):
        result = SandboxConfigBuilder._normalize_path("~/")
        assert result == str(Path.home())

    def test_tilde_subpath_expanded(self):
        result = SandboxConfigBuilder._normalize_path("~/.ssh")
        expected = str(Path.home() / ".ssh")
        assert result == expected

    def test_absolute_path_unchanged(self):
        result = SandboxConfigBuilder._normalize_path("/usr/local/bin")
        assert result == "/usr/local/bin"

    def test_permission_rules_strip_trailing_slash(self):
        rules = SandboxConfigBuilder._permission_rules(["/tmp/data/"], ("Read",))
        assert rules == ["Read(//tmp/data/**)"]

    def test_permission_rules_convert_windows_absolute_paths(self):
        rules = SandboxConfigBuilder._permission_rules(["E:\\tmp\\data"], ("Write",))
        assert rules == ["Write(//e/tmp/data/**)"]


# =====================================================================
# 10. Workspace auto-injection
# =====================================================================


class TestWorkspaceInjection:
    """Workspace directory is auto-injected into allow paths."""

    def test_workspace_in_both_allow_read_and_allow_write(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        with tempfile.TemporaryDirectory() as ws:
            fs = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]["filesystem"]
            ws_str = str(Path(ws))
            assert ws_str in fs["allowRead"]
            assert ws_str in fs["allowWrite"]

    def test_no_workspace_when_none(self):
        builder = SandboxConfigBuilder(enabled=True, deny_read=[])
        fs = _parse(builder.build(workspace_dir=None))["sandbox"]["filesystem"]
        assert fs["allowWrite"] == []

    def test_workspace_prepended_to_extra_paths(self):
        builder = SandboxConfigBuilder(
            enabled=True, allow_write=["/extra"], deny_read=[],
        )
        with tempfile.TemporaryDirectory() as ws:
            fs = _parse(builder.build(workspace_dir=Path(ws)))["sandbox"]["filesystem"]
            ws_str = str(Path(ws))
            assert fs["allowWrite"][0] == ws_str
            assert "/extra" in fs["allowWrite"]


# =====================================================================
# 11. SandboxOptions Pydantic model defaults
# =====================================================================


class TestSandboxOptionsDefaults:
    """SandboxOptions should have correct defaults matching sandbox-runtime semantics."""

    def test_disabled_by_default(self):
        opts = SandboxOptions()
        assert opts.enabled is False

    def test_deny_write_defaults_empty(self):
        """deny_write defaults to empty (writes are deny-all by default)."""
        opts = SandboxOptions()
        assert opts.deny_write == []

    def test_deny_read_defaults_to_home(self):
        opts = SandboxOptions()
        assert opts.deny_read == ["~/"]

    def test_allowed_domains_defaults_empty(self):
        """Empty allowed_domains = block all (not allow all)."""
        opts = SandboxOptions()
        assert opts.allowed_domains == []

    def test_allow_unsandboxed_commands_defaults_false(self):
        """Must default to False to prevent dangerouslyDisableSandbox escape."""
        opts = SandboxOptions()
        assert opts.allow_unsandboxed_commands is False

    def test_denied_domains_defaults_empty(self):
        opts = SandboxOptions()
        assert opts.denied_domains == []

    def test_from_env_includes_workspace_root(self):
        """RunnerConfig.from_env() should expose workspace_root."""
        cfg = RunnerConfig.from_env()
        assert isinstance(cfg.workspace_root, Path)


# =====================================================================
# 12. Full integration: workspace-only sandbox
# =====================================================================


class TestWorkspaceOnlySandbox:
    """Simulate the most common use case: only workspace directory is R/W."""

    def test_workspace_only_rw(self):
        """Agent can only read/write inside workspace, everything else blocked."""
        builder = SandboxConfigBuilder(
            enabled=True,
            deny_read=["~/"],
        )
        with tempfile.TemporaryDirectory() as ws:
            settings_json = builder.build(workspace_dir=Path(ws))
            assert settings_json is not None

            parsed = json.loads(settings_json)
            sandbox = parsed["sandbox"]
            fs = sandbox["filesystem"]
            ws_str = str(Path(ws))

            # Behaviour flags present
            assert sandbox["enabled"] is True

            # Only workspace is writable
            assert fs["allowWrite"] == [ws_str]
            # Workspace is re-allowed for reading within denied home
            assert ws_str in fs["allowRead"]
            # Home is blocked for reading
            assert "~/" in fs["denyRead"]

            # Network: empty = block all
            net = sandbox["network"]
            assert net["allowedDomains"] == []

            # Permissions: workspace has full R/W tool access
            perms = parsed["permissions"]
            ws_rule = SandboxConfigBuilder._to_permission_path_specifier(ws_str)
            assert f"Read({ws_rule}/**)" in perms["allow"]
            assert f"Edit({ws_rule}/**)" in perms["allow"]
            assert f"Write({ws_rule}/**)" in perms["allow"]


# =====================================================================
# 13. dangerouslyDisableSandbox escape prevention
# =====================================================================


class TestDangerouslyDisableSandboxPrevention:
    """Verify that the default config blocks dangerouslyDisableSandbox escape.

    The Bash tool accepts a ``dangerouslyDisableSandbox`` flag that runs the
    command OUTSIDE the sandbox. If ``allowUnsandboxedCommands`` is True, the
    CLI honours this flag — meaning the Agent can trivially escape all
    filesystem and network restrictions.
    """

    def test_default_blocks_escape(self):
        """Default SandboxConfigBuilder must set allowUnsandboxedCommands=False."""
        builder = SandboxConfigBuilder(enabled=True)
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["allowUnsandboxedCommands"] is False

    def test_explicit_true_allows_escape(self):
        """User can explicitly opt-in to allow unsandboxed commands."""
        builder = SandboxConfigBuilder(
            enabled=True, allow_unsandboxed_commands=True,
        )
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["allowUnsandboxedCommands"] is True

    def test_sandbox_options_default_blocks_escape(self):
        """SandboxOptions Pydantic model must default to False."""
        opts = SandboxOptions(enabled=True)
        assert opts.allow_unsandboxed_commands is False

    def test_sandbox_options_propagate_default(self):
        """SandboxOptions → SandboxConfigBuilder should propagate the False default."""
        opts = SandboxOptions(enabled=True)
        builder = SandboxConfigBuilder(
            enabled=opts.enabled,
            allow_unsandboxed_commands=opts.allow_unsandboxed_commands,
        )
        sandbox = _parse(builder.build())["sandbox"]
        assert sandbox["allowUnsandboxedCommands"] is False
