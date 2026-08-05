"""Agent subprocess environment construction and host-env isolation.

The Claude CLI subprocess inherits the host ``os.environ`` before the SDK
merges ``options.env`` on top.  This module builds the explicit env dict and
blanks out every host variable that is not on the passthrough allowlist, so
the agent (and every Bash child it spawns) cannot read host secrets.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cckit._engine.model_bridge import PreparedModelEndpoint
    from cckit.agent import Agent
    from cckit.types import ModelConfig, RunContext

# Variables that must be inherited as-is from the host process so that basic
# shell tooling (ls, git, python, node …) works inside the Claude CLI
# subprocess.
AGENT_ENV_PASSTHROUGH: frozenset[str] = frozenset(
    {
        # --- shell / filesystem ---
        "PATH",  # binary lookup — without this ls/git/python are gone
        "HOME",  # many tools write config to $HOME
        "USER",  # username (git author, some CLIs)
        "LOGNAME",  # POSIX alias of USER
        "SHELL",  # default shell for subprocess spawning
        "TERM",  # terminal type (colour output, readline)
        "LANG",  # locale — affects sort order, file encoding
        "LC_ALL",  # overrides all LC_* at once
        "LC_CTYPE",  # character classification / encoding
        "TMPDIR",  # POSIX temp dir ($TMPDIR on macOS)
        "TMP",  # Windows / some Linux tools
        "TEMP",  # Windows alias
        "PWD",  # current working directory
        "OLDPWD",  # previous directory (cd -)
        "SHLVL",  # shell nesting counter
        # --- Python runtime (Claude CLI runs on Node but subprocesses may use py) ---
        "PYTHONPATH",  # extra Python module search paths
        "PYTHONUNBUFFERED",  # flush stdout/stderr immediately
        "VIRTUAL_ENV",  # active venv (affects pip, python binary)
        # --- Node.js / Claude CLI runtime ---
        "NODE_PATH",  # Node module search path
        "NODE_OPTIONS",  # Node JVM flags
        "NVM_DIR",  # nvm installation root
        "NVM_BIN",  # nvm active bin dir
        # --- git ---
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_SSH_COMMAND",  # custom SSH wrapper for git
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        # --- Claude Code internals ---
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_TMPDIR",
        # --- Windows system variables ---
        # The bundled Claude Code CLI (>= 2.x, shipped with
        # claude-agent-sdk 0.2.x) crashes with STATUS_STACK_BUFFER_OVERRUN
        # (0xC0000409) when these are blanked to "". The older 1.x CLI
        # tolerated empty values; 2.x does not. These are non-secret
        # system identifiers (system root, processor, standard dirs) and
        # must be inherited as-is on Windows so the CLI subprocess can
        # initialise. They are absent on macOS/Linux, so listing them is
        # harmless there.
        "SYSTEMROOT",  # %SystemRoot% — required by the Windows C runtime
        "WINDIR",  # %WinDir% — Windows directory
        "COMSPEC",  # %ComSpec% — default shell (cmd.exe) path
        "OS",  # "Windows_NT"
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "NUMBER_OF_PROCESSORS",
        # --- Windows user / app directories ---
        "USERPROFILE",  # Windows home directory (equivalent of $HOME)
        "APPDATA",  # roaming app data (CLI writes config here)
        "LOCALAPPDATA",  # local app data
        "HOMEDRIVE",  # drive of user home, e.g. "C:"
        "HOMEPATH",  # path of user home, e.g. "\\Users\\name"
        "PROGRAMDATA",  # all-users app data
        "ALLUSERSPROFILE",  # alias of PROGRAMDATA on some Windows
        "PROGRAMFILES",  # 64-bit Program Files
        "PROGRAMFILES(X86)",  # 32-bit Program Files
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "PUBLIC",  # %PUBLIC% shared user directory
        # --- plugin-declared business env (not statically discoverable) ---
        # Agent/Spec/ContextConfig declare no "required env names" field;
        # plugins inject these via ContextProvider at runtime (os.environ
        # or ctx.env). Listing them here keeps the isolation loop from
        # blanking them so the agent subprocess + subagents inherit them.
        # See EvaluatorContextProvider in output/ui_rubric_evaluator/context.py.
        # NOTE: append new plugin env var names here as plugins are added;
        # alternatively write them into RunContext.env, which is auto-passthrough.
        "PLAYWRIGHT_LIVE_SESSION",  # browser subagent live session id
        "WAVE_EVALUATION_RUBRIC_DIR",  # rubric pack abs path
        "WAVE_EVALUATION_OUTPUT_DIR",  # evaluation output dir
    }
)


def is_windows() -> bool:
    return sys.platform == "win32"


def is_root_user() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def build_agent_env(
        agent: Agent,
        ctx: RunContext,
        model: ModelConfig,
        prepared_model: PreparedModelEndpoint,
) -> dict[str, str]:
    """Build the env dict handed to the Claude CLI subprocess.

    Contains NO git credentials — those are injected only into git
    subprocesses via ``GitConfig.build_git_env()``.

    Layering, in order: caller-provided ``ctx.env`` → ``ContextConfig`` env
    vars → ``ModelConfig.max_tokens`` → resolved model endpoint (bridge mode
    relies on this to force the CLI through the local compatibility server) →
    host-env isolation.
    """
    env: dict[str, str] = dict(ctx.env)

    # -- ContextConfig → CLI env vars (auto-compact threshold, etc.) --
    context_cfg = agent.context
    if context_cfg is not None:
        env.update(context_cfg.to_env())

    # -- ModelConfig.max_tokens → CLAUDE_CODE_MAX_OUTPUT_TOKENS --
    # When the user explicitly sets max_tokens on ModelConfig, propagate it
    # to the Claude CLI subprocess so getMaxOutputTokensForModel() uses the
    # same limit.  This is critical for CLAUDE_CODE_AUTO_COMPACT_WINDOW to
    # work correctly: effectiveContextWindow = window - min(maxOutputTokens, 20000).
    # With max_tokens=None (default) we leave CLAUDE_CODE_MAX_OUTPUT_TOKENS
    # unset so the CLI falls back to its own model-specific defaults.
    if model.max_tokens is not None:
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(model.max_tokens))

    if prepared_model.api_key:
        # ANTHROPIC_API_KEY  → sent as X-Api-Key header (direct Anthropic API)
        # ANTHROPIC_AUTH_TOKEN → sent as Authorization: Bearer header (LLM gateway / proxy)
        # Both are injected so the CLI authenticates correctly regardless of
        # whether the endpoint is a first-party Anthropic host or a third-party proxy.
        env["ANTHROPIC_API_KEY"] = prepared_model.api_key
        env["ANTHROPIC_AUTH_TOKEN"] = prepared_model.api_key
    if prepared_model.base_url:
        env["ANTHROPIC_BASE_URL"] = prepared_model.base_url
        # Third-party proxies often reject Anthropic-specific beta headers
        # and non-essential traffic (telemetry, autoupdater, etc.)
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "0")
        env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "0")
        env.setdefault("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
        # Disable extended thinking — many proxies don't support it
        env.setdefault("MAX_THINKING_TOKENS", "0")
        # 打开agent team功能
        env.setdefault("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "1")

    _isolate_host_env(env, ctx)
    return env


def _isolate_host_env(env: dict[str, str], ctx: RunContext) -> None:
    """Blank out host ``os.environ`` keys that are not explicitly allowed.

    The SDK's SubprocessCLITransport unconditionally inherits the host
    process's os.environ before merging options.env on top.  Since
    options.env is merged last, writing "" for every host variable that is
    NOT explicitly allowed effectively clears its value in the CLI
    subprocess (and transitively in every Bash child it spawns).

    Limitation: the variable name is still visible via ``env``; only the
    value is wiped.  This is the best we can do within the SDK's
    ``dict[str, str]`` interface (None / unset is not supported upstream).
    """
    # ctx.env keys are plugin-declared and explicitly forwarded — treat them
    # as passthrough too so the isolation loop never blanks them. (env already
    # contains ctx.env via dict(ctx.env) in the caller, so this is a semantic
    # guard: it makes the intent explicit and survives future reordering.)
    ctx_env_keys = frozenset(ctx.env.keys())
    for key in os.environ:
        if key not in env and key not in AGENT_ENV_PASSTHROUGH and key not in ctx_env_keys:
            env[key] = ""
