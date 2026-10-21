"""SDK bridge — the only module that directly interacts with the claude-agent-sdk.

Extracted from the old ``executor.py``.  Provides a standalone async generator
that calls ``query()``, updates ``RunState``, and yields the SDK's native
message objects unchanged.

Session resume / fork is handled transparently via ``ClaudeAgentOptions.resume``
and ``ClaudeAgentOptions.fork_session`` — no special logic needed here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from cckit._engine.state import RunState
from cckit.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

_RESULT_ERROR_SUBTYPES: dict[str, str] = {
    "error_max_turns": "max_turns",
    "error_max_budget_usd": "max_budget",
    "error_during_execution": "agent_error",
}

_TERMINAL_REASONS: dict[str, str] = {
    "max_turns": "max_turns",
    "max_budget": "max_budget",
    "api_error": "api_error",
}


def _classify_sdk_error(exc: BaseException) -> str:
    """Return a stable code for an SDK failure without parsing its message.

    The SDK's ``ResultError`` (claude-agent-sdk >= 0.2.151) exposes
    ``subtype`` / ``terminal_reason`` / ``api_error_status``; older SDK
    versions only expose the plain ``ProcessError`` message, so every
    attribute is read defensively.
    """
    subtype = getattr(exc, "subtype", None)
    if isinstance(subtype, str):
        code = _RESULT_ERROR_SUBTYPES.get(subtype)
        if code:
            return code

    reason = getattr(exc, "terminal_reason", None)
    if isinstance(reason, str):
        code = _TERMINAL_REASONS.get(reason)
        if code:
            return code
        if reason:
            return reason

    if getattr(exc, "api_error_status", None) is not None:
        return "api_error"

    if isinstance(subtype, str) and subtype.startswith("error_"):
        return subtype.removeprefix("error_")

    return "agent_execution_failed"


def _sdk_error_details(exc: BaseException, state: RunState) -> dict[str, Any]:
    """Collect structured diagnostics from the SDK exception and run state.

    Kept separate from ``detail`` on purpose: ``detail`` is the legacy
    free-form string appended to ``str(exc)``; these fields are for callers
    that want to branch or persist structured data.
    """
    details: dict[str, Any] = {}

    for attr in ("subtype", "terminal_reason", "api_error_status", "session_id"):
        value = getattr(exc, attr, None)
        if value is not None:
            details[attr] = value

    exit_code = getattr(exc, "exit_code", None)
    if exit_code is not None:
        details["exit_code"] = exit_code

    errors = getattr(exc, "errors", None)
    if isinstance(errors, str):
        details["errors"] = [errors[:500]]
    elif errors:
        details["errors"] = [str(item)[:500] for item in errors]

    final_message = state.final_message
    if final_message is not None:
        for attr in (
            "num_turns",
            "stop_reason",
            "subtype",
            "terminal_reason",
            "api_error_status",
            "session_id",
        ):
            value = getattr(final_message, attr, None)
            if value is not None:
                details.setdefault(attr, value)

    if state.session_id:
        details.setdefault("session_id", state.session_id)

    return details


def _patch_log_cli_command() -> None:
    """Monkey-patch SubprocessCLITransport to log the real CLI command once.

    The patch wraps ``start()`` so that the ``_build_command()`` result is
    logged at DEBUG level right before the subprocess is launched.  It is
    idempotent — the original ``start`` is stored on the class the first time
    and reused on subsequent calls.
    """
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (  # noqa: WPS433
            SubprocessCLITransport,
        )
    except ImportError:
        return

    if getattr(SubprocessCLITransport, "_cckit_start_patched", False):
        return  # already patched

    original_connect = SubprocessCLITransport.connect  # type: ignore[attr-defined]

    async def _patched_connect(self: Any) -> None:  # type: ignore[no-untyped-def]
        # Call original connect() first — it resolves _cli_path internally.
        # _build_command() requires _cli_path to be set, so we log *after*.
        await original_connect(self)
        try:
            cmd = self._build_command()
            # Mask sensitive flags: --api-key / anything that looks like a token
            _SENSITIVE = {"--api-key", "--auth-token"}
            safe_cmd: list[str] = []
            skip_next = False
            for part in cmd:
                if skip_next:
                    safe_cmd.append("***")
                    skip_next = False
                elif part in _SENSITIVE:
                    safe_cmd.append(part)
                    skip_next = True
                else:
                    safe_cmd.append(part)
            logger.info("[CLI command] %s", " ".join(safe_cmd))
        except Exception:
            logger.debug("Failed to log CLI command", exc_info=True)

    SubprocessCLITransport.connect = _patched_connect  # type: ignore[method-assign]
    SubprocessCLITransport._cckit_start_patched = True  # type: ignore[attr-defined]


async def run_sdk_query(
    prompt: str,
    options: Any,
    state: RunState,
) -> AsyncIterator[Any]:
    """Call ``query(prompt, options=options)`` and yield SDK messages.

    Uses the top-level ``query()`` async generator from ``claude_agent_sdk``
    instead of the stateful ``query`` context-manager approach.
    Session resume is handled transparently by ``ClaudeAgentOptions.resume``.
    """
    try:
        from claude_agent_sdk import query  # noqa: WPS433
    except ImportError as exc:
        raise AgentExecutionError(
            "claude-agent-sdk is not installed",
            detail=str(exc),
            code="sdk_not_installed",
        ) from exc

    _patch_log_cli_command()

    try:
        async for message in query(prompt=prompt, options=options):
            state.observe(message)
            yield message
    except Exception as exc:
        stderr_text = state.stderr_text
        detail = str(exc)
        if stderr_text:
            detail = f"{detail}\nClaude stderr:\n{stderr_text}"
        raise AgentExecutionError(
            f"SDK query failed: {exc}",
            detail=detail,
            code=_classify_sdk_error(exc),
            details=_sdk_error_details(exc, state),
        ) from exc
