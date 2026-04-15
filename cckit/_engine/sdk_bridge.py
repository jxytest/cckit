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

    original_start = SubprocessCLITransport.start  # type: ignore[attr-defined]

    async def _patched_start(self: Any) -> None:  # type: ignore[no-untyped-def]
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
        logger.debug("[CLI command] %s", " ".join(safe_cmd))
        await original_start(self)

    SubprocessCLITransport.start = _patched_start  # type: ignore[method-assign]
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
        ) from exc
