"""Claude CLI detection and validation.

Checks that the ``claude`` CLI is installed and accessible.
Gives a friendly error message if not.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from clab.exceptions import ClabError

logger = logging.getLogger(__name__)
_checked = False


def check_claude_cli() -> None:
    """Verify the Claude CLI is installed.  Called once on first Runner init."""
    global _checked  # noqa: PLW0603
    if _checked:
        return

    if not shutil.which("claude"):
        raise ClabError(
            "Claude CLI not found on PATH.\n\n"
            "clab requires the Claude CLI to be installed.\n"
            "Install it with: npm install -g @anthropic-ai/claude-code\n"
            "Then run: claude login"
        )

    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.info("Claude CLI version: %s", result.stdout.strip())
    except Exception as exc:
        logger.warning("Could not determine Claude CLI version: %s", exc)

    _checked = True
