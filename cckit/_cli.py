"""Claude CLI detection and validation.

Checks that the ``claude`` CLI is installed and accessible.
Optionally verifies API key connectivity before execution.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request

from cckit.exceptions import CckitError, ConnectivityError

logger = logging.getLogger(__name__)
_checked = False

# Anthropic API endpoint for connectivity check
_DEFAULT_API_BASE = "https://api.anthropic.com"
_CONNECTIVITY_PATH = "/v1/messages"


def check_claude_cli() -> None:
    """Verify the Claude CLI is installed.  Called once on first Runner init."""
    global _checked  # noqa: PLW0603
    if _checked:
        return

    if not shutil.which("claude"):
        raise CckitError(
            "Claude CLI not found on PATH.\n\n"
            "cckit requires the Claude CLI to be installed.\n"
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


def check_api_connectivity(
    api_key: str = "",
    base_url: str = "",
    *,
    model: str = "",
    timeout: int = 10,
) -> None:
    """Verify API key and network connectivity to Anthropic API.

    Sends a minimal ``/v1/messages`` request with ``max_tokens=1``
    to validate the API key with negligible cost.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    base_url:
        API base URL.  Falls back to ``ANTHROPIC_BASE_URL`` env var or default.
    model:
        Model name for the check request.  Falls back to ``"claude-sonnet-4-20250514"``.
    timeout:
        HTTP request timeout in seconds.

    Raises
    ------
    ConnectivityError
        With a clear message indicating whether the problem is the API key,
        network, or an unexpected server error.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ConnectivityError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Set it via environment variable or pass api_key in ModelConfig."
        )

    base = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "") or _DEFAULT_API_BASE).rstrip("/")
    url = f"{base}{_CONNECTIVITY_PATH}"

    # Minimal messages request — max_tokens=1 to minimize cost
    check_model = model or "claude-sonnet-4-20250514"
    body = json.dumps({
        "model": check_model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "x-api-key": key,
            "Authorization": f"Bearer {key}",  # 兼容 OpenAI 格式的中转网关
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status == 200:
                logger.info("API connectivity check passed (key=...%s)", key[-4:])
                return
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            detail = json.loads(exc.read().decode()).get("error", {}).get("message", "")
        except Exception:
            detail = ""

        if status == 401:
            raise ConnectivityError(
                f"API key is invalid (HTTP 401).\n"
                f"Key ending in ...{key[-4:]}\n"
                f"Detail: {detail}" if detail else
                f"API key is invalid (HTTP 401). Key ending in ...{key[-4:]}"
            ) from exc
        if status == 403:
            raise ConnectivityError(
                f"API key lacks permission (HTTP 403).\n"
                f"Detail: {detail}" if detail else
                f"API key lacks permission (HTTP 403)."
            ) from exc
        # Other HTTP errors — still connected, just unexpected
        raise ConnectivityError(
            f"API returned HTTP {status}.\nDetail: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectivityError(
            f"Cannot reach API at {base}.\n"
            f"Check network connectivity, proxy settings, or firewall.\n"
            f"Detail: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ConnectivityError(
            f"API connection timed out after {timeout}s.\n"
            f"URL: {url}\n"
            f"Check network connectivity or increase timeout."
        ) from exc
