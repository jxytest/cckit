"""Built-in MCP server configs for cckit agents.

This module provides:

1. **:func:`sdk_mcp_server`** — helper for building in-process SDK MCP servers
   using ``@tool`` decorators.  Use this as a starting point for custom tools.

2. **:func:`playwright_mcp_server`** — convenience factory that returns a
   :class:`~claude_agent_sdk.types.McpStdioServerConfig` for
   `@playwright/mcp <https://github.com/microsoft/playwright-mcp>`_,
   giving agents full browser automation capabilities.

Usage::

    from cckit.tools.platform import sdk_mcp_server, playwright_mcp_server

    agent = Agent(
        name="web-tester",
        tools=["Bash"],
        mcp_servers={
            "my-tools":   sdk_mcp_server(),
            "playwright": playwright_mcp_server(),
        },
        instruction="Use playwright to test the UI.",
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK MCP server example
# ---------------------------------------------------------------------------


def sdk_mcp_server() -> Any:
    """Example: build an in-process SDK MCP server with ``@tool`` decorators.

    Copy and adapt this function to create your own MCP tools.  The returned
    value is a :class:`~claude_agent_sdk.types.McpSdkServerConfig` dict ready
    for ``Runner(mcp_servers={"my-tools": sdk_mcp_server()})``.

    Example::

        from cckit.tools.platform import sdk_mcp_server
        from cckit import Agent

        agent = Agent(name="my-agent", mcp_servers={"my-tools": sdk_mcp_server()})

    To define your own tools, adapt the pattern below::

        from claude_agent_sdk import create_sdk_mcp_server, tool
        from claude_agent_sdk.types import McpSdkServerConfig

        @tool("fetch_data", "Fetch data from the platform", {"id": str})
        async def fetch_data(args: dict) -> dict:
            result = await my_api.get(args["id"])
            return {"content": [{"type": "text", "text": result}]}

        server = create_sdk_mcp_server("my-tools", tools=[fetch_data])
        config = McpSdkServerConfig(type="sdk", name="my-tools", instance=server)
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: WPS433
    from claude_agent_sdk.types import McpSdkServerConfig  # noqa: WPS433

    @tool(
        "echo",
        "Echo the input message back (example tool — replace with your own)",
        {"message": str},
    )
    async def echo(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": args.get("message", "")}]}

    server = create_sdk_mcp_server("example-tools", tools=[echo])
    return McpSdkServerConfig(type="sdk", name="example-tools", instance=server)


# ---------------------------------------------------------------------------
# Pre-built third-party MCP server configs
# ---------------------------------------------------------------------------


def playwright_mcp_server(
    *,
    headless: bool = True,
    extra_args: list[str] | None = None,
) -> Any:
    """Return a :class:`~claude_agent_sdk.types.McpStdioServerConfig` for
    `@playwright/mcp <https://github.com/microsoft/playwright-mcp>`_.

    This spawns ``npx @playwright/mcp@latest`` as a subprocess MCP server,
    giving the agent full browser automation capabilities (navigation, clicks,
    form filling, screenshots, …).

    Requirements:
        * Node.js and ``npx`` must be available on ``PATH``.
        * ``@playwright/mcp`` will be fetched automatically by ``npx`` on first
          use, or pre-install it::

              npm install -g @playwright/mcp

    Parameters
    ----------
    headless:
        Run browsers in headless mode (default ``True``).  Pass
        ``headless=False`` for a visible browser window (useful for debugging).
    extra_args:
        Additional CLI arguments forwarded verbatim to ``@playwright/mcp``.

    Returns
    -------
    McpStdioServerConfig
        A TypedDict ready for ``Runner(mcp_servers={"playwright": ...})``.

    Example::

        from cckit.tools.platform import playwright_mcp_server
        from cckit import Runner, Agent

        agent = Agent(
            name="browser-agent",
            tools=["Bash"],
            mcp_servers={"playwright": playwright_mcp_server()},
            instruction="Use playwright MCP tools to automate browser tasks.",
        )
    """
    from claude_agent_sdk.types import McpStdioServerConfig  # noqa: WPS433

    args: list[str] = ["@playwright/mcp@latest"]
    if headless:
        args.append("--headless")
    if extra_args:
        args.extend(extra_args)

    return McpStdioServerConfig(type="stdio", command="npx", args=args)
