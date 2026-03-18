"""Platform MCP tools exposed to agents.

These tools give agents the ability to interact with the platform (e.g.
read failure info, write test cases back, report progress).  They are
registered via ``create_sdk_mcp_server`` and injected into agents that
declare them in ``AgentConfig.mcp_tool_names``.

Usage::

    from core.agent.mcp.platform_tools import get_platform_mcp_server
    server = get_platform_mcp_server()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _get_failure_info(args: dict[str, Any]) -> dict[str, Any]:
    """Retrieve failure details for a given test case from the platform."""
    project_id = args.get("project_id", "")
    case_id = args.get("case_id", "")
    logger.info("get_failure_info: project=%s case=%s", project_id, case_id)

    # TODO: wire up to real platform API
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Failure info for project={project_id}, case={case_id}: "
                    "placeholder — connect to platform API."
                ),
            }
        ]
    }


async def _write_test_cases(args: dict[str, Any]) -> dict[str, Any]:
    """Write generated test cases back to the platform."""
    project_id = args.get("project_id", "")
    cases = args.get("cases", "")
    logger.info("write_test_cases: project=%s (%d chars)", project_id, len(cases))

    # TODO: wire up to real platform API
    return {
        "content": [
            {
                "type": "text",
                "text": f"Wrote test cases to project {project_id} (placeholder).",
            }
        ]
    }


async def _report_progress(args: dict[str, Any]) -> dict[str, Any]:
    """Report agent progress back to the platform."""
    task_id = args.get("task_id", "")
    status = args.get("status", "")
    message = args.get("message", "")
    logger.info("report_progress: task=%s status=%s", task_id, status)

    # TODO: wire up to real platform API / WebSocket push
    return {
        "content": [
            {
                "type": "text",
                "text": f"Progress reported: task={task_id}, status={status}, message={message}",
            }
        ]
    }


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------

def get_platform_mcp_server() -> Any:
    """Create and return an SDK MCP server with all platform tools.

    Returns the server object expected by
    ``ClaudeAgentOptions(mcp_servers={"platform": server})``.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: WPS433

    @tool(
        "get_failure_info",
        "Get failure details for a test case from the platform",
        {"project_id": str, "case_id": str},
    )
    async def get_failure_info(args: dict[str, Any]) -> dict[str, Any]:
        return await _get_failure_info(args)

    @tool(
        "write_test_cases",
        "Write generated test cases back to the platform",
        {"project_id": str, "cases": str},
    )
    async def write_test_cases(args: dict[str, Any]) -> dict[str, Any]:
        return await _write_test_cases(args)

    @tool(
        "report_progress",
        "Report agent execution progress back to the platform",
        {"task_id": str, "status": str, "message": str},
    )
    async def report_progress(args: dict[str, Any]) -> dict[str, Any]:
        return await _report_progress(args)

    return create_sdk_mcp_server(
        "platform-tools",
        tools=[get_failure_info, write_test_cases, report_progress],
    )
