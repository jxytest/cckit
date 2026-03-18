"""Agent registry — register and look up agent types by name.

Usage::

    from core.agent.registry import agent_registry

    @agent_registry.register
    class MyAgent(BaseAgent):
        ...

    agent = agent_registry.get("my_agent_type")
"""

from __future__ import annotations

import logging
from typing import TypeVar

from core.agent.base import BaseAgent
from core.exceptions import AgentNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[BaseAgent])


class AgentRegistry:
    """In-memory registry mapping ``agent_type`` → ``BaseAgent`` subclass."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgent]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cls: T) -> T:
        """Class decorator that registers an agent by its ``config().agent_type``.

        Example::

            @agent_registry.register
            class FixAgent(BaseAgent):
                ...
        """
        instance = cls()
        agent_type = instance.agent_type
        if agent_type in self._agents:
            logger.warning(
                "Overwriting agent type %r (was %s, now %s)",
                agent_type,
                self._agents[agent_type].__name__,
                cls.__name__,
            )
        self._agents[agent_type] = cls
        logger.info("Registered agent: %s → %s", agent_type, cls.__name__)
        return cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, agent_type: str) -> type[BaseAgent]:
        """Return the agent class for *agent_type*, or raise."""
        try:
            return self._agents[agent_type]
        except KeyError:
            available = ", ".join(sorted(self._agents)) or "(none)"
            raise AgentNotFoundError(
                f"Agent type {agent_type!r} not found. Available: {available}"
            ) from None

    def get_instance(self, agent_type: str) -> BaseAgent:
        """Convenience: return a fresh instance of the agent."""
        return self.get(agent_type)()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_agents(self) -> dict[str, type[BaseAgent]]:
        """Return a copy of the internal registry dict."""
        return dict(self._agents)

    def __contains__(self, agent_type: str) -> bool:
        return agent_type in self._agents

    def __len__(self) -> int:
        return len(self._agents)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

agent_registry = AgentRegistry()
