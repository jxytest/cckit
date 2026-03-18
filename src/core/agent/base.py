"""Abstract base class for all agents.

Subclass ``BaseAgent`` to define a new agent — typically only two methods
need implementing:

* ``config()`` — static declaration (who am I, what tools do I have)
* ``build_prompt(ctx)`` — dynamic prompt construction at runtime

Optional overrides:

* ``required_params()`` — declare required keys in ``extra_params``
* ``on_before_execute(ctx)`` — setup hook
* ``on_after_execute(ctx, result)`` — teardown hook
* ``on_error(ctx, error)`` — error recovery hook
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.agent.schemas import AgentConfig, AgentResult, ExecutionContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class every agent must inherit from."""

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def config(self) -> AgentConfig:
        """Return the static configuration for this agent type."""

    @abstractmethod
    def build_prompt(self, ctx: ExecutionContext) -> str:
        """Build the user prompt from the execution context."""

    # ------------------------------------------------------------------
    # Optional: parameter declaration
    # ------------------------------------------------------------------

    def required_params(self) -> list[str]:
        """Declare required keys in ``ctx.extra_params``.

        The executor validates these **before** execution starts and emits
        an ERROR event if any are missing.  Override to declare requirements::

            def required_params(self) -> list[str]:
                return ["test_file", "error_log"]
        """
        return []

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    async def on_before_execute(self, ctx: ExecutionContext) -> None:  # noqa: B027
        """Called before the SDK query begins.

        Override to perform setup such as writing fixture files into the
        workspace or fetching external data.
        """

    async def on_after_execute(  # noqa: B027
        self,
        ctx: ExecutionContext,
        result: AgentResult,
    ) -> None:
        """Called after the SDK query completes (success or failure).

        Override to perform teardown such as creating a merge request,
        uploading artefacts, or sending notifications.
        """

    async def on_error(  # noqa: B027
        self,
        ctx: ExecutionContext,
        error: Exception,
    ) -> None:
        """Called when the SDK query raises an exception.

        Override to perform error-specific recovery, cleanup, or alerting.
        This is called **before** ``on_after_execute``.
        """

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> str:
        return self.config().agent_type

    @property
    def display_name(self) -> str:
        cfg = self.config()
        return cfg.display_name or cfg.agent_type

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.agent_type!r}>"
