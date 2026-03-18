"""Business agents — auto-registers all agent implementations.

Importing this package triggers side-effect imports of every agent module,
which in turn registers each agent class via the ``@agent_registry.register``
decorator.  This keeps registration declarative — just drop a new file in
this directory and it becomes available.

Usage::

    import agents  # noqa: F401  — triggers auto-registration
    from core.agent.registry import agent_registry
    agent_cls = agent_registry.get("fix")
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def _auto_register() -> None:
    """Import every sub-module so their ``@agent_registry.register`` decorators fire."""
    package_path = __path__  # type: ignore[name-defined]
    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{__name__}.{module_info.name}"
        try:
            importlib.import_module(module_name)
            logger.debug("Auto-registered agent module: %s", module_name)
        except Exception:
            logger.exception("Failed to import agent module: %s", module_name)


_auto_register()
