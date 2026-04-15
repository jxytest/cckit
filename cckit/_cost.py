"""Cost calculation utilities for cckit agent runs.

Responsible for recalculating per-model token costs after a run completes,
replacing the costUSD values emitted by Claude Code (which only knows
Anthropic's first-party pricing) with values from either:

1. User-configured per-token rates on :class:`~cckit.types.ModelConfig`, or
2. LiteLLM's built-in price table (supports hundreds of providers).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cckit.types import ModelConfig

logger = logging.getLogger(__name__)


def recalculate_model_usage_costs(
    model_usage: dict[str, Any],
    configs: dict[str, "ModelConfig"],
) -> dict[str, Any]:
    """Recalculate ``costUSD`` for each model in *model_usage*.

    Parameters
    ----------
    model_usage:
        The ``model_usage`` dict from ``ResultMessage`` (keyed by short model
        name, e.g. ``"glm-5"``), as produced by Claude Code.
    configs:
        Mapping of short model name → :class:`~cckit.types.ModelConfig` for
        all models used in this run (primary + sub-agents).  Models absent
        from *configs* still receive a LiteLLM price-table lookup.

    Returns
    -------
    dict
        A *new* dict with the same structure but recalculated ``costUSD``
        values.  All other fields (``inputTokens``, ``outputTokens``, etc.)
        are preserved unchanged.
    """
    result: dict[str, Any] = {}
    for model_name, usage in model_usage.items():
        if not isinstance(usage, dict):
            result[model_name] = usage
            continue

        input_tokens = int(usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("outputTokens") or 0)
        cache_read = int(usage.get("cacheReadInputTokens") or 0)
        cache_creation = int(usage.get("cacheCreationInputTokens") or 0)

        new_cost = _cost_for_model(
            model_name=model_name,
            configs=configs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )
        result[model_name] = {**usage, "costUSD": new_cost}

    return result


def _cost_for_model(
    model_name: str,
    configs: dict[str, "ModelConfig"],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Return the USD cost for one model's token counts.

    Uses ``ModelConfig.input_cost_per_token`` / ``output_cost_per_token`` when
    the user has configured explicit rates; otherwise delegates to
    ``litellm.cost_per_token`` with the model's full LiteLLM name.

    Returns ``0.0`` if cost cannot be determined.
    """
    cfg = configs.get(model_name)
    litellm_model = cfg.model if cfg is not None else model_name

    # Build custom_cost_per_token when user has configured explicit rates
    custom_cost: Any = None
    if (
        cfg is not None
        and cfg.input_cost_per_token is not None
        and cfg.output_cost_per_token is not None
    ):
        try:
            _litellm = importlib.import_module("litellm")
            custom_cost = _litellm.CostPerToken(
                input_cost_per_token=cfg.input_cost_per_token,
                output_cost_per_token=cfg.output_cost_per_token,
            )
        except Exception:
            logger.debug("Failed to build CostPerToken for model %s", model_name, exc_info=True)

    # Delegate to litellm.cost_per_token — handles cache tokens natively
    try:
        _litellm = importlib.import_module("litellm")
        prompt_cost, completion_cost = _litellm.cost_per_token(
            model=litellm_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
            custom_cost_per_token=custom_cost,
        )
        return prompt_cost + completion_cost
    except Exception:
        logger.debug("litellm.cost_per_token failed for model %s", model_name, exc_info=True)

    return 0.0
