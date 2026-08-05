"""Model resolution — merge Agent-level model config with Runner defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cckit.types import ModelConfig

if TYPE_CHECKING:
    from cckit.agent import Agent
    from cckit.types import RunContext


def resolve_model(
        agent: Agent,
        ctx: RunContext | None,
        base: ModelConfig,
        *,
        is_sub_agent: bool = False,
) -> ModelConfig:
    """Merge *agent*'s model_config with the runner default *base*.

    ``ctx.model`` is a caller-level override intended for the **top-level**
    agent only.  Sub-agents that declare their own ``model`` should honour
    that declaration; ``ctx.model`` must NOT shadow it.
    """
    agent_model = agent.model_config
    # ctx.model is only applied to the top-level agent, never to sub-agents
    # that explicitly declare their own model.
    override_model = (
        ""
        if is_sub_agent and agent_model is not None
        else (ctx.model if ctx is not None else "").strip()
    )

    if agent_model is None:
        if not override_model:
            return base
        return base.model_copy(update={"model": override_model})

    return ModelConfig(
        model=override_model or agent_model.model or base.model,
        api_key=agent_model.api_key or base.api_key,
        base_url=agent_model.base_url or base.base_url,
        max_tokens=agent_model.max_tokens,
        max_turns=agent_model.max_turns if agent_model.max_turns > 0 else base.max_turns,
        timeout_seconds=agent_model.timeout_seconds or base.timeout_seconds,
        # Thinking config: agent wins, else inherit runner default. Both the
        # new ``thinking`` field and the deprecated ``disable_thinking`` flag
        # are propagated so sub-agents keep their reasoning configuration.
        thinking=agent_model.thinking or base.thinking,
        disable_thinking=agent_model.disable_thinking or base.disable_thinking,
        supports_vision=agent_model.supports_vision and base.supports_vision,
        # Cost overrides use ``is None`` (not ``or``) so an explicit 0.0
        # (free model) is honoured instead of being masked by the base's
        # non-zero cost — ``0.0 or x`` would wrongly fall through to ``x``.
        input_cost_per_token=(
            agent_model.input_cost_per_token
            if agent_model.input_cost_per_token is not None
            else base.input_cost_per_token
        ),
        output_cost_per_token=(
            agent_model.output_cost_per_token
            if agent_model.output_cost_per_token is not None
            else base.output_cost_per_token
        ),
        # Inherit gateway mode + platform static headers (dimension,
        # feature-phase-name, …) from the runner default so sub-agent
        # requests carry the same gateway routing/identity. The bridge
        # derives a per-route ``custom-model-name`` from each sub's own
        # model, so it is intentionally not part of extra_headers here.
        cw_gateway=agent_model.cw_gateway or base.cw_gateway,
        extra_headers=agent_model.extra_headers or dict(base.extra_headers),
    )
