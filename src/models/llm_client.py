"""LangChain chat model for Claude.

The agent graph only needs something with `.invoke(messages) -> AIMessage`,
so tests can swap in a fake model.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from ..utils.config import Settings

# Server-side refusal fallback: if the primary model declines a request, the API
# re-runs it on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_chat_model(settings: Settings, tools: list[dict]):
    """ChatAnthropic bound to the agent's tools.

    Credentials resolve from ANTHROPIC_API_KEY (loaded from .env by load_settings).
    """
    model = ChatAnthropic(
        model=settings.model,
        max_tokens=settings.max_tokens,
        # Adaptive thinking is on by default for this model; effort sets its depth.
        output_config={"effort": settings.effort},
        betas=[FALLBACK_BETA],
        model_kwargs={
            "fallbacks": "default",
            # Cache the growing conversation prefix: each loop turn re-sends the history.
            "cache_control": {"type": "ephemeral"},
        },
    )
    return model.bind_tools(tools)
