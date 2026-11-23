"""Core agent package plus a factory that wires everything together."""

from __future__ import annotations

from typing import Any

from ..models import build_chat_model
from ..prompts import build_system_prompt
from ..tools import TOOL_DEFINITIONS
from ..utils.config import Settings
from .agent import ShellAgent
from .memory import open_checkpointer, validate_session_id
from .runner import run_turn, transcript
from .state import ShellState, Usage


def build_agent(settings: Settings, checkpointer: Any, llm: Any | None = None) -> ShellAgent:
    settings.sandbox.mkdir(parents=True, exist_ok=True)
    return ShellAgent(
        llm=llm or build_chat_model(settings, TOOL_DEFINITIONS),
        settings=settings,
        system_prompt=build_system_prompt(settings.knowledge_base_dir),
        checkpointer=checkpointer,
    )


__all__ = [
    "ShellAgent", "ShellState", "Usage", "build_agent", "open_checkpointer", "run_turn",
    "transcript", "validate_session_id",
]
