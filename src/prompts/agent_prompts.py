"""Prompt assembly: system prompt + project knowledge base + per-task wrappers."""

from __future__ import annotations

from pathlib import Path

from .system_prompts import SYSTEM_PROMPT

TRUNCATED_TOOL_CALL = (
    "Your previous response hit the output token limit before this tool call was complete, "
    "so it was not run. Retry it."
)


def load_knowledge_base(directory: Path) -> str:
    """Concatenate markdown notes from data/knowledge_base/ (sorted, so the prompt is stable)."""
    if not directory.is_dir():
        return ""
    parts = []
    for path in sorted(directory.glob("*.md")):
        parts.append(f'<document name="{path.name}">\n{path.read_text(encoding="utf-8").strip()}\n</document>')
    return "\n".join(parts)


def build_system_prompt(knowledge_base_dir: Path) -> str:
    kb = load_knowledge_base(knowledge_base_dir)
    if not kb:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\nShell guidelines to follow:\n<knowledge_base>\n{kb}\n</knowledge_base>\n"
