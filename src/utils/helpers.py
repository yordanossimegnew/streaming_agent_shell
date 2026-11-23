"""Small shared helpers."""

from __future__ import annotations


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long output so the model sees both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [{len(text) - limit} characters omitted] ...\n\n{text[-half:]}"
