"""Session memory: a LangGraph SQLite checkpointer (also what makes interrupt/resume work)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    if not _SAFE_ID.match(session_id):
        raise ValueError(f"Invalid session id: {session_id!r}")
    return session_id


def open_checkpointer(directory: Path) -> SqliteSaver:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(directory / "sessions.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
