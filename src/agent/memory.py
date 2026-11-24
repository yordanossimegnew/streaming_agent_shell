"""Session memory: a LangGraph PostgreSQL checkpointer (also what makes interrupt/resume work)."""

from __future__ import annotations

import re

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    if not _SAFE_ID.match(session_id):
        raise ValueError(f"Invalid session id: {session_id!r}")
    return session_id


def open_checkpointer(database_url: str) -> PostgresSaver:
    """Connection-pooled LangGraph checkpointer; creates its tables on first use."""
    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(conninfo, min_size=1, max_size=10, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
