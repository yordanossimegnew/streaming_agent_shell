from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

from src.utils.config import Settings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def fresh_database() -> str:
    """The test database with LangGraph's checkpoint tables emptied, so every test starts clean."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set: point it at an empty PostgreSQL database (see .env.example)")
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DO $$ BEGIN IF to_regclass('checkpoints') IS NOT NULL THEN "
                     "TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes; END IF; END $$")
    return TEST_DATABASE_URL


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        sandbox=tmp_path / "sandbox",
        database_url=fresh_database(),
        knowledge_base_dir=tmp_path / "kb",
        log_dir=tmp_path / "logs",
        command_timeout=15,
        max_steps=6,
    )
