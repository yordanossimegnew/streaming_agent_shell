from __future__ import annotations

import pytest

from src.utils.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        sandbox=tmp_path / "sandbox",
        sessions_dir=tmp_path / "sessions",
        knowledge_base_dir=tmp_path / "kb",
        log_dir=tmp_path / "logs",
        command_timeout=15,
        max_steps=6,
    )
