"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROVAL_MODES = ("ask", "auto", "readonly")


def default_shell() -> str | None:
    """bash if available (Linux/macOS, or Git Bash on Windows), else the system shell."""
    if sys.platform == "win32":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        if git_bash.exists():
            return str(git_bash)
    return shutil.which("bash")


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_steps: int = 25
    sandbox: Path = PROJECT_ROOT / "sandbox"
    shell: str | None = None  # None -> auto-detect
    command_timeout: int = 60
    # ask: risky commands need approval | auto: approve them automatically | readonly: refuse them
    approval_mode: str = "ask"
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    sessions_dir: Path = PROJECT_ROOT / "data" / "sessions"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"

    @property
    def shell_path(self) -> str | None:
        return self.shell or default_shell()


def load_settings(**overrides) -> Settings:
    """Build Settings from env vars, then apply any non-None keyword overrides."""
    load_dotenv(PROJECT_ROOT / ".env")
    mode = os.getenv("SHELL_APPROVAL_MODE", Settings.approval_mode).lower()
    if mode not in APPROVAL_MODES:
        raise ValueError(f"SHELL_APPROVAL_MODE must be one of {APPROVAL_MODES}, got {mode!r}")
    settings = Settings(
        model=os.getenv("AGENT_MODEL", Settings.model),
        effort=os.getenv("AGENT_EFFORT", Settings.effort),
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", Settings.max_tokens)),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", Settings.max_steps)),
        sandbox=Path(os.getenv("SHELL_SANDBOX", str(Settings.sandbox))),
        shell=os.getenv("SHELL_EXECUTABLE") or None,
        command_timeout=int(os.getenv("SHELL_COMMAND_TIMEOUT", Settings.command_timeout)),
        approval_mode=mode,
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    clean = {k: v for k, v in overrides.items() if v is not None}
    if "sandbox" in clean:
        clean["sandbox"] = Path(clean["sandbox"])
    return replace(settings, **clean)
