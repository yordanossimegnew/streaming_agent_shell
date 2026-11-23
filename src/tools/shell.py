"""Run a command in the sandbox and stream its output line by line."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..utils.helpers import truncate

MAX_RESULT_CHARS = 12_000
# Only these variables reach commands, so secrets like ANTHROPIC_API_KEY never leak.
_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TZ")


@dataclass
class CommandResult:
    command: str
    exit_code: int | None
    output: str
    duration: float
    timed_out: bool = False

    def to_text(self) -> str:
        status = "timed out (killed)" if self.timed_out else f"exit code {self.exit_code}"
        body = truncate(self.output, MAX_RESULT_CHARS) if self.output else "(no output)"
        return f"$ {self.command}\n{body}\n[{status}, {self.duration:.1f}s]"


def sandbox_env(sandbox: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update({"HOME": str(sandbox), "TERM": "dumb", "PYTHONUNBUFFERED": "1", "PAGER": "cat", "GIT_PAGER": "cat"})
    tmp = sandbox / ".tmp"
    tmp.mkdir(exist_ok=True)
    env.update({"TMPDIR": str(tmp), "TEMP": str(tmp), "TMP": str(tmp)})
    return env


def run_streaming(
    command: str,
    sandbox: Path,
    shell: str | None,
    timeout: int,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    """Run `command` with cwd=sandbox; call on_line(line) for each output line as it appears."""
    sandbox.mkdir(parents=True, exist_ok=True)
    argv = [shell, "-c", command] if shell else command
    popen_kwargs = dict(
        cwd=sandbox,
        env=sandbox_env(sandbox),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # interleave stderr with stdout, in order
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=shell is None,
    )
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True  # own process group, so we can kill children

    start = time.monotonic()
    proc = subprocess.Popen(argv, **popen_kwargs)
    timed_out = threading.Event()

    def kill_on_timeout() -> None:
        if proc.poll() is None:
            timed_out.set()
            _kill_tree(proc)

    timer = threading.Timer(timeout, kill_on_timeout)
    timer.start()
    lines: list[str] = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            lines.append(line)
            if on_line:
                on_line(line)
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            _kill_tree(proc)
    return CommandResult(
        command=command,
        exit_code=None if timed_out.is_set() else proc.returncode,
        output="\n".join(lines),
        duration=time.monotonic() - start,
        timed_out=timed_out.is_set(),
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
