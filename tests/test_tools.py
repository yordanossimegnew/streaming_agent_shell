"""Shell runner and write_file."""

from __future__ import annotations

import time

import pytest

from src.tools import SandboxError, run_streaming, write_file
from src.utils.config import default_shell

SHELL = default_shell()
pytestmark = pytest.mark.skipif(SHELL is None, reason="bash is required for shell tests")


def test_output_streams_line_by_line(tmp_path):
    seen: list[tuple[float, str]] = []
    result = run_streaming("for i in 1 2 3; do echo line $i; sleep 0.4; done", tmp_path, SHELL, 10,
                           on_line=lambda line: seen.append((time.monotonic(), line)))
    assert [line for _, line in seen] == ["line 1", "line 2", "line 3"]
    assert seen[2][0] - seen[0][0] > 0.5  # lines arrived over time, not all at the end
    assert result.exit_code == 0 and "line 3" in result.to_text()


def test_stderr_and_exit_code(tmp_path):
    result = run_streaming("echo out; echo err >&2; exit 3", tmp_path, SHELL, 10)
    assert result.output.splitlines() == ["out", "err"]
    assert result.exit_code == 3 and "[exit code 3" in result.to_text()


def test_runs_in_sandbox_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    (tmp_path / "marker.txt").write_text("here")
    result = run_streaming('ls; echo "key=[$ANTHROPIC_API_KEY]"', tmp_path, SHELL, 10)
    assert "marker.txt" in result.output
    assert "key=[]" in result.output


def test_timeout_kills_command(tmp_path):
    start = time.monotonic()
    result = run_streaming("echo started; sleep 30", tmp_path, SHELL, 2)
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - start < 15
    assert "timed out" in result.to_text()


def test_write_file_stays_in_sandbox(tmp_path):
    assert write_file(tmp_path, "src/app.py", "print(1)\n").startswith("Created")
    assert (tmp_path / "src" / "app.py").read_text() == "print(1)\n"
    with pytest.raises(SandboxError):
        write_file(tmp_path, "../escape.txt", "x")
