"""write_file: create files in the sandbox without fragile shell heredocs."""

from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    pass


def resolve_in_sandbox(sandbox: Path, relative: str) -> Path:
    root = sandbox.resolve()
    target = (root / relative).resolve()
    if target != root and not target.is_relative_to(root):
        raise SandboxError(f"'{relative}' is outside the sandbox.")
    return target


def write_file(sandbox: Path, path: str, content: str) -> str:
    target = resolve_in_sandbox(sandbox, path)
    if target.is_dir():
        raise SandboxError(f"'{path}' is a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if existed else 'Created'} {path} ({len(content.splitlines())} lines)."
