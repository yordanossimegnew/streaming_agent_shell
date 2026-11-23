"""Tools the agent can call, plus the policy that gates them."""

from __future__ import annotations

from .files import SandboxError, resolve_in_sandbox, write_file
from .policy import ALLOW, ASK, DENY, PolicyDecision, classify
from .shell import CommandResult, run_streaming


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TOOL_DEFINITIONS = [
    {
        "name": "run_command",
        "description": (
            "Run a bash command with the sandbox folder as the working directory. Output streams "
            "to the user live; you get stdout+stderr and the exit code. Read-only commands run "
            "immediately; commands that change files, install packages or run code need the "
            "user's approval; dangerous or out-of-sandbox commands are refused. Use relative paths."
        ),
        "input_schema": _schema({
            "command": {"type": "string", "description": "The bash command to run."},
            "reason": {"type": "string", "description": "One short sentence on why, shown to the user when approval is needed."},
        }),
        "strict": True,
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file in the sandbox (needs approval). Prefer this over heredocs.",
        "input_schema": _schema({
            "path": {"type": "string", "description": "Path relative to the sandbox."},
            "content": {"type": "string", "description": "Full file content."},
        }),
        "strict": True,
    },
]


def review(call: dict) -> PolicyDecision:
    """Policy decision for a tool call."""
    if call["name"] == "run_command":
        return classify(call["args"].get("command", ""))
    if call["name"] == "write_file":
        return PolicyDecision(ASK, f"creates or overwrites {call['args'].get('path', '?')}")
    return PolicyDecision(DENY, f"unknown tool {call['name']}")


__all__ = [
    "ALLOW", "ASK", "DENY", "CommandResult", "PolicyDecision", "SandboxError", "TOOL_DEFINITIONS",
    "classify", "resolve_in_sandbox", "review", "run_streaming", "write_file",
]
