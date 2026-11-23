"""Command policy: decide whether a shell command may run.

Every command the agent proposes is classified before anything executes:

    allow  read-only commands (ls, cat, grep, git status...)    -> run immediately
    ask    anything that changes state or runs code             -> needs human approval
    deny   dangerous or out-of-sandbox commands                 -> never runs

This is a guard rail, not a security boundary: shell syntax is too rich to parse
perfectly. Real isolation comes from running the API in a container (see Dockerfile).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

ALLOW, ASK, DENY = "allow", "ask", "deny"
_SEVERITY = {ALLOW: 0, ASK: 1, DENY: 2}
_RISK = {ALLOW: "low", ASK: "medium", DENY: "high"}


@dataclass(frozen=True)
class PolicyDecision:
    action: str  # allow | ask | deny
    reason: str

    @property
    def risk(self) -> str:
        return _RISK[self.action]

    def to_dict(self) -> dict:
        return {**asdict(self), "risk": self.risk}


# A program name in command position: at the start, or after ; & | ( or `sudo `.
# Anchoring avoids false hits on file names such as "ssh-notes.md" or "history.txt".
def _cmd(names: str) -> re.Pattern:
    return re.compile(r"(^|[;&|(]\s*|\bsudo\s+|\bxargs\s+)(" + names + r")(\s|$)")


# Whole-command patterns that are never allowed.
_DENY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_cmd(r"sudo|su|doas"), "privilege escalation"),
    (_cmd(r"shutdown|reboot|halt|poweroff"), "power control"),
    (_cmd(r"mkfs(\.\w+)?|fdisk|parted|diskpart|format"), "disk formatting"),
    (_cmd(r"dd"), "raw disk writes (dd)"),
    (re.compile(r":\(\)\s*\{"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b"), "piping a download into a shell"),
    (_cmd(r"eval"), "eval of dynamic code"),
    (_cmd(r"nc|ncat|netcat|telnet|ssh|scp|sftp|ftp|rsync"), "remote access / data transfer"),
    (_cmd(r"crontab|systemctl|service|launchctl|schtasks"), "system services / scheduling"),
    (_cmd(r"chown|chgrp"), "changing file ownership"),
    (re.compile(r"\bchmod\b.*\b(777|\+s|u\+s|g\+s)\b"), "unsafe permissions"),
    (_cmd(r"kill|pkill|killall|taskkill"), "killing processes"),
    (_cmd(r"history"), "shell history access"),
    (re.compile(r"/etc/(passwd|shadow|sudoers)"), "system credential files"),
]

# Read-only programs that run without approval.
_READ_ONLY = {
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "echo", "printf",
    "pwd", "tree", "du", "df", "stat", "file", "sort", "uniq", "diff", "cmp", "which", "whoami",
    "date", "uname", "basename", "dirname", "realpath", "cut", "tr", "nl", "jq", "true", "false",
    "test", "[", "env", "printenv", "ps", "md5sum", "sha256sum", "column", "seq", "sleep", "cd",
}
_GIT_READ_ONLY = {"status", "log", "diff", "show", "branch", "ls-files", "rev-parse", "blame", "shortlog", "tag"}
_VERSION_FLAGS = {"--version", "-V", "version"}
_OPERATORS = {";", "&&", "||", "|", "&", "(", ")", "|&"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>", "<", "<<", "<<<"}
_SAFE_ABSOLUTE = {"/dev/null", "/dev/stdout", "/dev/stderr"}
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_DRIVE_ANYWHERE = re.compile(r"(?<![\w/])[A-Za-z]:[\\/][^\s'\"]*")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def classify(command: str) -> PolicyDecision:
    command = command.strip()
    if not command:
        return PolicyDecision(DENY, "empty command")
    if len(command) > 4000:
        return PolicyDecision(DENY, "command is too long to review")

    lowered = command.lower()
    for pattern, reason in _DENY_PATTERNS:
        if pattern.search(lowered):
            return PolicyDecision(DENY, f"not allowed: {reason}")
    # Checked on the raw text: POSIX parsing would turn C:\Windows into C:Windows.
    drive = _WINDOWS_DRIVE_ANYWHERE.search(command)
    if drive:
        return PolicyDecision(DENY, f"'{drive.group(0)}' is an absolute path outside the sandbox")
    if "`" in command or "$(" in command:
        return PolicyDecision(ASK, "uses command substitution, so it can run anything")

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return PolicyDecision(ASK, "could not parse the command (unbalanced quotes?)")

    path_problem = _check_paths(tokens)
    if path_problem:
        return PolicyDecision(DENY, path_problem)

    decision: PolicyDecision | None = None
    for i, token in enumerate(tokens):
        if token in _REDIRECTS and token not in {"<", "<<", "<<<"}:
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            if target not in _SAFE_ABSOLUTE:
                decision = _worse(decision, PolicyDecision(ASK, "writes to a file (redirection)"))
    for segment in _segments(tokens):
        decision = _worse(decision, _classify_segment(segment))
    return decision or PolicyDecision(ALLOW, "read-only command")


def _segments(tokens: list[str]) -> list[list[str]]:
    segments, current = [], []
    for token in tokens:
        if token in _OPERATORS:
            if current:
                segments.append(current)
            current = []
        elif token in _REDIRECTS:
            current.append(token)
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _classify_segment(segment: list[str]) -> PolicyDecision:
    words = [w for w in segment if not _ASSIGNMENT.match(w)]
    if not words:
        return PolicyDecision(ALLOW, "variable assignment")
    program = PurePosixPath(words[0]).name.lower()
    args = words[1:]

    if program == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        if sub in _GIT_READ_ONLY:
            return PolicyDecision(ALLOW, f"read-only git {sub}")
        return PolicyDecision(ASK, f"git {sub or '(no subcommand)'} can change the repository")
    if args and args[0] in _VERSION_FLAGS:
        return PolicyDecision(ALLOW, "version check")
    if program == "find" and any(a in {"-delete", "-exec", "-execdir", "-ok"} for a in args):
        return PolicyDecision(ASK, "find with -delete/-exec changes or runs things")
    if program == "sed" and any(a == "-i" or a.startswith("-i") for a in args):
        return PolicyDecision(ASK, "sed -i edits files in place")
    if program in _READ_ONLY:
        return PolicyDecision(ALLOW, "read-only command")
    if program == "rm" and any(a.startswith("-") and "r" in a.lower() for a in args):
        return PolicyDecision(ASK, "recursive delete")
    if program in {"python", "python3", "node", "bash", "sh", "ruby", "perl", "deno", "bun"}:
        return PolicyDecision(ASK, f"runs a {program} program")
    if program in {"pip", "pip3", "npm", "pnpm", "yarn", "uv", "apt", "apt-get", "brew", "cargo"}:
        return PolicyDecision(ASK, f"{program} installs or changes packages")
    if program in {"curl", "wget"}:
        return PolicyDecision(ASK, "network access")
    return PolicyDecision(ASK, f"{program} may change files or state")


def _check_paths(tokens: list[str]) -> str | None:
    """Reject paths that point outside the sandbox (absolute, home, parent traversal)."""
    for token in tokens:
        if "://" in token or token in _OPERATORS or token in _REDIRECTS:
            continue
        value = token.split("=", 1)[1] if _ASSIGNMENT.match(token) else token
        if not value:
            continue
        if value.startswith("~"):
            return f"'{value}' points to the home directory, outside the sandbox"
        if _WINDOWS_PATH.match(value):
            return f"'{value}' is an absolute path outside the sandbox"
        if value.startswith("/") and value not in _SAFE_ABSOLUTE and not value.startswith("/dev/fd/"):
            return f"'{value}' is an absolute path outside the sandbox"
        if ".." in PurePosixPath(value).parts:
            depth = 0
            for part in PurePosixPath(value).parts:
                depth += -1 if part == ".." else (0 if part in {".", ""} else 1)
                if depth < 0:
                    return f"'{value}' climbs out of the sandbox"
    return None


def _worse(a: PolicyDecision | None, b: PolicyDecision) -> PolicyDecision:
    if a is None:
        return b
    return b if _SEVERITY[b.action] > _SEVERITY[a.action] else a
