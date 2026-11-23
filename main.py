"""Entry point. The project is a FastAPI application; an interactive terminal CLI is included.

    uvicorn main:app --reload           # API server (docs at /docs)
    python main.py                      # same, on port 8000
    python main.py cli                  # chat with the agent in your terminal
"""

from __future__ import annotations

import argparse
import sys

from src.api import create_app
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

DIM, BOLD, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")


_ASKED: set[str] = set()  # commands that waited for approval


def render(event: dict) -> None:
    kind = event["type"]
    if kind == "token":
        print(event["text"], end="", flush=True)
    elif kind == "tool_call":
        if event["effective"] == "ask":
            _ASKED.add(event["id"])
        target = event["args"].get("command") or f"write_file {event['args'].get('path')}"
        tag = {"allow": GREEN + "auto", "ask": YELLOW + "needs approval", "deny": RED + "refused"}[event["effective"]]
        print(f"\n{BOLD}{CYAN}$ {target}{RESET}  {DIM}[{tag}{RESET}{DIM}]{RESET}")
    elif kind == "command_start" and event["call_id"] in _ASKED:
        print(f"{BOLD}{CYAN}$ {event['command']}{RESET}  {DIM}[{GREEN}approved{RESET}{DIM}]{RESET}")
    elif kind == "output":
        print(f"{DIM}{event['line']}{RESET}")
    elif kind == "command_done":
        color = GREEN if event["exit_code"] == 0 else RED
        status = "timed out" if event["timed_out"] else f"exit {event['exit_code']}"
        print(f"{color}  ↳ {status} · {event['duration']}s{RESET}")
    elif kind == "refused":
        print(f"{RED}  ✗ {event['reason']}{RESET}")
    elif kind == "error":
        print(f"\n{RED}{event['message']}{RESET}")


def cli() -> None:
    from src.agent import open_checkpointer
    from src.api.sessions import SessionManager

    manager = SessionManager(settings, open_checkpointer(settings.sessions_dir))
    session = manager.create()
    print(f"{DIM}sandbox: {settings.sandbox} · shell: {settings.shell_path} · mode: {settings.approval_mode}{RESET}")
    print("Type a task. 'exit' to quit.\n")
    while True:
        try:
            text = input(f"{BOLD}you ❯ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        start = manager.send(session, text)
        while True:
            for event in manager.stream(session, start):
                if event:
                    render(event)
            if session.status != "awaiting_approval":
                break
            decisions = {}
            for item in session.pending:
                target = item["args"].get("command") or f"write_file {item['args'].get('path')}"
                print(f"\n{YELLOW}Approve?{RESET} {BOLD}{target}{RESET}\n  {DIM}why: {item['args'].get('reason', '')}"
                      f" · policy: {item['reason']}{RESET}")
                answer = input("  [y/N] ").strip().lower()
                decisions[item["id"]] = "approve" if answer in {"y", "yes"} else "reject"
            start = manager.approve(session, decisions)
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Agent Shell")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Run the FastAPI server (default).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("cli", help="Chat with the agent in this terminal.")
    args = parser.parse_args()
    if args.command == "cli":
        cli()
    else:
        import uvicorn

        uvicorn.run(app, host=getattr(args, "host", "0.0.0.0"), port=getattr(args, "port", 8000))


if __name__ == "__main__":
    sys.exit(main())
