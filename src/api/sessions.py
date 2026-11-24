"""Shell sessions: one LangGraph thread each, turns run on worker threads, events kept for streaming."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anthropic

from ..agent import build_agent, run_turn
from ..utils.config import Settings
from ..utils.logger import get_logger

log = get_logger("sessions")
MAX_SESSIONS = 100


class SessionBusy(Exception):
    pass


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    title: str = ""
    status: str = "idle"  # idle | running | awaiting_approval
    pending: list[dict] | None = None
    events: list[dict] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition, repr=False)


class SessionManager:
    def __init__(self, settings: Settings, checkpointer: Any, llm_factory: Callable[[], Any] | None = None):
        self.settings = settings
        self.checkpointer = checkpointer
        self.llm_factory = llm_factory
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ queries
    def create(self) -> Session:
        session = Session(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._sessions[session.id] = session
            for old in sorted(self._sessions.values(), key=lambda s: s.created_at)[:-MAX_SESSIONS]:
                if old.status != "running":
                    del self._sessions[old.id]
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    # ------------------------------------------------------------------ turns
    def send(self, session: Session, text: str) -> int:
        """Start a turn with a user message. Returns the event index to stream from."""
        with session.cond:
            if session.status != "idle":
                raise SessionBusy(f"Session is {session.status.replace('_', ' ')}.")
            session.title = session.title or text[:80]
            return self._start(session, {"message": text}, {"type": "user", "text": text})

    def approve(self, session: Session, decisions: dict[str, str]) -> int:
        """Resume a paused turn with {"<call id>": "approve" | "reject"}."""
        with session.cond:
            if session.status != "awaiting_approval":
                raise SessionBusy("Nothing is waiting for approval.")
            unknown = set(decisions) - {p["id"] for p in session.pending or []}
            if unknown:
                raise ValueError(f"Unknown command ids: {sorted(unknown)}")
            return self._start(session, {"decisions": decisions}, {"type": "approval", "decisions": decisions})

    def _start(self, session: Session, turn_kwargs: dict, first_event: dict) -> int:
        start = len(session.events)
        session.status, session.pending = "running", None
        self._append(session, first_event)
        threading.Thread(target=self._worker, args=(session, turn_kwargs), daemon=True,
                         name=f"turn-{session.id}").start()
        return start

    def stream(self, session: Session, start: int = 0, keepalive: float = 15.0) -> Iterator[dict | None]:
        """Yield events from `start` until the current turn ends (None = keepalive tick)."""
        index = start
        while True:
            with session.cond:
                session.cond.wait_for(lambda: len(session.events) > index or session.status != "running",  # noqa: B023 (called before index changes)
                                      timeout=keepalive)
                batch = session.events[index:]
                finished = session.status != "running"
            for event in batch:
                yield event
            index += len(batch)
            if not batch:
                if finished:
                    return
                yield None

    # ------------------------------------------------------------------ worker
    def _append(self, session: Session, event: dict) -> None:
        with session.cond:
            session.events.append({"seq": len(session.events), "time": round(time.time(), 3), **event})
            session.cond.notify_all()

    def _worker(self, session: Session, turn_kwargs: dict) -> None:
        try:
            agent = build_agent(self.settings, self.checkpointer, llm=self.llm_factory() if self.llm_factory else None)
            result = run_turn(agent, session.id, lambda e: self._append(session, e), **turn_kwargs)
        except anthropic.AuthenticationError:
            result = self._error(session, "Authentication failed: set ANTHROPIC_API_KEY in .env.")
        except anthropic.RateLimitError:
            result = self._error(session, "Rate limited by the model API; retry later.")
        except anthropic.APIError as exc:
            result = self._error(session, f"Model API error: {exc}")
        except Exception as exc:
            log.exception("turn crashed in session %s", session.id)
            result = self._error(session, f"{type(exc).__name__}: {exc}")

        # Publish turn_done before flipping status, so streams never end without it.
        self._append(session, {"type": "turn_done", **result})
        with session.cond:
            session.pending = result.get("pending")
            session.status = "awaiting_approval" if session.pending else "idle"
            session.cond.notify_all()

    def _error(self, session: Session, message: str) -> dict:
        self._append(session, {"type": "error", "message": message})
        return {"status": "error", "final_text": message, "usage": None, "pending": None}
