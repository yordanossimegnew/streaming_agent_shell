"""FastAPI app: shell sessions, streamed turns (SSE) and human approval of risky commands."""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..agent import build_agent, open_checkpointer, transcript, validate_session_id
from ..tools import classify
from ..utils.config import Settings, load_settings
from .schemas import (
    ApprovalRequest,
    HealthResponse,
    MessageRequest,
    PolicyCheck,
    SessionDetail,
    SessionSummary,
    TurnStarted,
)
from .sessions import Session, SessionBusy, SessionManager


def create_app(settings: Settings | None = None, llm_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Streaming Agent Shell", version="1.0.0")
    checkpointer = open_checkpointer(settings.database_url)
    sessions = SessionManager(settings, checkpointer, llm_factory=llm_factory)

    def get_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    def summary(s: Session) -> dict:
        return {"session_id": s.id, "title": s.title, "status": s.status, "created_at": s.created_at}

    def started(s: Session, start: int) -> TurnStarted:
        return TurnStarted(session_id=s.id, start=start, stream_url=f"/sessions/{s.id}/stream?start={start}")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(model=settings.model, shell=settings.shell_path, sandbox=str(settings.sandbox),
                              approval_mode=settings.approval_mode)

    @app.get("/policy/check", response_model=PolicyCheck)
    def policy_check(command: str = Query(..., min_length=1)) -> PolicyCheck:
        """Dry-run the command policy without executing anything."""
        d = classify(command)
        return PolicyCheck(command=command, action=d.action, risk=d.risk, reason=d.reason)

    @app.post("/sessions", response_model=SessionSummary, status_code=201)
    def create_session() -> dict:
        return summary(sessions.create())

    @app.get("/sessions", response_model=list[SessionSummary])
    def list_sessions() -> list[dict]:
        return [summary(s) for s in sessions.list()]

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    def session_detail(session_id: str) -> dict:
        s = get_session(session_id)
        return {**summary(s), "pending": s.pending, "events": s.events}

    @app.post("/sessions/{session_id}/messages", response_model=TurnStarted, status_code=202)
    def send_message(session_id: str, request: MessageRequest) -> TurnStarted:
        s = get_session(session_id)
        try:
            return started(s, sessions.send(s, request.content))
        except SessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/approval", response_model=TurnStarted, status_code=202)
    def approve(session_id: str, request: ApprovalRequest) -> TurnStarted:
        s = get_session(session_id)
        try:
            return started(s, sessions.approve(s, request.decisions))
        except SessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/stream")
    def stream(session_id: str, start: int = Query(0, ge=0)) -> StreamingResponse:
        """Server-Sent Events for the current turn: tokens, tool calls, output lines, approvals."""
        s = get_session(session_id)

        def sse():
            for event in sessions.stream(s, start=start):
                yield ": keepalive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/sessions/{session_id}/transcript")
    def session_transcript(session_id: str) -> dict:
        """Message history from the LangGraph checkpointer."""
        try:
            validate_session_id(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        agent = build_agent(settings, checkpointer, llm=object())  # no model call needed to read state
        return {"session_id": session_id, "messages": transcript(agent, session_id)}

    return app
