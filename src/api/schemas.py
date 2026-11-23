"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000, description="What the agent should do.")


class ApprovalRequest(BaseModel):
    decisions: dict[str, Literal["approve", "reject"]] = Field(
        ..., description='Map of pending command id -> "approve" or "reject".'
    )


class TurnStarted(BaseModel):
    session_id: str
    stream_url: str = Field(..., description="Server-Sent Events URL for this turn (starts at `start`).")
    start: int


class SessionSummary(BaseModel):
    session_id: str
    title: str
    status: str
    created_at: float


class SessionDetail(SessionSummary):
    pending: list[dict] | None
    events: list[dict]


class PolicyCheck(BaseModel):
    command: str
    action: str
    risk: str
    reason: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    shell: str | None
    sandbox: str
    approval_mode: str
