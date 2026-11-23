"""FastAPI endpoints: sessions, SSE streaming, approvals."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api import create_app
from tests.fakes import ScriptedChatModel, cmd, say


def client_for(settings, script):
    shared = ScriptedChatModel(script)  # one model across turns so the script continues
    return TestClient(create_app(settings, llm_factory=lambda: shared))


def read_stream(client, url):
    with client.stream("GET", url) as response:
        return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_health_and_policy_check(settings):
    client = client_for(settings, [])
    assert client.get("/health").json()["approval_mode"] == "ask"
    body = client.get("/policy/check", params={"command": "rm -rf /"}).json()
    assert body["action"] == "deny" and body["risk"] == "high"


def test_full_session_with_approval(settings):
    client = client_for(settings, [
        say("Checking first.", cmd("c1", "ls")),
        say("Writing the file.", cmd("c2", "echo 42 > answer.txt", "store the answer")),
        say("All done."),
    ])
    sid = client.post("/sessions").json()["session_id"]

    turn = client.post(f"/sessions/{sid}/messages", json={"content": "store 42"})
    assert turn.status_code == 202
    events = read_stream(client, turn.json()["stream_url"])
    types = [e["type"] for e in events]
    assert types[0] == "user" and "token" in types and "output" not in types[:2]
    assert types[-2:] == ["approval_required", "turn_done"]
    assert events[-1]["status"] == "awaiting_approval"

    detail = client.get(f"/sessions/{sid}").json()
    assert detail["status"] == "awaiting_approval" and detail["pending"][0]["id"] == "c2"
    assert client.post(f"/sessions/{sid}/messages", json={"content": "x"}).status_code == 409
    assert client.post(f"/sessions/{sid}/approval", json={"decisions": {"nope": "approve"}}).status_code == 422

    turn = client.post(f"/sessions/{sid}/approval", json={"decisions": {"c2": "approve"}})
    events = read_stream(client, turn.json()["stream_url"])
    assert events[0]["type"] == "approval" and events[-1]["status"] == "completed"
    assert (settings.sandbox / "answer.txt").read_text().strip() == "42"

    assert client.get(f"/sessions/{sid}").json()["status"] == "idle"
    roles = [m["role"] for m in client.get(f"/sessions/{sid}/transcript").json()["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert client.get("/sessions").json()[0]["title"] == "store 42"


def test_errors(settings):
    client = client_for(settings, [])
    assert client.get("/sessions/missing").status_code == 404
    sid = client.post("/sessions").json()["session_id"]
    assert client.post(f"/sessions/{sid}/approval", json={"decisions": {}}).status_code == 409
    assert client.post(f"/sessions/{sid}/messages", json={"content": ""}).status_code == 422
