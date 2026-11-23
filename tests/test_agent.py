"""The LangGraph shell agent: streaming, policy, interrupt/resume approvals."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.messages import ToolMessage

from src.agent import build_agent, open_checkpointer, run_turn
from tests.fakes import ScriptedChatModel, cmd, say, write


def make(settings, script):
    llm = ScriptedChatModel(script)
    return build_agent(settings, open_checkpointer(settings.sessions_dir), llm=llm), llm


def turn(agent, **kwargs):
    events = []
    result = run_turn(agent, "s1", events.append, **kwargs)
    return result, events


def kinds(events):
    return [e["type"] for e in events]


def test_tokens_stream_and_read_only_commands_run(settings):
    (settings.sandbox).mkdir(parents=True)
    (settings.sandbox / "hello.txt").write_text("hi")
    agent, llm = make(settings, [say("Listing the files now.", cmd("c1", "ls")), say("There is one file.")])
    result, events = turn(agent, message="what's here?")

    assert result["status"] == "completed" and result["final_text"] == "There is one file."
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 3 and "".join(tokens).startswith("Listing the files now.")
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["policy"]["action"] == "allow" and call["effective"] == "allow"
    assert {"type": "command_start", "call_id": "c1", "command": "ls"} in events
    assert {"type": "output", "call_id": "c1", "line": "hello.txt"} in events
    assert next(e for e in events if e["type"] == "command_done")["exit_code"] == 0
    assert "hello.txt" in llm.calls[1][-1].content


def test_risky_command_pauses_for_approval_then_runs(settings):
    agent, llm = make(settings, [say("Creating a note.", cmd("c1", "echo hello > note.txt", "save a note")),
                                 say("Saved.")])
    result, events = turn(agent, message="save a note")
    assert result["status"] == "awaiting_approval"
    assert kinds(events)[-1] == "approval_required"
    assert result["pending"][0]["args"]["command"] == "echo hello > note.txt"
    assert not (settings.sandbox / "note.txt").exists()  # nothing ran yet

    result, events = turn(agent, decisions={"c1": "approve"})
    assert result["status"] == "completed"
    assert (settings.sandbox / "note.txt").read_text().strip() == "hello"
    assert "command_done" in kinds(events)


def test_rejected_command_is_reported_to_model(settings):
    agent, llm = make(settings, [say("", cmd("c1", "pip install leftpad")), say("Okay, skipping that.")])
    turn(agent, message="install leftpad")
    result, events = turn(agent, decisions={"c1": "reject"})
    assert result["status"] == "completed"
    assert {"type": "refused", "call_id": "c1", "reason": "The user rejected this command."} in events
    tool_msg = llm.calls[1][-1]
    assert isinstance(tool_msg, ToolMessage) and tool_msg.status == "error"


def test_denied_command_never_asks(settings):
    agent, llm = make(settings, [say("", cmd("c1", "sudo rm -rf /")), say("I can't do that.")])
    result, events = turn(agent, message="wipe it")
    assert result["status"] == "completed" and "approval_required" not in kinds(events)
    refused = next(e for e in events if e["type"] == "refused")
    assert "privilege escalation" in refused["reason"]


def test_mixed_calls_in_one_turn(settings):
    agent, llm = make(settings, [
        say("", cmd("c1", "ls"), write("c2", "app.py", "print('hi')\n"), cmd("c3", "cat /etc/passwd")),
        say("Done."),
    ])
    result, events = turn(agent, message="go")
    assert [p["id"] for p in result["pending"]] == ["c2"]  # only the risky one waits
    result, events = turn(agent, decisions={"c2": "approve"})
    assert (settings.sandbox / "app.py").exists()
    results = llm.calls[1][-3:]
    assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
    assert [m.status for m in results] == ["success", "success", "error"]


def test_approval_modes(settings):
    agent, _ = make(replace(settings, approval_mode="auto"), [say("", cmd("c1", "echo x > a.txt")), say("ok")])
    result, _ = turn(agent, message="go")
    assert result["status"] == "completed" and (settings.sandbox / "a.txt").exists()

    agent, _ = make(replace(settings, approval_mode="readonly", sessions_dir=settings.sessions_dir / "ro"),
                    [say("", cmd("c1", "echo x > b.txt")), say("ok")])
    result, events = turn(agent, message="go")
    assert result["status"] == "completed" and not (settings.sandbox / "b.txt").exists()
    assert "read-only mode" in next(e for e in events if e["type"] == "refused")["reason"]


def test_failed_command_is_an_error_result(settings):
    agent, llm = make(settings, [say("", cmd("c1", "ls missing-dir")), say("It doesn't exist.")])
    turn(agent, message="list it")
    msg = llm.calls[1][-1]
    assert msg.status == "error" and "exit code" in msg.content


def test_step_limit(settings):
    script = [say("", cmd(f"c{i}", "ls")) for i in range(10)]
    agent, _ = make(settings, script)
    result, _ = turn(agent, message="loop")
    assert result["status"] == "max_steps"
