"""Run one turn of the graph and translate LangGraph's streams into simple events.

LangGraph stream modes used:
  messages -> (AIMessageChunk, metadata): Claude's reply, token by token
  custom   -> dicts written by nodes via get_stream_writer(): tool calls, output lines
  updates  -> per-node state updates; carries "__interrupt__" when the graph pauses
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from .agent import ShellAgent, message_text, stop_reason
from .state import Usage

EmitFn = Callable[[dict], None]


def chunk_text(chunk: AIMessageChunk) -> str:
    """Visible text in a streamed chunk (thinking blocks are skipped)."""
    if isinstance(chunk.content, str):
        return chunk.content
    return "".join(
        b.get("text", "") for b in chunk.content if isinstance(b, dict) and b.get("type") == "text"
    )


def run_turn(agent: ShellAgent, session_id: str, emit: EmitFn, *, message: str | None = None,
             decisions: dict[str, str] | None = None) -> dict:
    """Start a turn with a user message, or resume a paused one with approval decisions."""
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": agent.settings.max_steps * 3 + 10}
    before = len(agent.graph.get_state(config).values.get("messages", []))
    if message is not None:
        graph_input: Any = {"messages": [HumanMessage(message)], "steps": 0}
    else:
        graph_input = Command(resume=decisions or {})

    pending = None
    for mode, chunk in agent.graph.stream(graph_input, config, stream_mode=["messages", "custom", "updates"]):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessageChunk):
                if text := chunk_text(msg):
                    emit({"type": "token", "text": text})
        elif mode == "custom":
            emit(chunk)
        elif mode == "updates" and "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value["pending"]
            emit({"type": "approval_required", "pending": pending})

    messages = agent.graph.get_state(config).values.get("messages", [])
    new = messages[before:]
    usage = vars(Usage.from_messages(new))
    if pending:
        return {"status": "awaiting_approval", "final_text": "", "usage": usage, "pending": pending}

    last = messages[-1] if messages else None
    if isinstance(last, ToolMessage):
        status = "max_steps"
    elif isinstance(last, AIMessage):
        status = {"end_turn": "completed", "refusal": "refusal", "max_tokens": "max_tokens"}.get(
            stop_reason(last), "completed" if not last.tool_calls else "error")
    else:
        status = "error"
    final = message_text(last) if isinstance(last, AIMessage) else ""
    return {"status": status, "final_text": final, "usage": usage, "pending": None}


def transcript(agent: ShellAgent, session_id: str) -> list[dict]:
    """Plain-dict history of a session from the checkpointer."""
    config = {"configurable": {"thread_id": session_id}}
    out = []
    for msg in agent.graph.get_state(config).values.get("messages", []):
        if isinstance(msg, HumanMessage):
            out.append({"role": "user", "text": msg.content if isinstance(msg.content, str) else str(msg.content)})
        elif isinstance(msg, AIMessage):
            out.append({"role": "assistant", "text": message_text(msg),
                        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]})
        elif isinstance(msg, ToolMessage):
            out.append({"role": "tool", "name": msg.name, "text": str(msg.content), "is_error": msg.status == "error"})
    return out
