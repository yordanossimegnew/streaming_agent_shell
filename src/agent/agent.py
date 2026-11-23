"""The shell agent as a LangGraph state machine with human-in-the-loop approval.

    START -> agent -(tool calls)-> review -> shell -> agent -> ... -(answer)-> END
                                     |
                         interrupt() for risky commands: the graph pauses, the
                         checkpointer saves it, and it resumes with the user's decision

* agent  - Claude (LangChain ChatAnthropic) streams its reply token by token.
* review - applies the command policy. No side effects, so it is safe for LangGraph
           to re-run it when the graph resumes after an interrupt.
* shell  - runs approved commands, streaming each output line via the custom stream.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..tools import ALLOW, ASK, DENY, PolicyDecision, SandboxError, review, run_streaming, write_file
from ..utils.config import Settings
from ..utils.logger import get_logger
from .state import ShellState

log = get_logger("agent")

TRUNCATED = "Your response hit the output limit before this tool call was complete, so it was not run. Retry it."


def stop_reason(msg: AIMessage) -> str | None:
    return (msg.response_metadata or {}).get("stop_reason")


def message_text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


class ShellAgent:
    def __init__(self, llm: Any, settings: Settings, system_prompt: str, checkpointer: Any):
        self.llm = llm
        self.settings = settings
        self.system_prompt = system_prompt
        self.graph = self._build(checkpointer)

    # ------------------------------------------------------------------ nodes
    def _agent(self, state: ShellState) -> dict:
        response = self.llm.invoke([SystemMessage(self.system_prompt), *state["messages"]])
        writer = get_stream_writer()
        for call in response.tool_calls:
            policy = review(call)
            writer({"type": "tool_call", "id": call["id"], "name": call["name"], "args": call["args"],
                    "policy": policy.to_dict(), "effective": self._effective(policy).action})
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def _review(self, state: ShellState) -> dict:
        last: AIMessage = state["messages"][-1]
        if stop_reason(last) == "max_tokens":
            calls = [*last.tool_calls, *last.invalid_tool_calls]
            return {"verdicts": {c["id"]: {"run": False, "reason": TRUNCATED} for c in calls if c.get("id")}}

        verdicts, pending = {}, []
        for call in last.tool_calls:
            decision = review(call)
            effective = self._effective(decision)
            if effective.action == ALLOW:
                auto = decision.action == ASK
                verdicts[call["id"]] = {"run": True, "reason": "auto-approved" if auto else decision.reason}
            elif effective.action == DENY:
                verdicts[call["id"]] = {"run": False, "reason": f"Refused by policy: {effective.reason}."}
            else:
                pending.append({"id": call["id"], "name": call["name"], "args": call["args"],
                                "reason": decision.reason, "risk": decision.risk})

        if pending:
            # Pause the graph. The API resumes it with {"<call id>": "approve" | "reject"}.
            answers = interrupt({"pending": pending})
            for item in pending:
                approved = str(answers.get(item["id"], "reject")).lower() == "approve"
                verdicts[item["id"]] = {"run": approved,
                                        "reason": "approved by user" if approved else "The user rejected this command."}
        return {"verdicts": verdicts}

    def _shell(self, state: ShellState) -> dict:
        last: AIMessage = state["messages"][-1]
        verdicts = state.get("verdicts", {})
        writer = get_stream_writer()
        calls = [*last.tool_calls, *(last.invalid_tool_calls if stop_reason(last) == "max_tokens" else [])]
        results = []
        for call in calls:
            verdict = verdicts.get(call["id"], {"run": False, "reason": "No decision was recorded."})
            if not verdict["run"]:
                writer({"type": "refused", "call_id": call["id"], "reason": verdict["reason"]})
                results.append(ToolMessage(verdict["reason"], tool_call_id=call["id"], name=call.get("name") or "", status="error"))
                continue
            results.append(self._execute(call, writer))
        return {"messages": results}

    def _execute(self, call: dict, writer) -> ToolMessage:
        args, call_id = call["args"], call["id"]
        label = args.get("command") if call["name"] == "run_command" else f"write_file {args.get('path')}"
        writer({"type": "command_start", "call_id": call_id, "command": label})
        if call["name"] == "write_file":
            try:
                text = write_file(self.settings.sandbox, args["path"], args["content"])
                writer({"type": "output", "call_id": call_id, "line": text})
                writer({"type": "command_done", "call_id": call_id, "exit_code": 0, "duration": 0.0, "timed_out": False})
                return ToolMessage(text, tool_call_id=call_id, name="write_file")
            except SandboxError as exc:
                writer({"type": "refused", "call_id": call_id, "reason": str(exc)})
                return ToolMessage(str(exc), tool_call_id=call_id, name="write_file", status="error")

        result = run_streaming(
            args["command"], self.settings.sandbox, self.settings.shell_path, self.settings.command_timeout,
            on_line=lambda line: writer({"type": "output", "call_id": call_id, "line": line}),
        )
        writer({"type": "command_done", "call_id": call_id, "exit_code": result.exit_code,
                "duration": round(result.duration, 2), "timed_out": result.timed_out})
        failed = result.timed_out or result.exit_code != 0
        return ToolMessage(result.to_text(), tool_call_id=call_id, name="run_command",
                           status="error" if failed else "success")

    # ------------------------------------------------------------------ routing
    def _after_agent(self, state: ShellState) -> str:
        last: AIMessage = state["messages"][-1]
        if last.tool_calls or (stop_reason(last) == "max_tokens" and last.invalid_tool_calls):
            return "review"
        if stop_reason(last) == "pause_turn" and state.get("steps", 0) < self.settings.max_steps:
            return "agent"
        return END

    def _after_shell(self, state: ShellState) -> str:
        return END if state.get("steps", 0) >= self.settings.max_steps else "agent"

    def _build(self, checkpointer):
        g = StateGraph(ShellState)
        g.add_node("agent", self._agent)
        g.add_node("review", self._review)
        g.add_node("shell", self._shell)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._after_agent, ["review", "agent", END])
        g.add_edge("review", "shell")
        g.add_conditional_edges("shell", self._after_shell, ["agent", END])
        return g.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ helpers
    def _effective(self, decision: PolicyDecision) -> PolicyDecision:
        """Apply the approval mode: auto approves 'ask', readonly refuses it."""
        if decision.action != ASK:
            return decision
        if self.settings.approval_mode == "auto":
            return PolicyDecision(ALLOW, f"auto-approved ({decision.reason})")
        if self.settings.approval_mode == "readonly":
            return PolicyDecision(DENY, f"read-only mode ({decision.reason})")
        return decision
