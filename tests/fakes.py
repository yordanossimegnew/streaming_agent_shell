"""A scripted chat model that streams like ChatAnthropic, so tests exercise token streaming offline."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr


def say(text: str = "", *calls: dict, stop_reason: str | None = None) -> AIMessage:
    """Build a scripted reply: some text plus optional tool calls."""
    return AIMessage(
        content=[{"type": "text", "text": text}] if text else [],
        tool_calls=list(calls),
        response_metadata={"stop_reason": stop_reason or ("tool_use" if calls else "end_turn")},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def cmd(id_: str, command: str, reason: str = "test") -> dict:
    return {"id": id_, "name": "run_command", "args": {"command": command, "reason": reason}, "type": "tool_call"}


def write(id_: str, path: str, content: str) -> dict:
    return {"id": id_, "name": "write_file", "args": {"path": path, "content": content}, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Pops one scripted AIMessage per call. Streams text word by word, then the tool calls."""

    _script: list[AIMessage] = PrivateAttr(default_factory=list)
    _calls: list[list] = PrivateAttr(default_factory=list)

    def __init__(self, script: list[AIMessage], **kwargs: Any):
        super().__init__(**kwargs)
        self._script = list(script)

    @property
    def calls(self) -> list[list]:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages) -> AIMessage:
        self._calls.append(list(messages))
        if not self._script:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return self._script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        msg = self._next(messages)
        text = "".join(b["text"] for b in msg.content if b.get("type") == "text")
        for piece in re.findall(r"\S+\s*|\s+", text):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=[{"type": "text", "text": piece, "index": 0}]))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=[],
            tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i + 1}
                for i, c in enumerate(msg.tool_calls)
            ],
            response_metadata=msg.response_metadata,
            usage_metadata=msg.usage_metadata,
        ))
