"""Terminal-style Streamlit UI for the Streaming Agent Shell.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
EXAMPLES = json.loads((Path(__file__).resolve().parents[1] / "data" / "examples.json").read_text())

st.set_page_config(page_title="Agent Shell", page_icon="🖥️", layout="wide")
st.markdown("""<style>
.term {background:#0d1117;color:#c9d1d9;border-radius:10px;font:13px/1.5 'JetBrains Mono','Cascadia Code',Consolas,monospace;
       height:560px;overflow-y:auto;display:flex;flex-direction:column-reverse;border:1px solid #30363d}
.term-inner {padding:14px 18px}
.bar {background:#161b22;border-radius:10px 10px 0 0;padding:8px 14px;color:#8b949e;font:12px monospace;
      border:1px solid #30363d;border-bottom:none;margin-bottom:-10px;position:relative;z-index:1}
.dot {display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}
.user {color:#58a6ff;font-weight:bold;margin-top:12px}
.say {color:#e6edf3;white-space:pre-wrap;margin:4px 0}
.cmd {color:#7ee787;margin-top:8px;font-weight:bold}
.out {color:#a5b1bd;white-space:pre-wrap}
.ok {color:#3fb950;font-size:12px} .bad {color:#f85149;font-size:12px}
.badge {font-size:10px;padding:1px 6px;border-radius:8px;margin-left:8px;font-weight:normal;vertical-align:middle}
.allow {background:#1f6f3a;color:#d2f8dc} .ask {background:#8a6a12;color:#fff3c4} .deny {background:#8e1f1f;color:#ffd7d5}
.wait {color:#e3b341;border:1px dashed #e3b341;padding:6px 10px;border-radius:6px;margin:8px 0}
.cursor {display:inline-block;width:8px;height:15px;background:#c9d1d9;animation:b 1s step-end infinite;vertical-align:text-bottom}
@keyframes b {50% {opacity:0}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- API helpers
def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def stream(url: str):
    with requests.get(f"{API_URL}{url}", stream=True, timeout=(5, 300)) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


# ----------------------------------------------------------------------------- rendering
def render_terminal(events: list[dict], live: bool = False) -> str:
    parts, saying = [], ""
    asked: set[str] = set()  # commands that waited for approval

    def flush():
        nonlocal saying
        if saying.strip():
            parts.append(f'<div class="say">{html.escape(saying.strip())}</div>')
        saying = ""

    for e in events:
        kind = e["type"]
        if kind == "token":
            saying += e["text"]
            continue
        flush()
        if kind == "user":
            parts.append(f'<div class="user">❯ {html.escape(e["text"])}</div>')
        elif kind == "approval":
            values = list(e["decisions"].values())
            yes, no = values.count("approve"), values.count("reject")
            summary = " · ".join(s for s in (f"✓ approved {yes}" if yes else "", f"✗ rejected {no}" if no else "") if s)
            parts.append(f'<div class="{"ok" if not no else "bad"}">{summary} command{"s" if len(values) > 1 else ""}</div>')
        elif kind == "tool_call":
            args = e["args"]
            target = args.get("command") or f"write_file {args.get('path')}"
            label = {"allow": "auto", "ask": "needs approval", "deny": "refused"}[e["effective"]]
            if e["effective"] == "ask":
                asked.add(e["id"])
            parts.append(f'<div class="cmd">$ {html.escape(target)}'
                         f'<span class="badge {e["effective"]}">{label}</span></div>')
        elif kind == "command_start" and e["call_id"] in asked:
            parts.append(f'<div class="cmd">$ {html.escape(e["command"])}'
                         f'<span class="badge allow">approved</span></div>')
        elif kind == "output":
            parts.append(f'<div class="out">{html.escape(e["line"]) or "&nbsp;"}</div>')
        elif kind == "command_done":
            ok = e["exit_code"] == 0
            status = "timed out" if e["timed_out"] else f"exit {e['exit_code']}"
            parts.append(f'<div class="{"ok" if ok else "bad"}">↳ {status} · {e["duration"]}s</div>')
        elif kind == "refused":
            parts.append(f'<div class="bad">✗ {html.escape(e["reason"])}</div>')
        elif kind == "approval_required":
            n = len(e["pending"])
            parts.append(f'<div class="wait">⏸ waiting for your approval ({n} command{"s" if n > 1 else ""})</div>')
        elif kind == "error":
            parts.append(f'<div class="bad">error: {html.escape(e["message"])}</div>')
    flush()
    if live:
        tail = f'<div class="say">{html.escape(saying)}<span class="cursor"></span></div>' if saying else '<span class="cursor"></span>'
        parts.append(tail)
    title = st.session_state.get("title") or "new session"
    return (f'<div class="bar"><span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e"></span>'
            f'<span class="dot" style="background:#27c93f"></span>&nbsp; agent@sandbox: ~ — {html.escape(title[:60])}</div>'
            f'<div class="term"><div class="term-inner">{"".join(parts) or "<span class=out>Type a task below to start.</span>"}</div></div>')


def run_stream(url: str, screen) -> None:
    last = 0.0
    for event in stream(url):
        st.session_state.events.append(event)
        if event["type"] != "token" or time.monotonic() - last > 0.08:
            screen.markdown(render_terminal(st.session_state.events, live=True), unsafe_allow_html=True)
            last = time.monotonic()
    st.rerun()


# ----------------------------------------------------------------------------- state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.events = []

health = api("GET", "/health")
with st.sidebar:
    st.title("🖥️ Agent Shell")
    st.caption("LangGraph agent in a sandboxed bash, streamed live")
    if health:
        st.success("API connected", icon="✅")
        st.markdown(f"**Model:** `{health['model']}`  \n**Approval mode:** `{health['approval_mode']}`  \n"
                    f"**Sandbox:** `{Path(health['sandbox']).name}/`")
    else:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")
    if st.button("➕ New session", width="stretch", disabled=not health):
        st.session_state.session_id = None
        st.session_state.events = []
        st.session_state.title = ""
        st.rerun()
    sessions = (api("GET", "/sessions") or []) if health else []
    if sessions:
        labels = {f"{s['title'] or '(empty)'} · {s['status']}": s["session_id"] for s in sessions}
        current = next((k for k, v in labels.items() if v == st.session_state.session_id), None)
        picked = st.selectbox("Sessions", list(labels), index=list(labels).index(current) if current else 0)
        if labels[picked] != st.session_state.session_id and st.button("Open session", width="stretch"):
            detail = api("GET", f"/sessions/{labels[picked]}")
            st.session_state.session_id = labels[picked]
            st.session_state.events = detail["events"] if detail else []
            st.session_state.title = detail["title"] if detail else ""
            st.rerun()
    st.divider()
    st.markdown("**Policy tester**")
    probe = st.text_input("Command", placeholder="rm -rf build", label_visibility="collapsed")
    if probe and health:
        d = api("GET", "/policy/check", params={"command": probe})
        if d:
            icon = {"allow": "🟢", "ask": "🟡", "deny": "🔴"}[d["action"]]
            st.markdown(f"{icon} **{d['action']}**: {d['reason']}")

# ----------------------------------------------------------------------------- main
screen = st.empty()
screen.markdown(render_terminal(st.session_state.events), unsafe_allow_html=True)

detail = api("GET", f"/sessions/{st.session_state.session_id}") if st.session_state.session_id else None
if detail and detail["status"] == "awaiting_approval":
    st.markdown("#### ⏸ Approve these commands?")
    choices = {}
    for item in detail["pending"]:
        args = item["args"]
        target = args.get("command") or f"write_file {args.get('path')}"
        with st.container(border=True):
            c1, c2 = st.columns([5, 2])
            c1.code(target, language="bash")
            why = f"Why: {args['reason']} · " if args.get("reason") else ""
            c1.caption(f"{why}Policy: {item['reason']} · risk {item['risk']}")
            choices[item["id"]] = c2.radio("Decision", ["approve", "reject"], key=f"d-{item['id']}",
                                           horizontal=True, label_visibility="collapsed")
    if st.button("Submit decisions", type="primary"):
        started = api("POST", f"/sessions/{st.session_state.session_id}/approval", json={"decisions": choices})
        if started:
            run_stream(started["stream_url"], screen)

busy = bool(detail and detail["status"] != "idle")
with st.expander("Example tasks"):
    for ex in EXAMPLES:
        if st.button(ex["name"], key=f"ex-{ex['name']}", disabled=busy or not health):
            st.session_state.queued = ex["task"]
            st.rerun()

prompt = st.chat_input("Ask the agent to do something in the sandbox…", disabled=busy or not health)
prompt = prompt or st.session_state.pop("queued", None)
if prompt:
    if not st.session_state.session_id:
        st.session_state.session_id = api("POST", "/sessions")["session_id"]
        st.session_state.title = prompt
    started = api("POST", f"/sessions/{st.session_state.session_id}/messages", json={"content": prompt})
    if started:
        run_stream(started["stream_url"], screen)
    else:
        st.error("Could not start the turn (is another one still running?).")
