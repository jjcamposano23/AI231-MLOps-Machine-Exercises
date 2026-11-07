"""Agent MCP server: the "thinker" for long, multi-step jobs, delegated by the voice "talker".

run_task(task) runs a tool-using agent loop (web search + a sandboxed workspace) with a stronger LLM and returns a
short spoken summary. Files it writes are served at http://<pi>:8080/<path>, so "build me a flappy bird game" ends
with a playable page anyone in the room can open.

Providers (AGENT_PROVIDER): anthropic (default, AGENT_MODEL=claude-sonnet-5-5) | openrouter (AGENT_MODEL=...).
Run:  python -m mcp_servers.agent_server
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import re
import socket
import threading
import time
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

from mcp_servers import emulator_bus
from mcp_servers.common import DATA, get_logger, load_env, serve, tool

load_env()
log = get_logger("agent")

PROVIDER = os.environ.get("AGENT_PROVIDER", "openrouter").lower()
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5" if PROVIDER == "anthropic" else "openai/gpt-oss-120b")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", 12))
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", 4000))   # OpenRouter reserves credit per max_tokens
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", DATA / "workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
WEB_PORT = int(os.environ.get("AGENT_WEB_PORT", 8080))

SYSTEM = f"""You are the background worker of a voice assistant running on a Raspberry Pi.
You get one task, finish it with your tools, then reply with a SHORT plain-text summary (1-3 sentences) that will
be read aloud: no markdown, no lists, no URLs except the page link if you built one.
- Use web_search for anything current (prices, flights, news, schedules). Give concrete numbers and names.
- To build something (web page, game, app), ALWAYS make it a browser app: write ONE complete, self-contained
  file named index.html (inline CSS and JavaScript, no CDN, no Python, no build step) into its own folder,
  e.g. flappy/index.html. Never write .py files for games or apps: the user opens the result in a browser. It will be served at http://<pi-address>:{WEB_PORT}/<folder>/.
- Keep files small and working (a game or page in under ~250 lines). Don't ask questions; make sensible choices."""

TOOLS = [
    {"name": "web_search", "description": "Search the web; returns an answer and top results.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "write_file", "description": "Write a text file inside the workspace (relative path).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "read_file", "description": "Read a text file from the workspace.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "list_files", "description": "List files in the workspace.",
     "input_schema": {"type": "object", "properties": {}}},
]

_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))


def _safe(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError("path escapes the workspace")
    return p


async def _run_tool(name: str, args: dict, written: list[str]) -> str:
    try:
        if name == "web_search":
            key = os.environ.get("TAVILY_API_KEY", "")
            r = await _http.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {key}"},
                                 json={"query": args["query"], "max_results": 5, "include_answer": True})
            d = r.json()
            return json.dumps({"answer": d.get("answer"), "results": [
                {"title": x.get("title"), "url": x.get("url"), "content": (x.get("content") or "")[:600]}
                for x in d.get("results", [])]})[:6000]
        if name == "write_file":
            p = _safe(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            written.append(str(p.relative_to(WORKSPACE)).replace("\\", "/"))
            return f"wrote {len(args['content'])} chars to {args['path']}"
        if name == "read_file":
            return _safe(args["path"]).read_text(encoding="utf-8")[:20000]
        if name == "list_files":
            return "\n".join(str(p.relative_to(WORKSPACE)).replace("\\", "/") for p in WORKSPACE.rglob("*")
                             if p.is_file())[:4000] or "(empty)"
        return f"unknown tool {name}"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


async def _loop_anthropic(task: str, written: list[str]) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": task}]
    for _ in range(MAX_STEPS):
        resp = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, tools=TOOLS,
                                            messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            return " ".join(b.text for b in resp.content if b.type == "text").strip()
        results = []
        for u in uses:
            out = await _run_tool(u.name, u.input, written)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
        messages.append({"role": "user", "content": results})
    return "I ran out of steps before finishing."


async def _loop_openrouter(task: str, written: list[str]) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(MAX_STEPS):
        r = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=MAX_TOKENS)
        m = r.choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return (m.content or "").strip()
        for c in m.tool_calls:
            out = await _run_tool(c.function.name, json.loads(c.function.arguments or "{}"), written)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return "I ran out of steps before finishing."


# ------------------------------------------------------------------ live progress for the web emulator
_RE_PATH = re.compile(r'"path"\s*:\s*"([^"]*)"')
_RE_CONTENT = re.compile(r'"content"\s*:\s*"')


def _partial_content(args_text: str) -> str:
    """Best-effort decode of a half-streamed write_file arguments JSON: the file text written so far."""
    m = _RE_CONTENT.search(args_text)
    if not m:
        return ""
    raw = args_text[m.end():]
    for cut in range(0, 8):                               # trim an unfinished escape (\" or é) at the end
        try:
            return json.loads('"' + raw[:len(raw) - cut] + '"')
        except json.JSONDecodeError:
            continue
    return raw.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t").replace("\\\\", "\\")


def _step(**kw) -> None:
    emulator_bus.emit("agent_step", **kw)


def _tool_detail(name: str, args: dict) -> str:
    if name == "web_search":
        return args.get("query", "")
    if name in ("write_file", "read_file"):
        return args.get("path", "")
    return ""


async def _loop_openrouter_stream(task: str, written: list[str]) -> str:
    """Same loop as _loop_openrouter, but streamed so the emulator can show the agent working: each step, each
    tool call, and the code of a file while it is being written (about once a second)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for n in range(1, MAX_STEPS + 1):
        _step(phase="thinking", step=n, text=f"Step {n}: deciding the next action")
        stream = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=MAX_TOKENS,
                                                      stream=True)
        content, calls, last_emit = [], {}, 0.0
        async for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d.content:
                content.append(d.content)
            for tc in d.tool_calls or []:
                c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    c["id"] = tc.id
                if tc.function and tc.function.name:
                    c["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    c["args"] += tc.function.arguments
            now = time.monotonic()
            if now - last_emit > 0.9 and calls:
                c = calls[max(calls)]
                if c["name"] == "write_file":
                    code = _partial_content(c["args"])
                    pm = _RE_PATH.search(c["args"])
                    _step(phase="writing", step=n, path=pm.group(1) if pm else "", chars=len(code), code=code[-3000:])
                    last_emit = now
        text = "".join(content).strip()
        if not calls:
            return text
        msg = {"role": "assistant", "content": text or None,
               "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                               "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                              for i, c in sorted(calls.items())]}
        messages.append(msg)
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            _step(phase="tool", step=n, tool=name, text=_tool_detail(name, args))
            out = await _run_tool(name, args, written)
            if name == "write_file" and "content" in args:
                _step(phase="wrote", step=n, path=args.get("path", ""), chars=len(args["content"]),
                      code=args["content"][-3000:])
                _emit_page(args.get("path", ""), task)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    return "I ran out of steps before finishing."


_emitted_pages: set[str] = set()


def _emit_page(path: str, task: str) -> None:
    """Send a written .html page to the emulator right away (live preview while the agent keeps working)."""
    if not path.endswith(".html"):
        return
    try:
        html = _safe(path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return
    if len(html) < 400_000:
        folder = path.split("/")[0]
        link = f"http://{_lan_ip()}:{WEB_PORT}/{folder}/" if "/" in path else f"http://{_lan_ip()}:{WEB_PORT}/{path}"
        emulator_bus.emit("agent_page", path=path, link=link, task=task, page_html=html)
        _emitted_pages.add(path)


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "localhost"


server = MCPServer("agent", instructions="Background agent for long multi-step tasks (research, building web pages "
                                         "or games). Slow: seconds to minutes.")


@tool(server)
async def run_task(task: str) -> dict:
    """Do a long, multi-step task end to end (research with web search, compare options, or build a web page/game)
    and return a short spoken summary plus any page link."""
    t0 = time.monotonic()
    written: list[str] = []
    log.info("task: %s", task)
    _emitted_pages.clear()
    _step(phase="start", text=f"Task received: {task}")
    try:
        if PROVIDER == "anthropic":
            summary = await _loop_anthropic(task, written)
        else:
            try:
                summary = await _loop_openrouter_stream(task, written)
            except Exception as e:  # noqa: BLE001
                if written:
                    raise
                log.warning("streaming agent failed (%r); retrying without streaming", e)
                _step(phase="thinking", text="Retrying without live progress")
                summary = await _loop_openrouter(task, written)
    except Exception as e:  # noqa: BLE001
        log.exception("agent failed")
        _step(phase="error", text=str(e)[:200])
        return {"ok": False, "summary": f"The background task failed: {e}"}
    pages = sorted({w.split("/")[0] for w in written if w.endswith(".html")})
    links = [f"http://{_lan_ip()}:{WEB_PORT}/{p}/" if not p.endswith(".html") else f"http://{_lan_ip()}:{WEB_PORT}/{p}"
             for p in pages]
    secs = round(time.monotonic() - t0, 1)
    log.info("done in %.1fs, files=%s", secs, written)
    _step(phase="done", text=f"Finished in {secs} s" + (f": {', '.join(written)}" if written else ""))
    for w in written:                           # the emulator renders the built page inline (iframe srcdoc)
        if w.endswith(".html") and w not in _emitted_pages:
            try:
                html = (WORKSPACE / w).read_text(encoding="utf-8")
                if len(html) < 400_000:
                    emulator_bus.emit("agent_page", path=w, link=links[0] if links else "",
                                      task=task, page_html=html)
            except Exception:  # noqa: BLE001
                pass
    return {"ok": True, "summary": summary, "files": written, "links": links, "seconds": secs}


def _start_web() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WORKSPACE))
    try:
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", WEB_PORT), handler)
    except OSError as e:
        log.warning("workspace web server not started: %s", e)
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log.info("workspace served at http://%s:%d/", _lan_ip(), WEB_PORT)


if __name__ == "__main__":
    _start_web()
    serve(server, "agent")
