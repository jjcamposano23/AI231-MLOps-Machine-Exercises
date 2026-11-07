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
- Keep files small and working. Don't ask questions; make sensible choices."""

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
        resp = await client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, tools=TOOLS,
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
        r = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=8000)
        m = r.choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return (m.content or "").strip()
        for c in m.tool_calls:
            out = await _run_tool(c.function.name, json.loads(c.function.arguments or "{}"), written)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return "I ran out of steps before finishing."


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
    loop = _loop_anthropic if PROVIDER == "anthropic" else _loop_openrouter
    try:
        summary = await loop(task, written)
    except Exception as e:  # noqa: BLE001
        log.exception("agent failed")
        return {"ok": False, "summary": f"The background task failed: {e}"}
    pages = sorted({w.split("/")[0] for w in written if w.endswith(".html")})
    links = [f"http://{_lan_ip()}:{WEB_PORT}/{p}/" if not p.endswith(".html") else f"http://{_lan_ip()}:{WEB_PORT}/{p}"
             for p in pages]
    secs = round(time.monotonic() - t0, 1)
    log.info("done in %.1fs, files=%s", secs, written)
    for w in written:                           # the emulator renders the built page inline (iframe srcdoc)
        if w.endswith(".html"):
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
