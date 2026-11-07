"""A persistent MCP client connection used by the custom STT/TTS services and the background-agent tool,
plus auto-start of the local MCP server processes."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from assistant.config import ROOT


class McpLink:
    """One long-lived MCP session (the SDK session is task-bound, so a dedicated task owns it)."""

    def __init__(self, url: str, name: str = ""):
        self.url = url
        self.name = name or url
        self._session: ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            ready = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._own(ready), name=f"mcp-{self.name}")
            await ready

    async def _own(self, ready: asyncio.Future) -> None:
        try:
            async with streamable_http_client(self.url, max_sse_event_size=64 * 1024 * 1024) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    if not ready.done():
                        ready.set_result(True)
                    await self._stop.wait()
        except BaseException as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(str(e)))
            else:
                logger.warning(f"MCP link {self.name} dropped: {e!r}")
        finally:
            self._session = None

    async def call(self, tool: str, args: dict | None = None, timeout: float = 30.0) -> dict:
        if self._session is None:
            await self.start()
        res = await self._session.call_tool(tool, args or {}, read_timeout_seconds=timeout)
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            text = " ".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
            raise RuntimeError(f"{self.name}.{tool} failed: {text}")
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        for c in getattr(res, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                try:
                    return json.loads(t)
                except json.JSONDecodeError:
                    return {"text": t}
        return {}

    async def stream(self, tool: str, args: dict, timeout: float = 30.0):
        """Call a tool that streams data as progress notifications; yields each notification's message, in order,
        as it arrives. The tool's final result is ignored."""
        if self._session is None:
            await self.start()
        q: asyncio.Queue = asyncio.Queue()

        async def on_progress(progress, total, message):
            if message:
                q.put_nowait(message)

        call = asyncio.create_task(self._session.call_tool(tool, args, read_timeout_seconds=timeout,
                                                           progress_callback=on_progress))
        get = None
        try:
            while True:
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, call}, return_when=asyncio.FIRST_COMPLETED)
                if get in done:
                    yield get.result()
                    continue
                get.cancel()
                res = call.result()
                while not q.empty():
                    yield q.get_nowait()
                if getattr(res, "is_error", False) or getattr(res, "isError", False):
                    text = " ".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
                    raise RuntimeError(f"{self.name}.{tool} failed: {text}")
                return
        finally:                                           # interrupted (barge-in) or done: leave no task behind
            for t in (get, call):
                if t is not None and not t.done():
                    t.cancel()

    async def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 3)
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------------------------------------ server processes
SERVERS = {"asr": "mcp_servers.asr_server", "tts": "mcp_servers.tts_server", "assist": "mcp_servers.assist_server",
           "home": "mcp_servers.home_server", "agent": "mcp_servers.agent_server"}
_children: list[subprocess.Popen] = []


def _port_open(url: str) -> bool:
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 80), timeout=0.3):
            return True
    except OSError:
        return False


def ensure_servers(urls: dict[str, str], wait_s: float = 25.0) -> None:
    """Start any local MCP server that is not already listening (e.g. not run as a systemd service)."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    started = []
    for name, url in urls.items():
        host = urlparse(url).hostname
        if _port_open(url) or host not in {"127.0.0.1", "localhost"} or name not in SERVERS:
            continue
        out = open(log_dir / f"mcp_{name}.log", "ab")
        p = subprocess.Popen([sys.executable, "-m", SERVERS[name]], cwd=str(ROOT), stdout=out, stderr=out,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        _children.append(p)
        started.append((name, url, p))
        logger.info(f"started {name} MCP server (pid {p.pid})")
    deadline = time.monotonic() + wait_s
    for name, url, p in started:
        while not _port_open(url):
            if p.poll() is not None:
                raise RuntimeError(f"{name} MCP server exited; see logs/mcp_{name}.log")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} MCP server did not start in {wait_s:.0f}s; see logs/mcp_{name}.log")
            time.sleep(0.2)


def stop_children() -> None:
    for p in _children:
        if p.poll() is None:
            p.terminate()
