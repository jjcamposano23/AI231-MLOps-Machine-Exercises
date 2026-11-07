"""Shared helpers for the ME3 MCP servers (all run on the Raspberry Pi).

Every server is a standalone process speaking MCP over streamable HTTP on localhost:
    asr    :8101  speech-to-text   (Deepgram nova-3, or any OpenAI-compatible /audio/transcriptions, e.g. SHARC DGX)
    tts    :8102  text-to-speech   (Deepgram Aura-2, or any OpenAI-compatible /audio/speech)
    assist :8103  real-world tools (time, weather, web search, timers/alarms/reminders, notes, messages)
    home   :8104  Pi actuators     (ME2 WS2812 strip, SSD1306 OLED, TM1637 display, music)
    agent  :8105  background agent (multi-step tasks: research, build a web page/game; serves results on :8080)
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ME3_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)

PORTS = {"asr": 8101, "tts": 8102, "assist": 8103, "home": 8104, "agent": 8105}


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=os.environ.get("MCP_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format=f"%(asctime)s {name}-mcp %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)      # keep-warm pings would flood the log
    return logging.getLogger(name)


def serve(server, name: str, json_response: bool = True) -> None:
    """Run an MCPServer over streamable HTTP on 127.0.0.1:<port> (path /mcp)."""
    port = int(os.environ.get(f"{name.upper()}_MCP_PORT", PORTS[name]))
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    print(f"[{name}-mcp] serving on http://{host}:{port}/mcp", file=sys.stderr, flush=True)
    server.run("streamable-http", host=host, port=port, streamable_http_path="/mcp",
               json_response=json_response, stateless_http=True, max_request_body_size=32 * 1024 * 1024)


class Warmth:
    """Keeps the provider connection hot while the user is talking.

    Measured on the Pi (tools/asr_gap_test.py): a Deepgram request after >=5 s idle takes ~1.35 s, after a ping
    <=1 s earlier ~0.65 s (Wi-Fi power save + server-side idle). warm() pings only if the connection has been
    idle for idle_s and no ping is in flight; real requests call before() so they never race a ping onto a
    second, cold connection."""

    def __init__(self, idle_s: float = 1.5):
        self.idle_s = idle_s
        self.last = 0.0
        self.busy = 0
        self._ping = asyncio.Lock()

    async def ping(self, fn) -> bool:
        if self.busy or self._ping.locked() or time.monotonic() - self.last < self.idle_s:
            return False
        async with self._ping:
            try:
                await fn()
            except Exception as e:  # noqa: BLE001
                logging.getLogger("warm").warning("warm ping failed: %r", e)
            self.last = time.monotonic()
        return True

    async def before(self) -> None:
        self.busy += 1
        async with self._ping:          # wait (<~0.3 s) for an in-flight ping, then reuse its connection
            pass

    def after(self) -> None:
        self.busy -= 1
        self.last = time.monotonic()


def tool(server, snapshot=None):
    """Drop-in for @server.tool() that also publishes each call to the web emulator (see emulator_bus):
    a tool_start event, then tool_result with args, result, duration and an optional state snapshot."""
    from mcp_servers import emulator_bus as bus
    sname = getattr(server, "name", "") or ""

    def deco(fn):
        name = fn.__name__
        quiet = name.startswith("internal_") or name in ("info", "warm")

        def done(t0, kwargs, result=None, error=None):
            if quiet:
                return
            snap = None
            if snapshot is not None:
                try:
                    snap = snapshot()
                except Exception:  # noqa: BLE001
                    pass
            bus.emit("tool_result", server=sname, tool=name, args=kwargs, result=result, error=error,
                     ms=round((time.monotonic() - t0) * 1000), state=snap)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                t0 = time.monotonic()
                if not quiet:
                    bus.emit("tool_start", server=sname, tool=name, args=kwargs)
                try:
                    r = await fn(*args, **kwargs)
                except Exception as e:
                    done(t0, kwargs, error=repr(e))
                    raise
                done(t0, kwargs, r)
                return r
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.monotonic()
                try:
                    r = fn(*args, **kwargs)
                except Exception as e:
                    done(t0, kwargs, error=repr(e))
                    raise
                done(t0, kwargs, r)
                return r
        return server.tool()(wrapper)
    return deco


def mark(stage: str, **fields) -> None:
    """Optional fdbench stage mark from inside an MCP server (no-op unless FDBENCH_MARKS is set)."""
    if not os.environ.get("FDBENCH_MARKS"):
        return
    try:
        sys.path.insert(0, str(ROOT / "fd-benchmark" / "adapters"))
        from fdbench_marks import mark as _mark
        _mark(stage, **fields)
    except Exception:  # noqa: BLE001
        pass


def make_http(timeout: float = 20.0):
    """httpx client that keeps its TLS connection open (default keep-alive is only 5 s, which made every utterance
    pay a fresh ~0.7 s handshake to the provider)."""
    import httpx
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0),
                             limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=600))


_warmers: set = set()


def keep_warm(client, url: str, every: float = 15.0) -> None:
    """Start (once) a background task that touches `url` every few seconds so the pooled connection stays warm.
    Call from inside a running event loop (e.g. at the top of a tool)."""
    import asyncio
    if url in _warmers:
        return
    _warmers.add(url)

    async def _loop():
        while True:
            try:
                await client.head(url)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(every)

    asyncio.get_running_loop().create_task(_loop())


def net_info() -> dict:
    """The Pi's LAN IP and Wi-Fi network name (shown on the OLED and in the web emulator; no traffic is sent)."""
    import socket
    import subprocess
    ip = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))                    # UDP connect: picks the outgoing interface, sends nothing
            ip = s.getsockname()[0]
    except OSError:
        pass
    ssid = None
    try:                                                  # NetworkManager connection on wlan0 (= Wi-Fi name)
        out = subprocess.run(["nmcli", "-t", "-g", "GENERAL.CONNECTION", "device", "show", "wlan0"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        ssid = out.removeprefix("netplan-wlan0-") or None
    except Exception:  # noqa: BLE001
        pass
    return {"ip": ip, "ssid": ssid}
