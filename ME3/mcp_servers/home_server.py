"""Home MCP server: drives the Pi's physical actuators (reused from ME2) and music playback.

Hardware (auto-detected; anything missing is simulated and reported as such):
  WS2812 strip  GPIO18 (rpi_ws281x, needs sudo -> run THIS server with sudo, the assistant stays unprivileged)
  SSD1306 OLED  I2C 0x3C (luma.oled)
  TM1637 4-dig  CLK=GPIO20, DIO=GPIO21 (lgpio)
  Music         mpv, files from ME3/data/music or MUSIC_DIR
Covers the ME2 intents: LIGHT_ON/OFF, BRIGHTNESS, COLOR, TEMPERATURE, TIMER display, PLAY_MUSIC, PAUSE, STOP,
NEXT, VOLUME_UP/DOWN.
Run:  python -m mcp_servers.home_server        (Pi with strip:  sudo -E .venv/bin/python -m mcp_servers.home_server)
"""
from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from mcp_servers.common import DATA, ROOT, get_logger, load_env, serve, tool

load_env()
log = get_logger("home")
sys.path.insert(0, str(ROOT))

COLORS = {"white": (255, 255, 255), "warm white": (255, 170, 80), "red": (255, 0, 0), "green": (0, 255, 0),
          "blue": (0, 0, 255), "yellow": (255, 200, 0), "orange": (255, 100, 0), "purple": (150, 0, 255),
          "pink": (255, 60, 150), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}

state = {"light_on": False, "color": "white", "rgb": [255, 255, 255], "brightness_pct": 100,
         "thermostat_c": None, "display": "", "music": {"playing": False, "paused": False, "track": None,
                                                        "volume": int(os.environ.get("MUSIC_VOLUME", 60))}}
_lock = threading.Lock()

# ---------------------------------------------------------------- hardware (optional)
strip = oled = seg = None
if os.environ.get("HOME_HARDWARE", "auto") != "off":
    try:
        from hardware.pi_strip import StripPanel
        strip = StripPanel(pin=int(os.environ.get("STRIP_PIN", 18)),
                           used=int(os.environ.get("STRIP_USED", 16)))
        log.info("WS2812 strip ready")
    except Exception as e:  # noqa: BLE001
        log.info("strip simulated (%s)", e)
    try:
        from hardware.pi_oled import OLED
        oled = OLED()
        oled.show("Iris ME3", "ready")
        log.info("OLED ready")
    except Exception as e:  # noqa: BLE001
        log.info("OLED simulated (%s)", e)
    try:
        from hardware.pi_tm1637 import TM1637
        seg = TM1637(clk=int(os.environ.get("TM_CLK", 20)), dio=int(os.environ.get("TM_DIO", 21)))
        log.info("TM1637 ready")
    except Exception as e:  # noqa: BLE001
        log.info("TM1637 simulated (%s)", e)


def _hw() -> dict:
    return {"strip": strip is not None, "oled": oled is not None, "seven_segment": seg is not None}


def _apply_light() -> None:
    if strip is None:
        return
    if not state["light_on"]:
        strip._fill(0, 0, 0)
        return
    k = state["brightness_pct"] / 100
    r, g, b = state["rgb"]
    strip._fill(r * k, g * k, b * k)


def _show(line1: str, line2: str = "") -> None:
    state["display"] = f"{line1} | {line2}".strip(" |")
    if oled is not None:
        try:
            oled.show(line1, line2)
        except Exception as e:  # noqa: BLE001
            log.warning("oled: %s", e)


def _snapshot() -> dict:
    return {**state, "music": dict(state["music"]), "hardware": _hw()}


server = MCPServer("home", instructions="Controls the room: lights (LED strip), colour, brightness, thermostat "
                                        "display, a small screen, and music.")


@tool(server, snapshot=_snapshot)
def set_lights(on: bool) -> dict:
    """Turn the room lights (LED strip) on or off."""
    with _lock:
        state["light_on"] = bool(on)
        _apply_light()
        _show("Lights", "ON" if on else "OFF")
    return {"ok": True, "light_on": state["light_on"], "hardware": strip is not None}


@tool(server, snapshot=_snapshot)
def set_light_color(color: str) -> dict:
    """Set the light colour by name (white, warm white, red, green, blue, yellow, orange, purple, pink, cyan)
    or hex like '#ff8800'. Turns the lights on."""
    c = color.strip().lower()
    if c.startswith("#") and len(c) == 7:
        rgb = [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    elif c in COLORS:
        rgb = list(COLORS[c])
    else:
        return {"ok": False, "error": f"unknown colour {color!r}", "known": sorted(COLORS)}
    with _lock:
        state.update(light_on=True, color=c, rgb=rgb)
        _apply_light()
        _show("Light colour", c)
    return {"ok": True, "color": c, "hardware": strip is not None}


@tool(server, snapshot=_snapshot)
def set_brightness(percent: int) -> dict:
    """Set the light brightness from 1 to 100 percent. Turns the lights on."""
    p = max(1, min(100, int(percent)))
    with _lock:
        state.update(light_on=True, brightness_pct=p)
        _apply_light()
        _show("Brightness", f"{p}%")
        if seg is not None:
            seg.stop_timer(); seg.number(p)
    return {"ok": True, "brightness_pct": p, "hardware": strip is not None}


@tool(server, snapshot=_snapshot)
def set_thermostat(degrees_c: int) -> dict:
    """Set the air-conditioner / thermostat target temperature in Celsius (16-30)."""
    t = max(16, min(30, int(degrees_c)))
    with _lock:
        state["thermostat_c"] = t
        _show("Thermostat", f"{t} C")
        if seg is not None:
            seg.stop_timer(); seg.number(t)
    return {"ok": True, "thermostat_c": t, "hardware": seg is not None}


@tool(server, snapshot=_snapshot)
def show_countdown(seconds: int) -> dict:
    """Show a live countdown on the 4-digit display (use together with a timer)."""
    s = max(1, min(5999, int(seconds)))
    if seg is not None:
        seg.start_timer(s)
    _show("Timer", f"{s // 60}:{s % 60:02d}")
    return {"ok": True, "seconds": s, "hardware": seg is not None}


@tool(server, snapshot=_snapshot)
def show_message(line1: str, line2: str = "") -> dict:
    """Show up to two short lines of text on the small OLED screen."""
    _show(line1[:18], line2[:18])
    return {"ok": True, "hardware": oled is not None}


@tool(server, snapshot=_snapshot)
def get_home_state() -> dict:
    """Current state of lights, thermostat, display and music, plus which hardware is physically attached."""
    return {**state, "hardware": _hw()}


# ---------------------------------------------------------------- music (mpv with a JSON IPC socket)
MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", DATA / "music"))
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
_IPC = "/tmp/me3-mpv.sock"
_mpv: subprocess.Popen | None = None
_playlist: list[Path] = []
_idx = 0


def _tracks() -> list[Path]:
    return sorted(p for p in MUSIC_DIR.rglob("*") if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"})


def _mpv_cmd(*cmd) -> bool:
    if _mpv is None or _mpv.poll() is not None or not hasattr(socket, "AF_UNIX"):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(_IPC)
            s.sendall((json.dumps({"command": list(cmd)}) + "\n").encode())
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("mpv ipc: %s", e)
        return False


def _start(track: Path) -> bool:
    global _mpv
    _stop()
    if shutil.which("mpv") is None:
        return False
    _mpv = subprocess.Popen(["mpv", "--no-video", "--really-quiet", f"--volume={state['music']['volume']}",
                             f"--input-ipc-server={_IPC}", str(track)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _stop() -> None:
    global _mpv
    if _mpv is not None and _mpv.poll() is None:
        _mpv.terminate()
    _mpv = None


@tool(server, snapshot=_snapshot)
def play_music(query: str = "") -> dict:
    """Play music. Optional query matches a song/artist/genre in the music folder; empty plays a random song."""
    global _playlist, _idx
    tracks = _tracks()
    if not tracks:
        state["music"].update(playing=True, paused=False, track=query or "radio")
        return {"ok": True, "playing": state["music"]["track"], "note": "no music files on the Pi; simulated"}
    q = query.lower().strip()
    hits = [t for t in tracks if q and q in t.stem.lower()] or tracks
    _playlist = hits if q else random.sample(tracks, len(tracks))
    _idx = 0
    real = _start(_playlist[0])
    state["music"].update(playing=True, paused=False, track=_playlist[0].stem)
    _show("Playing", _playlist[0].stem[:18])
    return {"ok": True, "playing": _playlist[0].stem, "hardware": real}


@tool(server, snapshot=_snapshot)
def pause_music() -> dict:
    """Pause the music (call resume_music to continue)."""
    _mpv_cmd("set_property", "pause", True)
    state["music"]["paused"] = True
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def resume_music() -> dict:
    """Resume paused music."""
    _mpv_cmd("set_property", "pause", False)
    state["music"]["paused"] = False
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def stop_music() -> dict:
    """Stop the music completely."""
    _stop()
    state["music"].update(playing=False, paused=False, track=None)
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def next_song() -> dict:
    """Skip to the next song."""
    global _idx
    if not _playlist:
        return play_music("")
    _idx = (_idx + 1) % len(_playlist)
    _start(_playlist[_idx])
    state["music"].update(playing=True, paused=False, track=_playlist[_idx].stem)
    return {"ok": True, "playing": _playlist[_idx].stem}


@tool(server, snapshot=_snapshot)
def change_volume(delta_percent: int) -> dict:
    """Turn the music volume up (positive, e.g. 15) or down (negative, e.g. -15)."""
    v = max(0, min(100, state["music"]["volume"] + int(delta_percent)))
    state["music"]["volume"] = v
    _mpv_cmd("set_property", "volume", v)
    return {"ok": True, "volume": v}


if __name__ == "__main__":
    serve(server, "home")
