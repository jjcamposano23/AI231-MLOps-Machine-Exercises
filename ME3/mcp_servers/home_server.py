"""Home MCP server: drives the Pi's physical actuators (reused from ME2) and music playback.

Hardware (auto-detected; anything missing is simulated and reported as such):
  WS2812 strip  GPIO18 (rpi_ws281x, needs sudo -> run THIS server with sudo, the assistant stays unprivileged)
  SSD1306 OLED  I2C 0x3C (luma.oled)
  TM1637 4-dig  CLK=GPIO20, DIO=GPIO21 (lgpio)
  Music         Spotify Connect speaker Iris (librespot) via the Web API; else files in data/music (VLC/mpv)
Covers the ME2 intents: LIGHT_ON/OFF, BRIGHTNESS, COLOR, TEMPERATURE, TIMER display, PLAY_MUSIC, PAUSE, STOP,
NEXT, VOLUME_UP/DOWN.
Run:  python -m mcp_servers.home_server        (Pi with strip:  sudo -E .venv/bin/python -m mcp_servers.home_server)
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from mcp_servers.common import DATA, ROOT, get_logger, load_env, net_info, serve, tool

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
strip = oled = seg = leds = None
if os.environ.get("HOME_HARDWARE", "auto") != "off":
    try:
        if os.environ.get("LEDS", "on") == "off":
            raise RuntimeError("LEDS=off")
        from hardware.pi_leds import LedPanel
        leds = LedPanel(yellow=int(os.environ.get("LED_YELLOW", 27)), red=int(os.environ.get("LED_RED", 17)),
                        blue=int(os.environ.get("LED_BLUE", 22)))
        log.info("LED bulbs ready (yellow/red/blue)")
    except Exception as e:  # noqa: BLE001
        log.info("LED bulbs simulated (%s)", e)
    try:
        if os.environ.get("STRIP", "on") == "off":   # GPIO18 PWM is shared with the 3.5 mm jack: STRIP=off frees it
            raise RuntimeError("STRIP=off")
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
    return {"strip": strip is not None, "leds": leds is not None, "oled": oled is not None,
            "seven_segment": seg is not None}


def _apply_light() -> None:
    if leds is not None:
        try:
            leds.set(state["light_on"], state["color"], state["rgb"], state["brightness_pct"])
        except Exception as e:  # noqa: BLE001
            log.warning("leds: %s", e)
    if strip is None:
        return
    if not state["light_on"]:
        strip._fill(0, 0, 0)
        return
    k = state["brightness_pct"] / 100
    r, g, b = state["rgb"]
    strip._fill(r * k, g * k, b * k)


_banner = True       # until the first real message, the OLED shows "Iris ready" + the Pi's IP (refreshed)


def _boot_banner() -> None:
    import time as _t
    last = None
    for _ in range(720):                               # up to 1 h after start
        if not _banner:
            return
        n = net_info()
        line = n["ip"] or "no network"
        if line != last and oled is not None:
            try:
                oled.show("Iris ready", line)
            except Exception:  # noqa: BLE001
                pass
            last = line
        state["display"] = f"Iris ready | {line}"
        _t.sleep(5)


def _show(line1: str, line2: str = "") -> None:
    global _banner
    _banner = False
    state["display"] = f"{line1} | {line2}".strip(" |")
    if oled is not None:
        try:
            oled.show(line1, line2)
        except Exception as e:  # noqa: BLE001
            log.warning("oled: %s", e)


def _snapshot() -> dict:
    return {**state, "music": dict(state["music"]), "hardware": _hw()}


threading.Thread(target=_boot_banner, name="oled-banner", daemon=True).start()


server = MCPServer("home", instructions="Controls the room: lights (yellow, red and blue bulbs), colour, brightness, thermostat "
                                        "display, a small screen, and music.")


@tool(server, snapshot=_snapshot)
def set_lights(on: bool) -> dict:
    """Turn the room lights on or off."""
    with _lock:
        state["light_on"] = bool(on)
        _apply_light()
        _show("Lights", "ON" if on else "OFF")
    return {"ok": True, "light_on": state["light_on"], "hardware": strip is not None or leds is not None}


@tool(server, snapshot=_snapshot)
def set_light_color(color: str) -> dict:
    """Set the light colour by name (white, warm white, red, green, blue, yellow, orange, purple, pink, cyan)
    or hex like '#ff8800'. Turns the lights on. The room has three bulbs (yellow, red, blue); other colours are
    mixed from them (purple = red + blue, white = all three)."""
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
    return {"ok": True, "color": c, "hardware": strip is not None or leds is not None}


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
    return {"ok": True, "brightness_pct": p, "hardware": strip is not None or leds is not None}


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


# ---------------------------------------------------------------- music
# Spotify first (mcp_servers/spotify_player.py: the Pi is a Spotify Connect speaker called "Iris"); without it,
# local files in data/music played with VLC (rc socket) or mpv. data/music_playing is a flag file the assistant's
# MusicGate reads (while music plays, only speech that names Iris counts as a command).
MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", DATA / "music"))
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
PLAYING_FLAG = DATA / "music_playing"
PLAYING_FLAG.unlink(missing_ok=True)
_RC = "/tmp/me3-vlc.sock"
_player: subprocess.Popen | None = None
_playlist: list[Path] = []
_idx = 0
_spotify = None
_duck_restore: int | None = None
_INFO = ("track", "artist", "album", "art")


def _sp():
    """The Spotify controller, or None when Spotify is not set up (no client id / not signed in)."""
    global _spotify
    if _spotify is None:
        try:
            from mcp_servers import spotify_player
            if spotify_player.configured():
                _spotify = spotify_player.Spotify()
                log.info("Spotify ready (device %s)", spotify_player.DEVICE)
        except Exception as e:  # noqa: BLE001
            log.warning("spotify unavailable: %s", e)
    return _spotify


def _set_playing(playing: bool, paused: bool = False, **info) -> None:
    m = state["music"]
    m.update(playing=playing, paused=paused, **info)
    if not playing:
        for k in _INFO + ("uri", "from"):
            m[k] = None
    try:
        if playing and not paused:
            PLAYING_FLAG.write_text(m.get("track") or "music")
        else:
            PLAYING_FLAG.unlink(missing_ok=True)
    except OSError:
        pass


def _tracks() -> list[Path]:
    return sorted(p for p in MUSIC_DIR.rglob("*") if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"})


def _rc(cmd: str) -> bool:
    if _player is None or _player.poll() is not None or not hasattr(socket, "AF_UNIX"):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(_RC)
            s.sendall((cmd + "\n").encode())
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("vlc rc: %s", e)
        return False


def _start_local(track: Path) -> bool:
    global _player
    _stop_local()
    vol = int(state["music"]["volume"] * 2.56)            # VLC rc volume: 256 = 100 %
    if shutil.which("cvlc"):
        Path(_RC).unlink(missing_ok=True)
        _player = subprocess.Popen(["cvlc", "-q", "--no-video", "--play-and-exit", "--intf", "oldrc", "--rc-unix", _RC,
                                    "--rc-fake-tty", str(track)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def set_vol():                                     # VLC 3 has no --volume option: set it over the rc socket
            import time as _t
            for _ in range(20):
                _t.sleep(0.15)
                if Path(_RC).exists() and _rc(f"volume {vol}"):
                    return
        threading.Thread(target=set_vol, daemon=True).start()
        return True
    if shutil.which("mpv"):
        _player = subprocess.Popen(["mpv", "--no-video", "--really-quiet", f"--volume={state['music']['volume']}",
                                    str(track)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        return True
    return False


def _stop_local() -> None:
    global _player
    if _player is not None and _player.poll() is None:
        _player.terminate()
        try:
            _player.wait(timeout=2)                        # reap it (no zombie vlc)
        except subprocess.TimeoutExpired:
            _player.kill()
    _player = None


def _spotify_call(fn, *a):
    try:
        return fn(*a), None
    except Exception as e:  # noqa: BLE001
        log.warning("spotify: %s", e)
        return None, str(e)


def _on_spotify() -> bool:
    return _sp() is not None and state["music"].get("source") == "Spotify"


@tool(server, snapshot=_snapshot)
def play_music(query: str = "", kind: str = "track") -> dict:
    """Play music on the room speaker from Spotify. query = what to play, e.g. 'Bohemian Rhapsody Queen' (a song),
    'Taylor Swift' with kind='artist', 'chill lofi' with kind='playlist', an album name with kind='album'.
    kind is one of track, artist, album, playlist (for a mood or genre use playlist). Empty query resumes."""
    sp = _sp()
    if sp is not None:
        if not query.strip():
            _, err = _spotify_call(sp.resume)
            if err is None:
                np, _ = _spotify_call(sp.now_playing)
                _set_playing(True, source="Spotify", **{k: (np or {}).get(k) for k in _INFO})
                return {"ok": True, "source": "spotify", "playing": (np or {}).get("track")}
            query, kind = "feel good hits", "playlist"
        info, err = _spotify_call(sp.play, query, kind)
        if err is not None:
            return {"ok": False, "source": "spotify", "error": err}
        _stop_local()
        _set_playing(True, source="Spotify", **{k: info.get(k) for k in _INFO + ("from",)})
        _show("Playing", (info.get("track") or query)[:18])
        return {"ok": True, "source": "spotify", **{k: v for k, v in info.items() if k not in ("art", "uri")}}
    global _playlist, _idx
    tracks = _tracks()
    if not tracks:
        _set_playing(True, track=query or "radio")
        return {"ok": True, "playing": state["music"]["track"], "note": "no Spotify and no music files; simulated"}
    q = query.lower().strip()
    words = [w for w in re.findall(r"[a-z0-9']+", q) if w not in {"by", "the", "song", "music", "some", "play", "a"}]

    def score(t: Path) -> int:                             # file name words shared with the request
        name = t.stem.lower()
        return sum(w in name for w in words)
    best = max((score(t) for t in tracks), default=0)
    hits = [t for t in tracks if words and score(t) == best and best > 0] or tracks
    _playlist = hits if q else random.sample(tracks, len(tracks))
    _idx = 0
    real = _start_local(_playlist[0])
    _set_playing(True, source="local", track=_playlist[0].stem)
    _show("Playing", _playlist[0].stem[:18])
    return {"ok": True, "source": "local", "playing": _playlist[0].stem, "hardware": real}


@tool(server, snapshot=_snapshot)
def pause_music() -> dict:
    """Pause the music (call resume_music to continue)."""
    if _on_spotify():
        _, err = _spotify_call(_sp().pause)
        if err:
            return {"ok": False, "error": err}
    elif not _rc("pause") and _player is not None:
        _player.send_signal(19)                             # SIGSTOP (mpv fallback)
    _set_playing(state["music"]["playing"], paused=True)
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def resume_music() -> dict:
    """Resume paused music."""
    if _sp() is not None and state["music"].get("source") in ("Spotify", None):
        _, err = _spotify_call(_sp().resume)
        if err:
            return {"ok": False, "error": err}
        if state["music"].get("source") is None:
            state["music"]["source"] = "Spotify"
    elif not _rc("pause") and _player is not None:
        _player.send_signal(18)                             # SIGCONT
    _set_playing(True, paused=False)
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def stop_music() -> dict:
    """Stop the music completely."""
    if _on_spotify():
        _spotify_call(_sp().pause)
    _stop_local()
    _set_playing(False, source=None)
    return {"ok": True}


@tool(server, snapshot=_snapshot)
def next_song() -> dict:
    """Skip to the next song."""
    global _idx
    if _on_spotify():
        np, err = _spotify_call(_sp().next)
        if err:
            return {"ok": False, "error": err}
        _set_playing(True, **{k: (np or {}).get(k) for k in _INFO})
        return {"ok": True, "playing": (np or {}).get("track"), "artist": (np or {}).get("artist")}
    if not _playlist:
        return play_music("")
    _idx = (_idx + 1) % len(_playlist)
    _start_local(_playlist[_idx])
    _set_playing(True, track=_playlist[_idx].stem)
    return {"ok": True, "playing": _playlist[_idx].stem}


@tool(server, snapshot=_snapshot)
def whats_playing() -> dict:
    """What song is playing right now (title, artist, album)."""
    if _sp() is not None:
        np, _ = _spotify_call(_sp().now_playing)
        if np:
            return {k: v for k, v in np.items() if k != "art"}
    m = state["music"]
    return {"playing": m.get("playing"), "track": m.get("track"), "artist": m.get("artist")}


def _apply_volume(v: int) -> str | None:
    if _on_spotify():
        _, err = _spotify_call(_sp().volume, v)
        return err
    _rc(f"volume {int(v * 2.56)}")
    return None


@tool(server, snapshot=_snapshot)
def change_volume(delta_percent: int) -> dict:
    """Turn the music volume up (positive, e.g. 15) or down (negative, e.g. -15)."""
    v = max(0, min(100, state["music"]["volume"] + int(delta_percent)))
    state["music"]["volume"] = v
    err = _apply_volume(v)
    return {"ok": err is None, "volume": v, **({"error": err} if err else {})}


@tool(server)
def internal_duck(on: bool) -> dict:
    """(assistant only) Lower the music while Iris speaks, then restore it."""
    global _duck_restore
    m = state["music"]
    if on and (not m.get("playing") or m.get("paused")):
        return {"ok": True, "skipped": True}
    if on and _duck_restore is None:
        _duck_restore = m["volume"]
        _apply_volume(max(5, int(_duck_restore * float(os.environ.get("MUSIC_DUCK", 0.3)))))
    elif not on and _duck_restore is not None:
        _apply_volume(_duck_restore)
        _duck_restore = None
    return {"ok": True}


if __name__ == "__main__":
    serve(server, "home")
