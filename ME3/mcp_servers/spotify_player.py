"""Spotify playback for the home MCP server: Web API control of a Spotify Connect speaker on the Pi.

The Pi runs librespot (from raspotify) as a user service named "Iris" (scripts/iris-spotify.service), which
plays through PipeWire to the Pi's speaker. This module finds that device and starts / pauses / skips music on it
with the Spotify Web API (spotipy, PKCE sign-in: only a client ID, no client secret).

Setup:  SPOTIFY_CLIENT_ID in .env, then once:  python -m tools.spotify_login   (token cached in data/)
Needs Spotify Premium (Spotify only allows playback control for Premium accounts).
"""
from __future__ import annotations

import os
import time

from mcp_servers.common import DATA, get_logger

log = get_logger("spotify")
SCOPES = "user-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing"
CACHE = DATA / "spotify_token.json"
REDIRECT = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
DEVICE = os.environ.get("SPOTIFY_DEVICE", "Iris")
KINDS = ("track", "artist", "album", "playlist")


def configured() -> bool:
    return bool(os.environ.get("SPOTIFY_CLIENT_ID")) and CACHE.exists() and os.environ.get("SPOTIFY", "on") != "off"


def auth_manager():
    from spotipy.cache_handler import CacheFileHandler
    from spotipy.oauth2 import SpotifyPKCE
    return SpotifyPKCE(client_id=os.environ["SPOTIFY_CLIENT_ID"], redirect_uri=REDIRECT, scope=SCOPES,
                       cache_handler=CacheFileHandler(cache_path=str(CACHE)), open_browser=False)


class SpotifyError(RuntimeError):
    pass


class Spotify:
    def __init__(self):
        import spotipy
        self.sp = spotipy.Spotify(auth_manager=auth_manager(), requests_timeout=6, retries=1)
        self._dev: str | None = None

    # ------------------------------------------------------------------ device
    def device_id(self, refresh: bool = False) -> str:
        if self._dev and not refresh:
            return self._dev
        devs = self.sp.devices().get("devices", [])
        mine = [d for d in devs if d["name"].lower() == DEVICE.lower()]
        if not mine:
            names = ", ".join(d["name"] for d in devs) or "none"
            raise SpotifyError(f"the Spotify speaker '{DEVICE}' is not online (devices seen: {names}); open Spotify "
                               f"on a phone on the same Wi-Fi and pick '{DEVICE}' once")
        self._dev = mine[0]["id"]
        return self._dev

    def _call(self, fn, **kw):
        """Run a player call on our device; retry once with a fresh device id (it changes when librespot restarts)."""
        import spotipy
        for attempt in range(2):
            try:
                return fn(device_id=self.device_id(refresh=attempt > 0), **kw)
            except spotipy.SpotifyException as e:
                if attempt or e.http_status not in (404, 502):
                    raise SpotifyError(f"Spotify: {e.msg}") from e

    # ------------------------------------------------------------------ playback
    def play(self, query: str, kind: str = "track") -> dict:
        kind = kind if kind in KINDS else "track"
        res = self.sp.search(q=query, type=kind, limit=5, market="from_token")
        items = [i for i in (res.get(f"{kind}s", {}).get("items") or []) if i]
        if not items:
            raise SpotifyError(f"nothing on Spotify for {kind} {query!r}")
        it = items[0]
        if kind == "track":
            self._call(self.sp.start_playback, uris=[it["uri"]])
        else:
            self._call(self.sp.start_playback, context_uri=it["uri"])
        try:
            self.sp.shuffle(kind == "playlist", device_id=self.device_id())
        except Exception:  # noqa: BLE001
            pass
        info = {"kind": kind, "name": it["name"]}
        if kind == "track":
            info.update(_track_info(it))
        else:
            time.sleep(1.0)                              # let the context start, then report the actual song
            info.update(self.now_playing() or {})
            info["from"] = f"{kind} {it['name']}"
        return info

    def now_playing(self) -> dict | None:
        cur = self.sp.current_playback(market="from_token")
        if not cur or not cur.get("item"):
            return None
        d = _track_info(cur["item"])
        d.update(is_playing=cur.get("is_playing", False), volume=(cur.get("device") or {}).get("volume_percent"))
        return d

    def pause(self):
        self._call(self.sp.pause_playback)

    def resume(self):
        self._call(self.sp.start_playback)

    def next(self):
        self._call(self.sp.next_track)
        time.sleep(0.8)
        return self.now_playing()

    def volume(self, pct: int):
        self._call(self.sp.volume, volume_percent=max(0, min(100, int(pct))))


def _track_info(t: dict) -> dict:
    imgs = (t.get("album") or {}).get("images") or []
    return {"track": t["name"], "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "album": (t.get("album") or {}).get("name"), "art": imgs[-2]["url"] if len(imgs) > 1 else
            (imgs[0]["url"] if imgs else None), "uri": t.get("uri")}
