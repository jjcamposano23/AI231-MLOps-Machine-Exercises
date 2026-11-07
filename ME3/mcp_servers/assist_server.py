"""Assist MCP server: everyday real-world tools for the voice agent.

Tools: get_time, get_weather (Open-Meteo, no key), web_search (Tavily), set_timer, set_alarm, create_reminder,
list_reminders, cancel_reminder, save_note, read_notes, send_message.
Timers/alarms/reminders fire inside this server; the assistant polls internal_pop_due() once a second and speaks them.
Run:  python -m mcp_servers.assist_server
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from mcp.server.mcpserver import MCPServer

from mcp_servers import emulator_bus
from mcp_servers.common import DATA, get_logger, load_env, serve, tool

load_env()
log = get_logger("assist")

TZ = os.environ.get("LOCAL_TZ", "Asia/Manila")
HOME_CITY = os.environ.get("HOME_CITY", "Quezon City")
_http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

NOTES = DATA / "notes.json"
OUTBOX = DATA / "outbox.json"

server = MCPServer("assist", instructions="Everyday tools: time, weather, web search, timers, alarms, reminders, "
                                          "notes and messages.")

# id -> {"id","kind","label","due" (epoch s),"created"}
_reminders: dict[str, dict] = {}
_due: list[dict] = []


def _now() -> datetime:
    return datetime.now(ZoneInfo(TZ))


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _save(path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


async def _fire_later(rid: str) -> None:
    item = _reminders.get(rid)
    if not item:
        return
    await asyncio.sleep(max(0.0, item["due"] - time.time()))
    if _reminders.pop(rid, None) is not None:
        _due.append(item)
        emulator_bus.emit("reminder_fired", item=item, state=_snapshot())
        log.info("due: %s", item)


def _snapshot() -> dict:
    """What the emulator shows: pending timers/alarms/reminders (with epoch due time), notes, sent messages."""
    return {"pending": sorted(({k: i[k] for k in ("id", "kind", "label", "due", "due_local")}
                               for i in _reminders.values()), key=lambda i: i["due"]),
            "notes": _load(NOTES, [])[-6:], "outbox": _load(OUTBOX, [])[-6:]}


def _schedule(kind: str, label: str, due_epoch: float) -> dict:
    rid = uuid.uuid4().hex[:6]
    item = {"id": rid, "kind": kind, "label": label, "due": due_epoch,
            "due_local": datetime.fromtimestamp(due_epoch, ZoneInfo(TZ)).strftime("%I:%M:%S %p").lstrip("0")}
    _reminders[rid] = item
    asyncio.get_running_loop().create_task(_fire_later(rid))
    return item


@tool(server, snapshot=_snapshot)
def get_time(timezone: str = "") -> dict:
    """Current local date and time. Pass an IANA timezone (e.g. 'Asia/Dubai') for another place."""
    tz = timezone or TZ
    try:
        now = datetime.now(ZoneInfo(tz))
    except Exception:  # noqa: BLE001
        return {"error": f"unknown timezone {tz!r}; use an IANA name like 'Asia/Tokyo'"}
    return {"timezone": tz, "time": now.strftime("%I:%M %p").lstrip("0"), "date": now.strftime("%A, %B %d, %Y")}


_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        80: "rain showers", 81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
        96: "thunderstorm with hail", 99: "thunderstorm with hail"}


# Geocoding from the Pi is slow (~2.7 s) and sometimes stalls in the TLS handshake, so common places are built in
# and every other lookup is cached; HTTP calls use short timeouts with retries.
_GEO: dict[str, dict] = {k: {"name": n, "country": "Philippines", "latitude": la, "longitude": lo} for k, n, la, lo in [
    ("quezon city", "Quezon City", 14.6760, 121.0437), ("qc", "Quezon City", 14.6760, 121.0437),
    ("diliman", "Diliman, Quezon City", 14.6538, 121.0685), ("up diliman", "Diliman, Quezon City", 14.6538, 121.0685),
    ("manila", "Manila", 14.5995, 120.9842), ("metro manila", "Manila", 14.5995, 120.9842),
    ("makati", "Makati", 14.5547, 121.0244), ("pasig", "Pasig", 14.5764, 121.0851),
    ("taguig", "Taguig", 14.5176, 121.0509), ("marikina", "Marikina", 14.6507, 121.1029),
    ("caloocan", "Caloocan", 14.6507, 120.9667), ("baguio", "Baguio", 16.4023, 120.5960),
    ("cebu", "Cebu City", 10.3157, 123.8854), ("cebu city", "Cebu City", 10.3157, 123.8854),
    ("davao", "Davao City", 7.1907, 125.4553), ("davao city", "Davao City", 7.1907, 125.4553),
    ("iloilo", "Iloilo City", 10.7202, 122.5621), ("cagayan de oro", "Cagayan de Oro", 8.4542, 124.6319),
    ("tagaytay", "Tagaytay", 14.1153, 120.9621), ("los banos", "Los Baños", 14.1699, 121.2441)]}


async def _get_json(url: str, params: dict, tries: int = 3) -> dict:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = await _http.get(url, params=params, timeout=httpx.Timeout(6.0, connect=3.0))
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            await asyncio.sleep(0.2 * (i + 1))
    raise last  # type: ignore[misc]


async def _geocode(city: str) -> dict | None:
    key = " ".join(city.lower().replace(",", " ").replace("philippines", "").split())
    if key not in _GEO:
        res = (await _get_json("https://geocoding-api.open-meteo.com/v1/search",
                               {"name": city, "count": 1})).get("results") or []
        if not res:
            return None
        _GEO[key] = res[0]
    return _GEO[key]


@tool(server, snapshot=_snapshot)
async def get_weather(city: str = "") -> dict:
    """Current weather and today's forecast for a city (default: the user's home city)."""
    city = city or HOME_CITY
    try:
        p = await _geocode(city)
        if p is None:
            return {"error": f"could not find {city!r}"}
        d = await _get_json("https://api.open-meteo.com/v1/forecast", {
            "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 1})
    except Exception as e:  # noqa: BLE001
        log.warning("weather failed: %r", e)
        return {"error": "the weather service is not responding right now; tell the user and offer to try again"}
    c, day = d["current"], d["daily"]
    return {"place": f"{p['name']}, {p.get('country', '')}", "now": _WMO.get(c["weather_code"], "unknown"),
            "temp_c": c["temperature_2m"], "feels_like_c": c["apparent_temperature"],
            "humidity_pct": c["relative_humidity_2m"], "wind_kmh": c["wind_speed_10m"],
            "today_high_c": day["temperature_2m_max"][0], "today_low_c": day["temperature_2m_min"][0],
            "rain_chance_pct": day["precipitation_probability_max"][0]}


@tool(server, snapshot=_snapshot)
async def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web for current information (news, prices, flights, schedules, facts) and get a short answer."""
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return {"error": "TAVILY_API_KEY is not set"}
    r = await _http.post("https://api.tavily.com/search",
                         headers={"Authorization": f"Bearer {key}"},
                         json={"query": query, "max_results": max(1, min(int(max_results), 5)),
                               "include_answer": True, "search_depth": "basic"})
    r.raise_for_status()
    d = r.json()
    return {"answer": d.get("answer"),
            "results": [{"title": x.get("title"), "url": x.get("url"), "snippet": (x.get("content") or "")[:300]}
                        for x in d.get("results", [])]}


@tool(server, snapshot=_snapshot)
async def set_timer(seconds: int, label: str = "timer") -> dict:
    """Start a countdown timer. The assistant announces it when it finishes."""
    item = _schedule("timer", label, time.time() + max(1, int(seconds)))
    return {"ok": True, "id": item["id"], "ends_at": item["due_local"], "seconds": int(seconds)}


@tool(server, snapshot=_snapshot)
async def set_alarm(time_hhmm: str, label: str = "alarm") -> dict:
    """Set an alarm for a clock time today or tomorrow, 24-hour 'HH:MM' (e.g. '06:30', '21:00')."""
    try:
        hh, mm = (int(x) for x in time_hhmm.strip().split(":"))
    except Exception:  # noqa: BLE001
        return {"error": "use 24-hour HH:MM, e.g. 06:30"}
    now = _now()
    due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    item = _schedule("alarm", label, due.timestamp())
    return {"ok": True, "id": item["id"], "rings_at": due.strftime("%A %I:%M %p")}


@tool(server, snapshot=_snapshot)
async def create_reminder(task: str, in_minutes: float) -> dict:
    """Remind the user about a task after a number of minutes (fractions allowed, e.g. 0.5)."""
    item = _schedule("reminder", task, time.time() + max(0.1, float(in_minutes)) * 60)
    return {"ok": True, "id": item["id"], "at": item["due_local"], "task": task}


@tool(server, snapshot=_snapshot)
def list_reminders() -> dict:
    """List the pending timers, alarms and reminders."""
    now = time.time()
    return {"pending": [{"id": i["id"], "kind": i["kind"], "label": i["label"], "at": i["due_local"],
                         "seconds_left": round(i["due"] - now)} for i in sorted(_reminders.values(),
                                                                                key=lambda x: x["due"])]}


@tool(server, snapshot=_snapshot)
def cancel_reminder(id_or_label: str) -> dict:
    """Cancel a pending timer/alarm/reminder by its id or label ('all' cancels everything)."""
    keys = list(_reminders) if id_or_label == "all" else \
        [k for k, v in _reminders.items() if k == id_or_label or v["label"].lower() == id_or_label.lower()]
    for k in keys:
        _reminders.pop(k, None)
    return {"cancelled": len(keys)}


@tool(server, snapshot=_snapshot)
def save_note(text: str) -> dict:
    """Save a short note (shopping list item, idea, something to remember)."""
    notes = _load(NOTES, [])
    notes.append({"text": text, "at": _now().isoformat(timespec="minutes")})
    _save(NOTES, notes)
    return {"ok": True, "count": len(notes)}


@tool(server, snapshot=_snapshot)
def read_notes(last_n: int = 5) -> dict:
    """Read back the most recent saved notes."""
    return {"notes": _load(NOTES, [])[-max(1, int(last_n)):]}


@tool(server, snapshot=_snapshot)
def send_message(to: str, text: str) -> dict:
    """Send a short text message to a contact (queued to the outbox)."""
    box = _load(OUTBOX, [])
    box.append({"to": to, "text": text, "at": _now().isoformat(timespec="seconds"), "status": "queued"})
    _save(OUTBOX, box)
    return {"ok": True, "to": to, "status": "queued"}


@tool(server, snapshot=_snapshot)
def internal_pop_due() -> dict:
    """(Used by the assistant runtime, not the LLM.) Return and clear the timers/alarms/reminders that have fired."""
    out = list(_due)
    _due.clear()
    return {"due": out}


if __name__ == "__main__":
    serve(server, "assist")
