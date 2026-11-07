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

from mcp_servers.common import DATA, get_logger, load_env, serve

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
        log.info("due: %s", item)


def _schedule(kind: str, label: str, due_epoch: float) -> dict:
    rid = uuid.uuid4().hex[:6]
    item = {"id": rid, "kind": kind, "label": label, "due": due_epoch,
            "due_local": datetime.fromtimestamp(due_epoch, ZoneInfo(TZ)).strftime("%I:%M:%S %p").lstrip("0")}
    _reminders[rid] = item
    asyncio.get_running_loop().create_task(_fire_later(rid))
    return item


@server.tool()
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


@server.tool()
async def get_weather(city: str = "") -> dict:
    """Current weather and today's forecast for a city (default: the user's home city)."""
    city = city or HOME_CITY
    g = await _http.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1})
    res = (g.json().get("results") or [])
    if not res:
        return {"error": f"could not find {city!r}"}
    p = res[0]
    w = await _http.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 1})
    d = w.json()
    c, day = d["current"], d["daily"]
    return {"place": f"{p['name']}, {p.get('country', '')}", "now": _WMO.get(c["weather_code"], "unknown"),
            "temp_c": c["temperature_2m"], "feels_like_c": c["apparent_temperature"],
            "humidity_pct": c["relative_humidity_2m"], "wind_kmh": c["wind_speed_10m"],
            "today_high_c": day["temperature_2m_max"][0], "today_low_c": day["temperature_2m_min"][0],
            "rain_chance_pct": day["precipitation_probability_max"][0]}


@server.tool()
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


@server.tool()
async def set_timer(seconds: int, label: str = "timer") -> dict:
    """Start a countdown timer. The assistant announces it when it finishes."""
    item = _schedule("timer", label, time.time() + max(1, int(seconds)))
    return {"ok": True, "id": item["id"], "ends_at": item["due_local"], "seconds": int(seconds)}


@server.tool()
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


@server.tool()
async def create_reminder(task: str, in_minutes: float) -> dict:
    """Remind the user about a task after a number of minutes (fractions allowed, e.g. 0.5)."""
    item = _schedule("reminder", task, time.time() + max(0.1, float(in_minutes)) * 60)
    return {"ok": True, "id": item["id"], "at": item["due_local"], "task": task}


@server.tool()
def list_reminders() -> dict:
    """List the pending timers, alarms and reminders."""
    now = time.time()
    return {"pending": [{"id": i["id"], "kind": i["kind"], "label": i["label"], "at": i["due_local"],
                         "seconds_left": round(i["due"] - now)} for i in sorted(_reminders.values(),
                                                                                key=lambda x: x["due"])]}


@server.tool()
def cancel_reminder(id_or_label: str) -> dict:
    """Cancel a pending timer/alarm/reminder by its id or label ('all' cancels everything)."""
    keys = list(_reminders) if id_or_label == "all" else \
        [k for k, v in _reminders.items() if k == id_or_label or v["label"].lower() == id_or_label.lower()]
    for k in keys:
        _reminders.pop(k, None)
    return {"cancelled": len(keys)}


@server.tool()
def save_note(text: str) -> dict:
    """Save a short note (shopping list item, idea, something to remember)."""
    notes = _load(NOTES, [])
    notes.append({"text": text, "at": _now().isoformat(timespec="minutes")})
    _save(NOTES, notes)
    return {"ok": True, "count": len(notes)}


@server.tool()
def read_notes(last_n: int = 5) -> dict:
    """Read back the most recent saved notes."""
    return {"notes": _load(NOTES, [])[-max(1, int(last_n)):]}


@server.tool()
def send_message(to: str, text: str) -> dict:
    """Send a short text message to a contact (queued to the outbox)."""
    box = _load(OUTBOX, [])
    box.append({"to": to, "text": text, "at": _now().isoformat(timespec="seconds"), "status": "queued"})
    _save(OUTBOX, box)
    return {"ok": True, "to": to, "status": "queued"}


@server.tool()
def internal_pop_due() -> dict:
    """(Used by the assistant runtime, not the LLM.) Return and clear the timers/alarms/reminders that have fired."""
    out = list(_due)
    _due.clear()
    return {"due": out}


if __name__ == "__main__":
    serve(server, "assist")
