"""Publishes what Iris hears, says and does to the web emulator (emulator/public/index.html).

Events go to Firebase Realtime Database under /iris/events/<key> ({device, kind, payload, ts}); the emulator page
(on Firebase Hosting, or opened as a local file) follows that node with the database's REST streaming API, so it
works from any network.

No-op unless EMULATOR_FIREBASE_URL (database URL) and EMULATOR_FIREBASE_SECRET (database secret, Pi only) are set.
emit() never blocks or raises: events go to a queue that a daemon thread writes in small batches.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time

log = logging.getLogger("emulator")
_q: queue.Queue = queue.Queue(maxsize=2000)
_lock = threading.Lock()
_thread: threading.Thread | None = None
_seq = 0
_inflight = False
MAX_STR = 4000                      # long strings are cut, except the agent's page_html


def _cfg() -> tuple[str, str]:
    return (os.environ.get("EMULATOR_FIREBASE_URL", "").rstrip("/"), os.environ.get("EMULATOR_FIREBASE_SECRET", ""))


def enabled() -> bool:
    url, secret = _cfg()
    return bool(url and secret) and os.environ.get("EMULATOR", "1") != "0"


def _clean(v, key: str = ""):
    if isinstance(v, dict):
        return {str(k).replace(".", "_").replace("/", "_").replace("$", "_").replace("#", "_")
                .replace("[", "_").replace("]", "_"): _clean(x, str(k))
                for k, x in v.items() if k not in ("audio_b64", "ctx")}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in list(v)[:50]]
    if isinstance(v, str):
        return v if key == "page_html" or len(v) <= MAX_STR else v[:MAX_STR] + "…"
    if v is None or isinstance(v, (bool, int, float)):
        return v
    return str(v)


def emit(kind: str, **payload) -> None:
    global _seq
    if not enabled():
        return
    _start()
    with _lock:
        _seq = (_seq + 1) % 10000
        key = f"e{int(time.time() * 1000):013d}{_seq:04d}"          # time-ordered, string-sorted key
    try:
        _q.put_nowait((key, {"device": os.environ.get("EMULATOR_DEVICE", "iris-pi"), "kind": kind,
                             "payload": _clean(payload) | {"t": time.time()}, "ts": {".sv": "timestamp"}}))
    except queue.Full:
        pass


def _start() -> None:
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_worker, name="emulator-bus", daemon=True)
            _thread.start()
            atexit.register(_flush)


def _flush(wait_s: float = 2.0) -> None:
    """At exit, give the writer thread a moment to send what is still queued (daemon threads die with the process)."""
    end = time.monotonic() + wait_s
    while (not _q.empty() or _inflight) and time.monotonic() < end:
        time.sleep(0.05)


def _worker() -> None:
    global _inflight
    import httpx
    url, secret = _cfg()
    client = httpx.Client(timeout=httpx.Timeout(8.0, connect=5.0))
    endpoint = f"{url}/iris/events.json"
    while True:
        batch = dict([_q.get()])
        _inflight = True
        while len(batch) < 25:
            try:
                k, v = _q.get_nowait()
                batch[k] = v
            except queue.Empty:
                break
        for attempt in range(2):
            try:                                   # one multi-child PATCH = one request per batch
                r = client.patch(endpoint, params={"auth": secret, "print": "silent"},
                                 content=json.dumps(batch, ensure_ascii=False, default=str))
                if r.status_code >= 300:
                    log.warning("emulator: HTTP %s %s", r.status_code, r.text[:200])
                break
            except Exception as e:  # noqa: BLE001
                if attempt:
                    log.warning("emulator: %r", e)
                time.sleep(0.3)
        _inflight = False
