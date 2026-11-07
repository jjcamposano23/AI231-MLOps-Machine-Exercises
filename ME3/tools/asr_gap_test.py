"""How does Deepgram /v1/listen latency depend on the idle gap before the request (and on a warm-up ping)?

    python -m tools.asr_gap_test [--wav some.wav]

Prints one line per (gap, mode): ms for the real request and whether httpx had to open a new connection.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import math
import os
import struct
import time
import wave

from mcp_servers.common import load_env, make_http

load_env()
KEY = os.environ.get("DEEPGRAM_API_KEY", "")
URL = "https://api.deepgram.com/v1/listen"
PARAMS = [("model", "nova-3"), ("language", "en")]


def tone_wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()


def n_conns(client) -> int:
    try:
        return len(client._transport._pool.connections)
    except Exception:  # noqa: BLE001
        return -1


DROPS = [0]


async def post(client, audio: bytes) -> float:
    t0 = time.monotonic()
    for attempt in range(2):
        try:
            r = await client.post(URL, params=PARAMS, content=audio,
                                  headers={"Authorization": f"Token {KEY}", "Content-Type": "audio/wav"})
            break
        except Exception:  # noqa: BLE001  (server closed the idle connection: count it, retry like asr_server)
            DROPS[0] += 1
            if attempt:
                raise
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    audio = open(a.wav, "rb").read() if a.wav else tone_wav(3.0)
    ping = tone_wav(0.1)
    c = make_http(20.0)
    await post(c, audio)                                           # open the connection
    for gap in (0.2, 1, 2, 3, 5, 8, 12):
        for mode in ("plain", "warm"):
            for _ in range(a.reps):
                if mode == "plain":
                    await asyncio.sleep(gap)
                else:                                              # ping every 1 s during the gap (as during speech)
                    t_end = time.monotonic() + gap
                    while time.monotonic() < t_end:
                        await post(c, ping)
                        await asyncio.sleep(min(1.0, max(0.0, t_end - time.monotonic())))
                d0 = DROPS[0]
                ms = await post(c, audio)
                print(f"gap {gap:>4}s {mode:5}  {ms:6.0f} ms   {'DROPPED+retry' if DROPS[0] > d0 else ''}", flush=True)
    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
