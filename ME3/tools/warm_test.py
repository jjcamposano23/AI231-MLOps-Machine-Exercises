"""End-to-end check of the warm() tools through the real asr/tts MCP servers (they must be running).

    python -m tools.warm_test [--gap 8] [--reps 3]

For each rep: stay idle `gap` s, then transcribe a 3 s clip and synthesize one sentence ("cold"); then the same
with warm() called every 1 s during the gap, like the assistant does while the user is talking ("warm").
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import time

from assistant.mcp_link import McpLink
from tools.asr_gap_test import tone_wav


async def once(asr: McpLink, tts: McpLink, audio_b64: str, gap: float, warm: bool) -> tuple[float, float]:
    t_end = time.monotonic() + gap
    while time.monotonic() < t_end:
        if warm:
            asyncio.create_task(asr.call("warm", {}, timeout=5))
            asyncio.create_task(tts.call("warm", {}, timeout=5))
        await asyncio.sleep(min(1.0, max(0.0, t_end - time.monotonic())))
    t0 = time.monotonic()
    await asr.call("transcribe", {"audio_b64": audio_b64, "mime": "audio/wav"}, timeout=20)
    asr_ms = (time.monotonic() - t0) * 1000
    t0 = time.monotonic()
    async for _ in tts.stream("synthesize_stream", {"text": "Sure thing, boss. Here's what I found.",
                                                     "sample_rate": 24000, "voice": ""}):
        break
    tts_ms = (time.monotonic() - t0) * 1000
    return asr_ms, tts_ms


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    asr, tts = McpLink("http://127.0.0.1:8101/mcp", "asr"), McpLink("http://127.0.0.1:8102/mcp", "tts")
    audio_b64 = base64.b64encode(tone_wav(3.0)).decode()
    await once(asr, tts, audio_b64, 0, False)
    rows = {"cold": [], "warm": []}
    for _ in range(a.reps):
        for mode in ("cold", "warm"):
            r = await once(asr, tts, audio_b64, a.gap, mode == "warm")
            rows[mode].append(r)
            print(f"{mode}: asr {r[0]:6.0f} ms   tts first chunk {r[1]:6.0f} ms", flush=True)
    for mode, rs in rows.items():
        print(f"MEAN {mode}: asr {sum(r[0] for r in rs) / len(rs):6.0f} ms   tts {sum(r[1] for r in rs) / len(rs):6.0f} ms")
    await asr.close()
    await tts.close()


if __name__ == "__main__":
    asyncio.run(main())
