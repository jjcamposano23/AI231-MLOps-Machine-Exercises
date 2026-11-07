"""TTS -> ASR round trip through our MCP servers (checks Deepgram + measures each leg)."""
import asyncio, base64, io, sys, time, wave
sys.path.insert(0, ".")
import numpy as np
from assistant.config import load
from assistant.mcp_link import McpLink, ensure_servers, stop_children

PHRASES = ["Turn the lights blue and set a timer for five minutes.",
           "What's the weather in Siquijor tomorrow?",
           "Hey Iris, build me a flappy bird game."]

def wav16k(pcm24: bytes) -> bytes:
    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    n = int(len(x) * 16000 / 24000)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(y.tobytes())
    return b.getvalue()

async def main():
    cfg = load()
    ensure_servers({"asr": cfg.asr_mcp_url, "tts": cfg.tts_mcp_url})
    tts, asr = McpLink(cfg.tts_mcp_url, "tts"), McpLink(cfg.asr_mcp_url, "asr")
    await tts.start(); await asr.start()
    for p in PHRASES:
        t0 = time.monotonic(); r = await tts.call("synthesize", {"text": p, "sample_rate": 24000}); t1 = time.monotonic()
        pcm = base64.b64decode(r["audio_b64"])
        a = await asr.call("transcribe", {"audio_b64": base64.b64encode(wav16k(pcm)).decode()}); t2 = time.monotonic()
        print(f"TTS {round((t1-t0)*1000)} ms ({r['audio_s']}s audio) | ASR {round((t2-t1)*1000)} ms | heard: {a['text']!r}")
    await tts.close(); await asr.close()
try: asyncio.run(main())
finally: stop_children()
