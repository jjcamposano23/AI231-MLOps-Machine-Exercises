"""TTS MCP server: text -> 16-bit mono PCM.

Providers (TTS_PROVIDER):
  deepgram       Deepgram Aura-2 REST /v1/speak (default; DEEPGRAM_API_KEY, TTS_VOICE e.g. aura-2-thalia-en)
  openai_compat  any OpenAI-compatible /audio/speech returning raw PCM (TTS_BASE_URL, TTS_MODEL, TTS_API_KEY, TTS_VOICE)

The assistant sends one sentence at a time, so time-to-first-audio is one sentence's synthesis time.
Run:  python -m mcp_servers.tts_server
"""
from __future__ import annotations

import base64
import os
import time

import httpx
from mcp.server.mcpserver import Context, MCPServer

from mcp_servers.common import Warmth, get_logger, load_env, make_http, serve

load_env()
log = get_logger("tts")

PROVIDER = os.environ.get("TTS_PROVIDER", "deepgram").lower()
VOICE = os.environ.get("TTS_VOICE", "aura-2-thalia-en" if PROVIDER == "deepgram" else "alloy")
MODEL = os.environ.get("TTS_MODEL", "aura-2" if PROVIDER == "deepgram" else "tts-1")

_http = make_http(30.0)
_warmth = Warmth()

server = MCPServer("tts", instructions="Text-to-speech for the ME3 voice agent. Call synthesize with one sentence.")


async def _deepgram(text: str, sample_rate: int, voice: str) -> bytes:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    r = await _http.post("https://api.deepgram.com/v1/speak",
                         params={"model": voice, "encoding": "linear16", "sample_rate": sample_rate,
                                 "container": "none"},
                         json={"text": text},
                         headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
    r.raise_for_status()
    return r.content


async def _openai_compat(text: str, sample_rate: int, voice: str) -> bytes:
    base = os.environ.get("TTS_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("TTS_API_KEY", "")
    r = await _http.post(f"{base}/audio/speech",
                         json={"model": MODEL, "input": text, "voice": voice, "response_format": "pcm"},
                         headers={"Authorization": f"Bearer {key}"} if key else {})
    r.raise_for_status()
    return r.content            # OpenAI pcm = 24 kHz 16-bit mono


@server.tool()
async def synthesize(text: str, sample_rate: int = 24000, voice: str = "") -> dict:
    """Synthesize speech for one sentence.

    Args:
        text: The sentence to speak.
        sample_rate: Output sample rate in Hz (default 24000).
        voice: Optional voice override.
    Returns:
        {"audio_b64": raw 16-bit little-endian mono PCM as base64, "sample_rate", "ms", "audio_s", "provider", "voice"}
    """
    t0 = time.monotonic()
    v = voice or VOICE
    if PROVIDER == "deepgram":
        pcm = await _deepgram(text, sample_rate, v)
    else:
        pcm = await _openai_compat(text, sample_rate, v)
        sample_rate = 24000
    ms = round((time.monotonic() - t0) * 1000)
    audio_s = round(len(pcm) / (2 * sample_rate), 2)
    log.info("synthesized %d chars -> %.2fs audio in %d ms", len(text), audio_s, ms)
    return {"audio_b64": base64.b64encode(pcm).decode(), "sample_rate": sample_rate, "ms": ms,
            "audio_s": audio_s, "provider": PROVIDER, "voice": v}


@server.tool()
async def synthesize_stream(text: str, ctx: Context, sample_rate: int = 24000, voice: str = "") -> dict:
    """Synthesize speech for one sentence and STREAM it: every ~100 ms of 16-bit mono PCM is sent as an MCP
    progress notification (message = base64 chunk) as soon as Deepgram produces it, so playback can start
    before synthesis finishes. The final result only carries timing.

    Args:
        text: The sentence to speak.
        sample_rate: Output sample rate in Hz (default 24000).
        voice: Optional voice override.
    """
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    v = voice or VOICE
    t0 = time.monotonic()
    first_ms, total, seq, buf = None, 0, 0, b""
    step = int(sample_rate * 0.1) * 2                      # ~100 ms of audio per notification
    req = _http.build_request("POST", "https://api.deepgram.com/v1/speak",
                              params={"model": v, "encoding": "linear16", "sample_rate": sample_rate,
                                      "container": "none"},
                              json={"text": text},
                              headers={"Authorization": f"Token {key}", "Content-Type": "application/json"})
    await _warmth.before()
    try:
        try:
            r = await _http.send(req, stream=True)
        except httpx.TransportError as e:                  # stale keep-alive connection: retry once
            log.warning("retrying after connection error: %r", e)
            r = await _http.send(req, stream=True)
    except BaseException:
        _warmth.after()
        raise
    try:
        if r.status_code != 200:
            raise RuntimeError(f"Deepgram TTS HTTP {r.status_code}: {(await r.aread())[:200]!r}")
        async for chunk in r.aiter_bytes():
            buf += chunk
            while len(buf) >= step:
                out, buf = buf[:step], buf[step:]
                if first_ms is None:
                    first_ms = round((time.monotonic() - t0) * 1000)
                seq += 1
                total += len(out)
                await ctx.report_progress(seq, None, base64.b64encode(out).decode())
    finally:
        await r.aclose()
        _warmth.after()
    if buf:
        out = buf[: len(buf) - len(buf) % 2]
        if first_ms is None:
            first_ms = round((time.monotonic() - t0) * 1000)
        seq += 1
        total += len(out)
        await ctx.report_progress(seq, None, base64.b64encode(out).decode())
    ms = round((time.monotonic() - t0) * 1000)
    log.info("streamed %d chars -> %.2fs audio, first chunk %s ms, done %d ms", len(text), total / (2 * sample_rate),
             first_ms, ms)
    return {"chunks": seq, "first_chunk_ms": first_ms, "ms": ms, "audio_s": round(total / (2 * sample_rate), 2),
            "sample_rate": sample_rate, "voice": v}


@server.tool()
async def warm() -> dict:
    """Keep the provider connection hot while the user is speaking (no-op if it was used in the last 1.5 s).
    The assistant calls this about once a second from VAD start until the reply starts."""
    if PROVIDER != "deepgram":
        return {"pinged": False}
    return {"pinged": await _warmth.ping(lambda: _deepgram("Hi.", 24000, VOICE))}


@server.tool()
async def info() -> dict:
    """Which TTS provider and voice this server uses."""
    return {"provider": PROVIDER, "voice": VOICE, "model": MODEL}


if __name__ == "__main__":
    serve(server, "tts", json_response=False)   # SSE, so audio chunks stream as progress notifications
