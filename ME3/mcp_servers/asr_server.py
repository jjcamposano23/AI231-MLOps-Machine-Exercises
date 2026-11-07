"""ASR MCP server: speech segment (WAV) -> text.

Providers (ASR_PROVIDER):
  deepgram       Deepgram pre-recorded API, model nova-3 (default; DEEPGRAM_API_KEY)
  openai_compat  any OpenAI-compatible /audio/transcriptions endpoint (ASR_BASE_URL, ASR_MODEL, ASR_API_KEY),
                 e.g. Groq whisper-large-v3-turbo, or the class's SHARC DGX vLLM (needs the VPN)

Run:  python -m mcp_servers.asr_server
"""
from __future__ import annotations

import base64
import os
import time

import httpx
from mcp.server.mcpserver import MCPServer

from mcp_servers.common import Warmth, get_logger, load_env, make_http, serve

load_env()
log = get_logger("asr")

PROVIDER = os.environ.get("ASR_PROVIDER", "deepgram").lower()
MODEL = os.environ.get("ASR_MODEL", "nova-3" if PROVIDER == "deepgram" else "whisper-large-v3-turbo")
LANG = os.environ.get("ASR_LANGUAGE", "en")
KEYTERMS = [k.strip() for k in os.environ.get("ASR_KEYTERMS", "Iris,Raspberry Pi,Siquijor,Davao,flappy bird").split(",") if k.strip()]

# one keep-alive connection pool for the life of the server (see common.make_http / keep_warm)
_http = make_http(20.0)

_warmth = Warmth()
_PING_WAV = b"RIFF" + (36 + 3200).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") +     (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little") +     (32000).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little") +     b"data" + (3200).to_bytes(4, "little") + bytes(3200)          # 0.1 s of silence

server = MCPServer("asr", instructions="Speech-to-text for the ME3 voice agent. Call transcribe with a WAV segment.")


async def _deepgram(audio: bytes, mime: str) -> dict:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    # no smart_format/punctuate: they add ~0.3-1 s server-side and the LLM does not need punctuation
    params = [("model", MODEL), ("language", LANG)]
    params += [("keyterm", k) for k in KEYTERMS]
    r = await _http.post("https://api.deepgram.com/v1/listen", params=params, content=audio,
                         headers={"Authorization": f"Token {key}", "Content-Type": mime})
    r.raise_for_status()
    alt = r.json()["results"]["channels"][0]["alternatives"][0]
    return {"text": alt.get("transcript", ""), "confidence": alt.get("confidence")}


async def _openai_compat(audio: bytes, mime: str) -> dict:
    base = os.environ.get("ASR_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    key = os.environ.get("ASR_API_KEY", "")
    files = {"file": ("speech.wav", audio, mime)}
    data = {"model": MODEL, "language": LANG, "response_format": "json"}
    r = await _http.post(f"{base}/audio/transcriptions", files=files, data=data,
                         headers={"Authorization": f"Bearer {key}"} if key else {})
    r.raise_for_status()
    return {"text": r.json().get("text", ""), "confidence": None}


@server.tool()
async def transcribe(audio_b64: str, mime: str = "audio/wav") -> dict:
    """Transcribe one speech segment.

    Args:
        audio_b64: The audio (16 kHz mono WAV) encoded as base64.
        mime: MIME type of the audio, default audio/wav.
    Returns:
        {"text", "confidence", "provider", "model", "ms", "audio_s"}
    """
    audio = base64.b64decode(audio_b64)
    t0 = time.monotonic()
    fn = _deepgram if PROVIDER == "deepgram" else _openai_compat
    await _warmth.before()
    try:
        try:
            out = await fn(audio, mime)
        except httpx.TransportError as e:               # stale keep-alive connection: retry once on a new one
            log.warning("retrying after connection error: %r", e)
            out = await fn(audio, mime)
    finally:
        _warmth.after()
    ms = round((time.monotonic() - t0) * 1000)
    audio_s = round(max(0, len(audio) - 44) / 32000, 2)          # 16 kHz * 2 bytes
    out.update(provider=PROVIDER, model=MODEL, ms=ms, audio_s=audio_s)
    log.info("transcribed %.2fs audio in %d ms: %r", audio_s, ms, out["text"])
    return out


@server.tool()
async def warm() -> dict:
    """Keep the provider connection hot while the user is speaking (no-op if it was used in the last 1.5 s).
    The assistant calls this about once a second between VAD start and VAD stop."""
    fn = _deepgram if PROVIDER == "deepgram" else _openai_compat
    return {"pinged": await _warmth.ping(lambda: fn(_PING_WAV, "audio/wav"))}


@server.tool()
async def info() -> dict:
    """Which ASR provider and model this server uses."""
    return {"provider": PROVIDER, "model": MODEL, "language": LANG}


if __name__ == "__main__":
    serve(server, "asr")
