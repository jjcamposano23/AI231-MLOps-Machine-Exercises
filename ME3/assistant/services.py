"""Pipecat STT/TTS services backed by our own MCP servers (asr :8101, tts :8102)."""
from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (BotStartedSpeakingFrame, ErrorFrame, Frame, TranscriptionFrame, TTSAudioRawFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from assistant.mcp_link import McpLink


class McpSTTService(SegmentedSTTService):
    """VAD-segmented STT: each finished utterance (WAV) goes to asr-mcp.transcribe().

    While the user talks it also pings asr-mcp.warm() and tts-mcp.warm() about once a second (and TTS until the
    reply starts), so neither Deepgram connection is idle when the real request comes (~1.35 s -> ~0.65 s on the
    Pi's Wi-Fi; see tools/asr_gap_test.py)."""

    def __init__(self, link: McpLink, tts_link: McpLink | None = None, warm: bool = True, **kwargs):
        kwargs.setdefault("ttfs_p99_latency", 0.8)
        super().__init__(settings=STTSettings(model="asr-mcp", language=None), **kwargs)
        self._link = link
        self._tts_link = tts_link
        self._warm = warm
        self._warm_task: asyncio.Task | None = None
        self._speaking = False
        self._stopped_at = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if self._warm:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._speaking = True
                if self._warm_task is None or self._warm_task.done():
                    self._warm_task = asyncio.create_task(self._warm_loop(), name="mcp-warm")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._speaking = False
                self._stopped_at = time.monotonic()
            elif isinstance(frame, BotStartedSpeakingFrame) and self._warm_task is not None:
                self._warm_task.cancel()
        await super().process_frame(frame, direction)

    async def _warm_loop(self) -> None:
        async def ping(link: McpLink) -> None:
            try:
                await link.call("warm", {}, timeout=5)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"warm {link.name}: {e!r}")

        while self._speaking or time.monotonic() - self._stopped_at < 6.0:
            if self._speaking:
                asyncio.create_task(ping(self._link))
            if self._tts_link is not None:
                asyncio.create_task(ping(self._tts_link))
            await asyncio.sleep(1.0)

    async def cleanup(self):
        if self._warm_task is not None:
            self._warm_task.cancel()
        await super().cleanup()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            await self.start_processing_metrics()
            res = await self._link.call("transcribe", {"audio_b64": base64.b64encode(audio).decode(),
                                                       "mime": "audio/wav"}, timeout=20)
            await self.stop_processing_metrics()
            text = (res.get("text") or "").strip()
            if text:
                logger.info(f"ASR [{res.get('ms')} ms]: {text}")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, result=res)
        except Exception as e:  # noqa: BLE001
            logger.error(f"ASR MCP error: {e!r}")
            yield ErrorFrame(error=f"ASR MCP error: {e}")


class McpTTSService(TTSService):
    """Sentence-by-sentence TTS: each sentence goes to tts-mcp.synthesize(); PCM is streamed out in chunks."""

    def __init__(self, link: McpLink, voice: str = "", **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model="tts-mcp", voice=voice or None, language=None), **kwargs)
        self._link = link
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        try:
            await self.start_ttfb_metrics()
            first = True
            async for b64 in self._link.stream("synthesize_stream", {"text": text, "sample_rate": self.sample_rate,
                                                                      "voice": self._voice}, timeout=30):
                if first:
                    first = False
                    await self.stop_ttfb_metrics()
                    await self.start_tts_usage_metrics(text)
                yield TTSAudioRawFrame(base64.b64decode(b64), self.sample_rate, 1, context_id=context_id)
        except Exception as e:  # noqa: BLE001
            logger.error(f"TTS MCP error: {e!r}")
            yield ErrorFrame(error=f"TTS MCP error: {e}")
