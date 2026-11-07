"""Acoustic echo cancellation (AEC) for an open speaker: real full duplex without headphones.

WebRTC's audio processing module (AEC3 + noise suppression + high-pass, via the `livekit` package; wheels for
Windows and Raspberry Pi aarch64) subtracts what the speaker is playing from the microphone signal:

    speaker output  ──far_end()──►  APM.process_reverse_stream   (the reference: what Iris is saying)
    microphone      ──AECFilter──►  APM.process_stream  ──► VAD / STT see the user, not Iris's echo

Without it the mic hears Iris, Silero VAD calls that an interruption, and she cuts herself off (or answers herself).
APM works on 10 ms frames at one rate (the mic rate, 16 kHz); the 24 kHz speaker audio is resampled for it.
"""
from __future__ import annotations

import threading

import numpy as np
from loguru import logger
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter


class EchoCanceller:
    def __init__(self, rate: int = 16000, delay_ms: int = 50, noise_suppression: bool = True):
        from livekit import rtc
        self._rtc = rtc
        self.rate = rate
        self.n = rate // 100                                   # samples per 10 ms frame
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=noise_suppression,
                                             high_pass_filter=True, auto_gain_control=False)
        self.apm.set_stream_delay_ms(delay_ms)
        self._rev = bytearray()
        self._fwd = bytearray()
        self._resamplers: dict = {}
        self._lock = threading.Lock()
        logger.info(f"AEC on (WebRTC APM, {rate} Hz, delay hint {delay_ms} ms)")

    def _to_rate(self, pcm: bytes, sr: int) -> bytes:
        if sr == self.rate:
            return pcm
        import soxr
        rs = self._resamplers.get(sr)
        if rs is None:
            rs = self._resamplers[sr] = soxr.ResampleStream(sr, self.rate, 1, dtype="int16")
        return rs.resample_chunk(np.frombuffer(pcm, dtype=np.int16)).tobytes()

    def far_end(self, pcm: bytes, sample_rate: int) -> None:
        """Audio that is about to be played by the speaker."""
        with self._lock:
            self._rev += self._to_rate(pcm, sample_rate)
            step = self.n * 2
            while len(self._rev) >= step:
                chunk = bytes(self._rev[:step])
                del self._rev[:step]
                self.apm.process_reverse_stream(self._rtc.AudioFrame(chunk, self.rate, 1, self.n))

    def near_end(self, pcm: bytes) -> bytes:
        """Microphone audio in; echo-cancelled audio out (whole 10 ms frames; any remainder waits for the next call)."""
        with self._lock:
            self._fwd += pcm
            step = self.n * 2
            out = bytearray()
            while len(self._fwd) >= step:
                frame = self._rtc.AudioFrame(bytearray(self._fwd[:step]), self.rate, 1, self.n)
                del self._fwd[:step]
                self.apm.process_stream(frame)                 # processed in place
                out += bytes(frame.data.cast("B")) if hasattr(frame.data, "cast") else bytes(frame.data)
            return bytes(out)


class AECFilter(BaseAudioFilter):
    """Pipecat mic filter that runs the echo canceller on every input chunk."""

    def __init__(self, aec: EchoCanceller):
        self._aec = aec

    async def start(self, sample_rate: int):
        if sample_rate != self._aec.rate:
            logger.warning(f"AEC expects {self._aec.rate} Hz input, got {sample_rate} Hz")

    async def stop(self):
        pass

    async def process_frame(self, frame):
        pass

    async def filter(self, audio: bytes) -> bytes:
        return self._aec.near_end(audio)
