"""Local mic/speaker transport that survives output-device changes.

Pipecat's LocalAudioTransport opens the speaker once. If that device goes away (a Bluetooth speaker powers off, or
Windows switches the default output), every later write fails with "[Errno -9988] Stream closed" until a restart.
This output reopens the stream on a fresh PortAudio instance (whose device list is current) and carries on.
"""
from __future__ import annotations

import time

import pyaudio
from loguru import logger
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.local.audio import (LocalAudioOutputTransport, LocalAudioTransport,
                                            LocalAudioTransportParams)


class ResilientAudioOutput(LocalAudioOutputTransport):
    RETRY_EVERY_S = 1.5

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, aec=None):
        super().__init__(py_audio, params)
        self._last_try = 0.0
        self._aec = aec                                # EchoCanceller: gets every chunk we play as its reference

    def _reopen(self) -> bool:
        try:
            if self._out_stream is not None:
                self._out_stream.close()
        except Exception:  # noqa: BLE001
            pass
        self._out_stream = None
        try:
            self._py_audio = pyaudio.PyAudio()       # fresh instance = fresh device list (the mic keeps its own)
            self._out_stream = self._py_audio.open(
                format=self._py_audio.get_format_from_width(2), channels=self._params.audio_out_channels,
                rate=self.sample_rate, output=True, output_device_index=self._params.output_device_index)
            self._out_stream.start_stream()
            name = self._py_audio.get_default_output_device_info().get("name", "?") \
                if self._params.output_device_index is None else f"#{self._params.output_device_index}"
            logger.warning(f"speaker stream reopened on {name}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"speaker reopen failed: {e}")
            self._out_stream = None
            return False

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._aec is not None:
            self._aec.far_end(frame.audio, frame.sample_rate)
        if self._out_stream is not None:
            try:
                return await super().write_audio_frame(frame)
            except OSError as e:
                logger.warning(f"speaker stream lost ({e}); reopening on the current output device")
        now = time.monotonic()
        if now - self._last_try < self.RETRY_EVERY_S:
            return False                              # drop this chunk; try again shortly
        self._last_try = now
        await self.get_event_loop().run_in_executor(self._executor, self._reopen)
        return False


class ResilientLocalAudioTransport(LocalAudioTransport):
    def __init__(self, params: LocalAudioTransportParams, aec=None):
        super().__init__(params)
        self._aec = aec

    def output(self):
        if not self._output:
            self._output = ResilientAudioOutput(self._pyaudio, self._params, aec=self._aec)
        return self._output
