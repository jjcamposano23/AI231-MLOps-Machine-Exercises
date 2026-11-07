"""Self-echo guard for rooms without hardware echo cancellation (laptop / Pi with an open speaker).

BotSpeechTracker sits between the LLM and the TTS and remembers the words Iris spoke in the last few seconds.
EchoFilter sits between STT and the user aggregator and drops a transcript whose words are mostly words Iris just
said (her own voice picked up by the mic), so she never answers herself. Real user speech passes through.
"""
from __future__ import annotations

import re
import time
from collections import deque

from loguru import logger
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, LLMTextFrame,
                                   TranscriptionFrame, TTSSpeakFrame, TTSTextFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class _Memory:
    def __init__(self, window_s: float):
        self.window_s = window_s
        self.items: deque = deque()          # (t, word)
        self.last_stop = 0.0
        self.speaking = False

    def echo_possible(self, tail_s: float) -> bool:
        return self.speaking or time.monotonic() - self.last_stop < tail_s

    def add(self, text: str) -> None:
        now = time.monotonic()
        self.items.extend((now, w) for w in _words(text))

    def recent(self) -> set[str]:
        cutoff = time.monotonic() - self.window_s
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()
        return {w for _, w in self.items}


class BotSpeechTracker(FrameProcessor):
    def __init__(self, memory: _Memory):
        super().__init__()
        self._m = memory

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMTextFrame, TTSSpeakFrame)) and not isinstance(frame, TTSTextFrame):
            self._m.add(frame.text)          # words about to be spoken (seen BEFORE the audio plays)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._m.speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._m.speaking = False
            self._m.last_stop = time.monotonic()
        await self.push_frame(frame, direction)


class EchoFilter(FrameProcessor):
    def __init__(self, memory: _Memory, overlap: float = 0.7, min_words: int = 2, tail_s: float = 2.0):
        super().__init__()
        self._m = memory
        self._overlap = overlap
        self._min_words = min_words
        self._tail_s = tail_s

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            words = _words(frame.text)
            recent = self._m.recent()
            # only while Iris talks or just after: a user asking the same words later is not an echo
            if words and recent and self._m.echo_possible(self._tail_s):
                share = sum(w in recent for w in words) / len(words)
                if share >= self._overlap and len(words) >= self._min_words:
                    logger.info(f"echo filter: dropped own speech {frame.text!r} ({share:.0%} overlap)")
                    return
        await self.push_frame(frame, direction)


def make_echo_guard(window_s: float = 6.0):
    m = _Memory(window_s)
    return BotSpeechTracker(m), EchoFilter(m)
