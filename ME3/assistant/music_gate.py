"""Music gate: lets Iris share the speaker with music without reacting to the song.

Iris listens all the time (no wake word), so while music plays the mic hears it. While the home server's
data/music_playing flag exists:
  * a transcript that doesn't name Iris is dropped (lyrics, the room);
  * "Iris, stop / pause / next / volume up|down the music" is handled right here: the home tool is called directly
    and Iris says a short "Okay", without waiting for end-of-turn detection or the LLM (music in the background
    can keep Smart Turn from ever deciding the user has finished);
  * other "Iris, ..." requests go on to the LLM as usual;
  * the music is ducked (home.internal_duck) while Iris speaks and restored shortly after.
Sits between the echo filter and the user aggregator.
"""
from __future__ import annotations

import asyncio
import re

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from mcp_servers.common import DATA

FLAG = DATA / "music_playing"
WAKE = re.compile(r"\b(iris|irish|ires|iras|eyris|irus|virus|iris's)\b", re.I)
# (pattern, tool, args, spoken confirmation)
COMMANDS = [
    (re.compile(r"\b(stop|turn off|shut off|kill|end)\b", re.I), "stop_music", {}, "Okay, music off."),
    (re.compile(r"\b(pause|hold)\b", re.I), "pause_music", {}, "Paused."),
    (re.compile(r"\b(next|skip|another)\b", re.I), "next_song", {}, "Next song."),
    (re.compile(r"\b(louder|volume up|turn (it )?up|increase)\b", re.I), "change_volume", {"delta_percent": 20},
     "Louder."),
    (re.compile(r"\b(quieter|softer|lower|volume down|turn (it )?down|decrease)\b", re.I), "change_volume",
     {"delta_percent": -20}, "Quieter."),
]


class MusicGate(FrameProcessor):
    def __init__(self, home_link=None, restore_after_s: float = 1.2):
        super().__init__()
        self._home = home_link
        self._restore_after = restore_after_s
        self._bot = False
        self._restore: asyncio.Task | None = None

    @staticmethod
    def music_on() -> bool:
        return FLAG.exists()

    def _duck(self, on: bool) -> None:
        if self._home is None:
            return

        async def go():
            try:
                if not on:
                    await asyncio.sleep(self._restore_after)
                await self._home.call("internal_duck", {"on": on}, timeout=5)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug(f"music duck: {e!r}")
        if self._restore is not None and not self._restore.done():
            self._restore.cancel()
        t = asyncio.create_task(go(), name="music-duck")
        if not on:
            self._restore = t

    async def _fast(self, text: str) -> bool:
        """Handle a music command directly. True if handled."""
        if self._home is None or not re.search(r"\b(music|song|it|volume|track|this)\b|^\W*\w+\W+(stop|pause|next|skip)",
                                               text, re.I):
            return False
        for pat, tool, args, say in COMMANDS:
            if pat.search(text):
                try:
                    await self._home.call(tool, args, timeout=8)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"music gate: {tool} failed: {e!r}")
                    return False
                logger.info(f"music gate: {text!r} -> {tool}{args or ''}")
                await self.push_frame(TTSSpeakFrame(say), FrameDirection.DOWNSTREAM)
                return True
        return False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._bot and self.music_on():
                self._duck(True)
            self._bot = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._bot:
                self._duck(False)
            self._bot = False
        elif (isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM
              and self.music_on()):
            text = frame.text or ""
            short_cmd = len(text.split()) <= 6 and re.search(r"\b(music|song)\b", text, re.I)
            if short_cmd and await self._fast(text):       # "stop the music" even if the ASR missed "Iris"
                return
            if not WAKE.search(text):
                logger.info(f"music gate: ignored {text!r} (music playing, Iris not named)")
                return
            if await self._fast(text):
                return
        await self.push_frame(frame, direction)
