"""Per-turn latency log on ONE clock (the Pi's time.monotonic()), for live sessions.

Course template section 5 (same definitions as fd-benchmark, so live logs and benchmark runs compare 1:1):
    VAD end -> ASR final -> LLM 1st token -> TTS 1st audio -> speaker
    E2E      = audio_out        - vad_end          (end of speech -> audio out)
    ASR final= asr_final        - vad_end          (end of speech -> transcript)
    LLM TTFT = llm_first_token  - asr_final        (transcript -> first token)
    TTS TTFB = tts_first_audio  - tts_request      (text -> first audio chunk)
    Barge-in = bot_stopped      - user speech start while the bot was talking
One JSON line per turn goes to logs/latency.jsonl; `python tools/latency_report.py` gives mean + p95.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import pipecat.frames.frames as F
from pipecat.observers.base_observer import BaseObserver


def _ms(a, b):
    return None if a is None or b is None else round((a - b) * 1000)


class LatencyObserver(BaseObserver):
    def __init__(self, latency_log: Path, turns_log: Path, session: str, **kwargs):
        super().__init__(**kwargs)
        self._lat = Path(latency_log)
        self._turns = Path(turns_log)
        self._session = session
        self._seen: deque = deque(maxlen=4096)
        self._seen_set: set = set()
        self._n = 0
        self._reset()
        self._bot_speaking = False
        self._barge_start = None

    def _reset(self):
        self.t = {"vad_end": None, "asr_final": None, "llm_request": None, "llm_first_token": None,
                  "tts_request": None, "tts_first_audio": None}
        self._text = []
        self._llm_open = self._tts_open = False
        self._tools = []

    def _first(self, frame) -> bool:
        fid = getattr(frame, "id", id(frame))
        if fid in self._seen_set:
            return False
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(fid)
        self._seen_set.add(fid)
        return True

    def _write(self, path: Path, rec: dict) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    async def on_push_frame(self, data):
        f = data.frame
        if not self._first(f):
            return
        now = time.monotonic()
        t = self.t
        if isinstance(f, F.TTSAudioRawFrame):
            if self._tts_open:
                self._tts_open = False
                if t["tts_first_audio"] is None:
                    t["tts_first_audio"] = now
        elif isinstance(f, F.LLMTextFrame):
            if self._llm_open:
                self._llm_open = False
                if t["llm_first_token"] is None and t["llm_request"] is not None:
                    t["llm_first_token"] = now
        elif isinstance(f, F.VADUserStartedSpeakingFrame):
            if self._bot_speaking:
                self._barge_start = now
        elif isinstance(f, F.VADUserStoppedSpeakingFrame):
            t["vad_end"] = now
            if t["llm_request"] is not None:            # a new utterance after an unanswered request
                for k in ("asr_final", "llm_request", "llm_first_token", "tts_request", "tts_first_audio"):
                    t[k] = None
        elif isinstance(f, F.TranscriptionFrame):
            t["asr_final"] = now
            self._text.append(f.text)
            self._write(self._turns, {"session": self._session, "mono": now, "role": "user", "text": f.text})
        elif isinstance(f, F.LLMContextFrame):
            if t["llm_request"] is None:
                t["llm_request"] = now
        elif isinstance(f, F.LLMFullResponseStartFrame):
            self._llm_open = True
        elif isinstance(f, F.TTSStartedFrame):
            self._tts_open = True
            if t["tts_request"] is None:
                t["tts_request"] = now
        elif isinstance(f, F.FunctionCallInProgressFrame):
            if any(x.get("id") == f.tool_call_id for x in self._tools):      # broadcast both ways: count once
                return
            self._tools.append({"id": f.tool_call_id, **{"name": f.function_name, "start": now}})
            self._write(self._turns, {"session": self._session, "mono": now, "role": "tool_call",
                                      "name": f.function_name, "args": getattr(f, "arguments", None)})
        elif isinstance(f, F.FunctionCallResultFrame):
            for x in self._tools:
                if x.get("id") == f.tool_call_id and "ms" not in x:
                    x["ms"] = _ms(now, x["start"])
        elif isinstance(f, F.BotStartedSpeakingFrame):
            self._bot_speaking = True
            if t["vad_end"] is not None and t["asr_final"] is not None:
                self._n += 1
                rec = {"session": self._session, "turn": self._n, "mono_vad_end": t["vad_end"],
                       "e2e_ms": _ms(now, t["vad_end"]),
                       "asr_final_ms": _ms(t["asr_final"], t["vad_end"]),
                       "llm_ttft_ms": _ms(t["llm_first_token"], t["asr_final"]),
                       "tts_ttfb_ms": _ms(t["tts_first_audio"], t["tts_request"]),
                       "tools": [{"name": x["name"], "ms": x.get("ms")} for x in self._tools],
                       "text": " ".join(self._text)}
                self._write(self._lat, rec)
                try:
                    from mcp_servers import emulator_bus
                    emulator_bus.emit("turn", **{k: v for k, v in rec.items() if not k.startswith("mono")})
                except Exception:  # noqa: BLE001
                    pass
            self._reset()
        elif isinstance(f, F.BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._barge_start is not None:
                self._write(self._lat, {"session": self._session, "event": "barge_in",
                                        "barge_in_ms": _ms(now, self._barge_start)})
                try:
                    from mcp_servers import emulator_bus
                    emulator_bus.emit("barge_in_ms", ms=_ms(now, self._barge_start), session=self._session)
                except Exception:  # noqa: BLE001
                    pass
                self._barge_start = None


class EmulatorObserver(BaseObserver):
    """Feeds the web emulator (mcp_servers/emulator_bus): what the user said (after the echo filter), what Iris
    said, tool calls, and Iris's state (listening / thinking / speaking / idle)."""

    def __init__(self, session: str, **kwargs):
        super().__init__(**kwargs)
        from mcp_servers import emulator_bus
        self._bus = emulator_bus
        self._session = session
        self._seen: deque = deque(maxlen=4096)
        self._seen_set: set = set()
        self._tools: set = set()
        self._reply: list[str] = []
        self._bot_speaking = False
        self._status = ""

    def _first(self, frame, tag: str = "") -> bool:
        fid = (getattr(frame, "id", id(frame)), tag)
        if fid in self._seen_set:
            return False
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(fid)
        self._seen_set.add(fid)
        return True

    def _state(self, s: str) -> None:
        if s != self._status:
            self._status = s
            self._bus.emit("status", state=s, session=self._session)

    async def on_push_frame(self, data):
        f = data.frame
        src = type(data.source).__name__
        dst = type(data.destination).__name__
        if isinstance(f, F.TranscriptionFrame):
            if "UserAggregator" in dst and self._first(f, "user"):
                self._bus.emit("user", text=f.text, session=self._session)
        elif isinstance(f, F.LLMTextFrame):
            if "LLMService" in src and self._first(f):
                self._reply.append(f.text)
        elif isinstance(f, F.LLMFullResponseEndFrame):
            if "LLMService" in src and self._first(f):
                text = "".join(self._reply).strip()
                self._reply = []
                if text:
                    self._bus.emit("iris", text=text, session=self._session)
        elif isinstance(f, F.TTSSpeakFrame):
            if self._first(f):
                self._bus.emit("iris", text=f.text, aside=True, session=self._session)
        elif isinstance(f, F.FunctionCallInProgressFrame):
            if f.tool_call_id not in self._tools:
                self._tools.add(f.tool_call_id)
                self._bus.emit("llm_tool", name=f.function_name, args=getattr(f, "arguments", None),
                               id=f.tool_call_id, session=self._session)
        elif isinstance(f, F.VADUserStartedSpeakingFrame):
            if self._first(f):
                if self._bot_speaking:
                    self._bus.emit("barge_in", session=self._session)
                self._state("listening")
        elif isinstance(f, F.VADUserStoppedSpeakingFrame):
            if self._first(f):
                self._state("thinking")
        elif isinstance(f, F.BotStartedSpeakingFrame):
            if self._first(f):
                self._bot_speaking = True
                self._state("speaking")
        elif isinstance(f, F.BotStoppedSpeakingFrame):
            if self._first(f):
                self._bot_speaking = False
                self._state("idle")
