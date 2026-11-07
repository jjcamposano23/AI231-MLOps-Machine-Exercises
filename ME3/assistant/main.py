"""Iris: a full-duplex voice agent for the Raspberry Pi (AI 231 ME3).

    mic -> Silero VAD -> asr-mcp (Deepgram) -> LLM (+ MCP tools: assist, home; + async background agent) ->
    tts-mcp (Deepgram) -> speaker.     Barge-in: user speech while Iris talks cancels the LLM/TTS and playback.

Run on the Pi (or laptop):   python -m assistant.main          (--list-devices to pick mic/speaker indexes)
Benchmark (fd-benchmark):    fd-benchmark/launch_assistant.py calls Assistant.build(transport=...) for us.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import uuid

from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.adapters.llm.schemas.direct_function import tool_options
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (LLMContextAggregatorPair,
                                                                    LLMUserAggregatorParams)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.mcp_service import MCPClient
from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from assistant.config import Config, load
from assistant.echo_filter import make_echo_guard
from assistant.mcp_link import McpLink, ensure_servers, stop_children
from assistant.observers import LatencyObserver
from assistant.services import McpSTTService, McpTTSService


# A tool turn needs a second LLM call after the tool returns (~1.5 s more), so Iris says a short filler at once.
LOOKUP_TOOLS = {"web_search", "get_weather", "get_time", "list_reminders", "read_notes", "get_home_state"}
DELEGATE_FILLERS = ["On it. I'll let you know when it's ready.", "Sure, working on that in the background."]
LOOKUP_FILLERS = ["One sec.", "Let me check.", "Checking."]
ACTION_FILLERS = ["Okay.", "Sure.", "Got it."]


PROMPT_VERSION = "v3"     # versioned in the repo (template section 2)


def system_prompt(cfg: Config, agent: bool) -> str:
    p = f"""You are {cfg.bot_name}, a voice assistant running on a Raspberry Pi in the user's room.
Personality: a calm, quick, unflappable tech assistant. Efficient and warm, with a light, dry wit now and then;
call the user "boss" occasionally. Confident, never rambling.
Everything you say is spoken aloud by a text-to-speech voice, so:
- Short, natural sentences; at most two per turn. Never read markdown, lists, emojis or URLs aloud; say numbers
  the way people say them.
- If the user speaks, stop and listen; then answer what they just said, not the old topic.
- Confirm before costly or irreversible actions (sending a message, cancelling all reminders, long background jobs
  that cost money) unless the user was already explicit.
- If the transcript looks garbled or like background talk not meant for you, say nothing important; ask briefly.
Use your tools instead of guessing for anything about now: time, weather, news, prices, flights, schedules
(web_search). Control the room with the home tools: lights on/off, colour, brightness, thermostat, the little
screen, music (play, pause, resume, stop, next song, volume). Timers, alarms, reminders, notes and messages use the
assist tools; when you set a timer also call show_countdown with the same seconds.
After a tool call, confirm in a few words what you did."""
    if agent:
        p += """
For long, multi-step jobs (building a website, game or app; researching and comparing several options; planning
a trip; writing a report) call delegate_task with a clear, complete task description. It runs in the background:
the system already tells the user "On it", so don't repeat that; keep chatting about other things. When its result arrives, tell the user the summary in one or two sentences. Never read a web
address aloud: if there is a page link, say "it's ready on the Pi's web page" and call show_message with
line1="Ready:" and line2 set to the address without "http://"."""
    return p


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper(),
               format="<green>{time:HH:mm:ss.SSS}</green> <level>{level: <7}</level> {message}")


class Assistant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.worker: PipelineWorker | None = None
        self.links: dict[str, McpLink] = {}
        self._standalone = False
        self._bg: list[asyncio.Task] = []

    # ------------------------------------------------------------------------------------------------- build
    async def build(self, transport=None) -> PipelineWorker:
        cfg = self.cfg
        benchmark = transport is not None
        if transport is None:
            from pipecat.transports.local.audio import LocalAudioTransportParams

            from assistant.audio import ResilientLocalAudioTransport
            aec = None
            if cfg.aec:
                from assistant.aec import AECFilter, EchoCanceller
                aec = EchoCanceller(cfg.audio_in_sample_rate, delay_ms=cfg.aec_delay_ms)
            transport = ResilientLocalAudioTransport(LocalAudioTransportParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=cfg.audio_in_sample_rate, audio_out_sample_rate=cfg.audio_out_sample_rate,
                input_device_index=cfg.input_device_index, output_device_index=cfg.output_device_index,
                audio_in_filter=AECFilter(aec) if aec else None), aec=aec)
            self._standalone = True
        use_agent = cfg.enable_agent and not benchmark       # agent results speak up unprompted: off in benchmarks

        urls = {"asr": cfg.asr_mcp_url, "tts": cfg.tts_mcp_url}
        if cfg.enable_tools:
            urls.update(cfg.tool_mcp_urls)
        if use_agent:
            urls["agent"] = cfg.agent_mcp_url
        if cfg.mcp_autostart:
            await asyncio.to_thread(ensure_servers, urls)

        self.links = {n: McpLink(u, n) for n, u in urls.items() if n in {"asr", "tts", "assist", "agent"}}
        await asyncio.gather(*(l.start() for l in self.links.values()))
        # info() also opens and keeps warm the servers' TLS connections to Deepgram (no cold first utterance)
        await asyncio.gather(self.links["asr"].call("info"), self.links["tts"].call("info"))
        stt = McpSTTService(self.links["asr"], tts_link=self.links["tts"], warm=os.environ.get("MCP_WARM", "1") != "0")
        tts = McpTTSService(self.links["tts"], sample_rate=cfg.audio_out_sample_rate)
        self._tts = tts
        llm, context, tools = await self.llm_and_context(use_agent)

        # ---- turn taking: Silero VAD + (Smart Turn v3 | silence timeout); barge-in on VAD (or after N words)
        vad = SileroVADAnalyzer(params=VADParams(confidence=cfg.vad_confidence, start_secs=cfg.vad_start_secs,
                                                 stop_secs=cfg.vad_stop_secs))
        start = ([MinWordsUserTurnStartStrategy(min_words=cfg.interrupt_min_words)] if cfg.interrupt_min_words
                 else [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()])
        stop = None if cfg.turn_stop == "smart" else [SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=cfg.speech_timeout)]
        user_agg, assistant_agg = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start, stop=stop)))

        stages = [transport.input(), VADProcessor(vad_analyzer=vad), stt]
        tracker = None
        if cfg.echo_filter:
            tracker, echo = make_echo_guard()
            self._echo_mem = tracker._m
            stages.append(echo)                      # drop transcripts of Iris's own voice (no hardware AEC)
        stages += [user_agg, llm] + ([tracker] if tracker is not None else []) + [tts, transport.output()]
        pipeline = Pipeline(stages + [assistant_agg])
        session = uuid.uuid4().hex[:8]
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True,
                                  audio_in_sample_rate=cfg.audio_in_sample_rate,
                                  audio_out_sample_rate=cfg.audio_out_sample_rate),
            observers=[LatencyObserver(cfg.latency_log, cfg.turns_log, session)])
        self._greet = cfg.greet and not benchmark
        logger.info(f"Iris ready: LLM {cfg.llm_provider}/{cfg.llm_model}, {len(tools)} tools, "
                    f"turn stop={cfg.turn_stop}, agent={'on' if use_agent else 'off'}, session {session}")
        return self.worker

    async def llm_and_context(self, use_agent: bool):
        """The talker LLM, its context and its tools (MCP servers + the async background agent)."""
        cfg = self.cfg
        tools = []
        if cfg.enable_tools:
            for name, url in cfg.tool_mcp_urls.items():
                try:
                    schema = await MCPClient(server_params=StreamableHttpParameters(url=url)).tools()
                    got = [t for t in schema.standard_tools if not t.name.startswith("internal_")]
                    tools += got
                    logger.info(f"{name} MCP: {len(got)} tools ({', '.join(t.name for t in got)})")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"{name} MCP server at {url} unavailable: {e!r}")
        if use_agent:
            agent = self.links["agent"]

            @tool_options(cancel_on_interruption=False, timeout_secs=900)
            async def delegate_task(params: FunctionCallParams, task: str):
                """Hand a long, multi-step job to the background agent (build a website/game/app, research and
                compare options, plan a trip, write a report). Returns a short summary when done; the conversation
                continues meanwhile.

                Args:
                    task: A complete, self-contained description of what to do and what to report back.
                """
                try:
                    res = await agent.call("run_task", {"task": task}, timeout=900)
                except Exception as e:  # noqa: BLE001
                    res = {"ok": False, "summary": f"The background agent failed: {e}"}
                await params.result_callback(res)

            tools.append(delegate_task)

        if cfg.llm_provider == "anthropic":
            from pipecat.services.anthropic.llm import AnthropicLLMService
            llm = AnthropicLLMService(api_key=cfg.anthropic_api_key, settings=AnthropicLLMService.Settings(
                model=cfg.llm_model, system_instruction=system_prompt(cfg, use_agent), max_tokens=400))
        else:
            from pipecat.services.openrouter.llm import OpenRouterLLMService
            llm = OpenRouterLLMService(api_key=cfg.openrouter_api_key, settings=OpenRouterLLMService.Settings(
                model=cfg.llm_model, system_instruction=system_prompt(cfg, use_agent), max_tokens=400,
                temperature=0.4, provider={"sort": cfg.openrouter_sort} if cfg.openrouter_sort else None))

        @llm.event_handler("on_function_calls_started")
        async def _filler(service, function_calls):
            tts = getattr(self, "_tts", None)
            names = {fc.function_name for fc in function_calls}
            if tts is None or not cfg.fillers:
                return
            pool = (DELEGATE_FILLERS if "delegate_task" in names else
                    LOOKUP_FILLERS if names & LOOKUP_TOOLS else ACTION_FILLERS)
            text = random.choice(pool)
            if getattr(self, "_echo_mem", None) is not None:
                self._echo_mem.add(text)
            await tts.queue_frame(TTSSpeakFrame(text, append_to_context=False))

        context = LLMContext(tools=ToolsSchema(standard_tools=tools)) if tools else LLMContext()
        return llm, context, tools

    # ------------------------------------------------------------------------------------------------- run
    async def _poll_due(self) -> None:
        """Speak timers/alarms/reminders that fired in the assist MCP server."""
        link = self.links.get("assist")
        if link is None:
            return
        while True:
            await asyncio.sleep(1.0)
            try:
                due = (await link.call("internal_pop_due", {}, timeout=5)).get("due", [])
            except Exception:  # noqa: BLE001
                continue
            for d in due:
                kind, label = d.get("kind", "timer"), d.get("label", "")
                if kind == "reminder":
                    msg = f"Reminder: {label}."
                elif kind == "alarm":
                    msg = f"It's time! Your {label or 'alarm'} is ringing." if label != "alarm" else "Your alarm is ringing."
                else:
                    msg = f"Your {label} timer is done." if label and label != "timer" else "Your timer is done."
                await self.worker.queue_frames([TTSSpeakFrame(msg)])

    async def run(self) -> None:
        runner = WorkerRunner(handle_sigint=self._standalone)
        await runner.add_workers(self.worker)
        self._bg.append(asyncio.create_task(self._poll_due()))
        if self._greet:
            await self.worker.queue_frames([TTSSpeakFrame(f"Hi, I'm {self.cfg.bot_name}. What can I do for you?")])
        try:
            await runner.run()
        finally:
            for t in self._bg:
                t.cancel()
            for l in self.links.values():
                await l.close()


def list_devices() -> None:
    import pyaudio
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        kinds = ("in " if d["maxInputChannels"] else "") + ("out" if d["maxOutputChannels"] else "")
        print(f"{i:3d}  {kinds:7s} {d['name']}  ({int(d['defaultSampleRate'])} Hz)")
    pa.terminate()


async def amain(args) -> None:
    cfg = load()
    setup_logging(cfg.log_level)
    if args.no_greet:
        cfg.greet = False
    problems = cfg.validate()
    if problems:
        for p in problems:
            logger.error(p)
        sys.exit("fix ME3/.env (see .env.example)")
    app = Assistant(cfg)
    await app.build()
    await app.run()


def main() -> None:
    ap = argparse.ArgumentParser(description="Iris full-duplex voice agent")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--no-greet", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        return list_devices()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass
    finally:
        stop_children()


if __name__ == "__main__":
    main()
