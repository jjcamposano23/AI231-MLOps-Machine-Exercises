# Iris: a full-duplex voice AI agent on a Raspberry Pi (AI 231 ME3)

Iris is always listening and you can interrupt her mid-sentence (barge-in).
She runs on a Raspberry Pi 4 with **Pipecat** as the agent harness.
Her own **MCP servers** handle speech recognition (ASR), speech synthesis (TTS), the tools and a background agent, all running on the Pi.
Silero VAD (voice activity detection) and the Smart Turn v3 model run locally to decide when you've finished speaking.

```
 USB mic ─► Pipecat (on the Pi) ────────────────────────────────────────────────────► wired speaker
            │ Silero VAD + Smart Turn v3 (local ONNX)  · barge-in: user speech cancels LLM/TTS + playback
            │ McpSTTService ──MCP──► asr-mcp :8101  ──► Deepgram nova-3        (or SHARC/Groq Whisper)
            │ LLM (OpenRouter gpt-4.1-mini) + MCP tools:
            │     assist-mcp :8103  time · weather · web search · timers/alarms/reminders · notes · messages
            │     home-mcp   :8104  ME2 actuators: WS2812 lights/colour/brightness · OLED · TM1637 · music
            │     delegate_task (async) ─MCP─► agent-mcp :8105 (multi-step: research, build a web page/game → :8080)
            │ McpTTSService ──MCP──► tts-mcp :8102  ──► Deepgram Aura-2 (sentence by sentence)
            └ LatencyObserver → logs/latency.jsonl (one clock: the Pi's monotonic clock)
```

## Run

```bash
bash scripts/install_pi.sh                      # once, on the Pi
python -m assistant.main --list-devices         # pick INPUT_DEVICE_INDEX / OUTPUT_DEVICE_INDEX for .env
bash scripts/start_mcp.sh                       # optional: start the MCP servers yourself (home server with sudo for the strip)
python -m assistant.main                        # talk to Iris (it starts any MCP server that isn't running)
```

Keys in `.env`: `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`, `TAVILY_API_KEY` (see `.env.example`).

Test without a mic:
- `python tools/mcp_smoke.py` calls every MCP server.
- `python tools/text_chat.py "turn the lights blue"` runs the same LLM and tools from text.

## Full duplex: what's implemented
- **Listening while speaking.** The mic is never muted. Silero VAD runs on every 20 ms frame, even while Iris talks.
- **Barge-in.** When the user speaks, Iris cancels the LLM stream and the TTS and clears playback. With `INTERRUPT_MIN_WORDS=N`, she only stops after N transcribed words, which makes her robust to backchannels ("uh-huh") and speaker echo.
- **Pause handling.** Smart Turn v3 checks whether a turn is semantically complete, so a mid-sentence pause doesn't trigger a reply. `TURN_STOP=timeout` switches to a plain silence timer.
- **Background work.** `delegate_task` is an asynchronous tool call. The conversation continues, and the result is spoken when it's ready.
- **Timers.** Timers, alarms and reminders fire inside `assist-mcp`, and Iris announces them unprompted.

## Latency benchmark (standardized method)
1. **Live sessions.** `assistant/observers.py` timestamps every stage on the Pi's `time.monotonic()`: VAD end → ASR final → LLM first token → TTS first audio → speaker, plus barge-in. It writes one JSON line per turn. `python tools/latency_report.py` prints mean and p95 against the class targets: E2E 4.0 s, ASR 0.8 s, LLM TTFT 1.5 s, TTS TTFB 0.6 s, barge-in 0.8 s.
2. **Class benchmark.** King's [fd-benchmark](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking) is in `fd-benchmark/`. It covers Full-Duplex-Bench v1 and v1.5 (pause handling, turn-taking, interruption, backchannel, background speech), the missed and false turn-taking rates, and the template's §5 stage table (10 phrases × 3 runs). This project uses the course layout (`assistant/main.py` → `Assistant.build(transport=...)`), so the benchmark plugs in without an adapter.
   ```bash
   # on the Pi (assistant):
   .venv/bin/python fd-benchmark/launch_assistant.py --check        # must print READY
   echo "FDBENCH_URL=ws://LAPTOP-IP:8791" > fd-benchmark/.env
   .venv/bin/python fd-benchmark/launch_assistant.py
   # on the laptop (benchmark):  see fd-benchmark/docs/GUIDE.md steps 2-3, 8, 10-11
   .venv\Scripts\python -m fdbench run --student 2025xxxxx --rounds 1 --connect direct --where remote --judge-provider none --yes --recordings k
   ```
   In benchmark mode, the greeting and the background agent are off (as in the class runs). Results go to `fd-benchmark/runs/<student>/<date>/results.html`. The deck summary at the top has copy buttons for the slides.

### Results (Raspberry Pi 4B 8 GB, Wi-Fi, heatsink, 2026-11-07)
fdbench output for both runs is in [`results/`](results/). Run 2 adds the `warm()` pings (see Challenges in the deck).

| Metric | Target | Run 1 mean | Run 2 mean | Run 2 p95 |
|---|---|---|---|---|
| E2E (end of speech → audio out) | < 4.0 s | 4.26 s | 4.08 s (median 3.86 s) | 5.65 s |
| ASR final | < 0.8 s | 1.57 s | 1.36 s | 1.58 s |
| LLM TTFT | < 1.5 s | 1.54 s | 1.72 s | 3.02 s |
| TTS TTFB | < 0.6 s | 1.02 s | 0.84 s | 1.05 s |
| Barge-in | < 0.8 s | 0.44 s | 0.38 s | 0.38 s |
| WER | < 15 % | 1.8 % | 6.2 % | — |

Deck (2-page template): [`deck/Iris-ME3-deck.pdf`](deck/Iris-ME3-deck.pdf), built by `deck/build_deck.js`.

## Demo script (≈2 min, follows the class's agreed flow)
1. "Hi Iris, build me a simple flappy bird game." She says "On it…" and the background agent starts.
2. While it runs: "Turn the lights blue… actually make it purple." This interrupts her mid-reply; the strip changes colour.
3. "Set a timer for 30 seconds." The countdown shows on the TM1637.
4. "What's the weather in Davao—no wait, in Siquijor?"
5. "I want to check… *(pause)* …flights from Manila to Siquijor tomorrow." Smart Turn waits through the pause.
6. The timer goes off. Then "Your flappy bird game is ready on the Pi's web page." Open `http://<pi>:8080/flappy/`.
7. Say "uh-huh" while she's talking (backchannel), then talk to someone else in the room (false-turn test).

## ME2 → ME3 mapping (top-10 ME2 commands)
| ME2 intent | ME3 tool |
|---|---|
| LIGHT_ON/OFF | home `set_lights` |
| COLOR | home `set_light_color` |
| BRIGHTNESS | home `set_brightness` |
| TEMPERATURE | home `set_thermostat` |
| PLAY_MUSIC | home `play_music` |
| PAUSE | home `pause_music` |
| STOP | home `stop_music` |
| NEXT | home `next_song` |
| VOLUME_UP/DOWN | home `change_volume` |
| TIMER | assist `set_timer` + home `show_countdown` |
| ALARM | assist `set_alarm` |
| CREATE_REMINDER / LIST_REMINDERS | assist `create_reminder` / `list_reminders` |
| WEATHER | assist `get_weather` |
| TIME | assist `get_time` |
| MESSAGE / CALL | assist `send_message` (queued outbox) |
| OUT_OF_SCOPE | the LLM answers or uses `web_search`; long jobs go to `delegate_task` |
