// ME3 deck on Sir's 2-page template (machine_exercise_2page.pdf): same grid, header, section tags and boxes.
const pptxgen = require("pptxgenjs");
const OUT = process.argv[2] || "Iris-ME3-deck.pptx";

const NAVY = "16243F", ORANGE = "E8762D", TEAL = "1B7A8A", INK = "1F2937", MUTED = "6B7280";
const BOXF = "EEF0F5", BOXL = "C9D0DB", ROWALT = "F4F6F9", OK = "1E7F4F", BAD = "C2410C";
const CLOUDF = "FDF1E8", PIF = "E8F3F5";
const MONO = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 like the template
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.title = "Iris - Full-Duplex Voice Agent on a Raspberry Pi (ME3)";
pres.author = "Jomar Christian Camposano";

const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontFace: "Arial", color: INK, margin: 0 }, o));

function frame(s, title, right, page) {
  s.background = { color: "FFFFFF" };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 1.17, fill: { color: NAVY }, line: { color: NAVY } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.17, w: 13.333, h: 0.06, fill: { color: ORANGE }, line: { color: ORANGE } });
  T(s, "AI 231 · MACHINE EXERCISE 3 · JOMAR CHRISTIAN CAMPOSANO", { x: 0.67, y: 0.27, w: 9, h: 0.25, fontSize: 10.5, bold: true, color: ORANGE });
  T(s, title, { x: 0.67, y: 0.52, w: 9.2, h: 0.5, fontSize: 26, bold: true, color: "FFFFFF" });
  T(s, right, { x: 8.2, y: 0.58, w: 4.47, h: 0.35, fontSize: 12, italic: true, color: "AEB8C8", align: "right" });
  s.addShape(pres.shapes.LINE, { x: 0.67, y: 7.08, w: 12.0, h: 0, line: { color: "D1D5DB", width: 0.75 } });
  T(s, "Machine Exercise 3 - Full-Duplex Voice Agent on Raspberry Pi", { x: 0.67, y: 7.14, w: 4.6, h: 0.22, fontSize: 8.5, color: MUTED });
  T(s, "github.com/jjcamposano23/AI231-MLOps-Machine-Exercises (ME3)", { x: 4.6, y: 7.14, w: 4.6, h: 0.22, fontSize: 8.5, italic: true, color: MUTED, align: "center" });
  T(s, `2-page deck · ${page}/2`, { x: 9.67, y: 7.14, w: 3.0, h: 0.22, fontSize: 8.5, bold: true, color: MUTED, align: "right" });
}

function section(s, n, title, sub, x, y, w = 6) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.02, w: 0.26, h: 0.26, rectRadius: 0.04, fill: { color: ORANGE }, line: { color: ORANGE } });
  T(s, String(n), { x, y: y + 0.02, w: 0.26, h: 0.26, fontSize: 11, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  T(s, [{ text: title, options: { fontSize: 17, bold: true, color: NAVY } },
        { text: sub ? "    " + sub : "", options: { fontSize: 11, italic: true, color: "8A94A6" } }],
    { x: x + 0.36, y, w, h: 0.3, valign: "middle" });
}

// "- item" paragraphs with an orange dash, like the template's bullet lists
function dashList(items, size, gap = 3) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    runs.push({ text: "- ", options: { color: ORANGE, bold: true, fontSize: size } });
    const parts = Array.isArray(it) ? it : [{ text: it }];
    parts.forEach((p, j) => runs.push({ text: p.text, options: Object.assign({ fontSize: size, color: INK, paraSpaceAfter: gap },
      p.options || {}, (j === parts.length - 1 && !last) ? { breakLine: true } : {}) }));
  });
  return runs;
}

function node(s, x, y, w, h, lines, fill, line, size = 7.5) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: line, width: 1 } });
  const runs = lines.map((l, i) => ({ text: l, options: Object.assign({ fontSize: size, color: INK, bold: i === 0, breakLine: i < lines.length - 1 },
    i === 0 ? { color: NAVY } : {}) }));
  T(s, runs, { x: x + 0.04, y, w: w - 0.08, h, align: "center", valign: "middle" });
}

function arrow(s, x1, y, x2, color = NAVY, dash) {   // horizontal arrow from x1 to x2 (either direction)
  const left = Math.min(x1, x2), w = Math.abs(x2 - x1);
  const o = { x: left, y, w, h: 0, line: { color, width: 1.25 } };
  if (dash) o.line.dashType = "dash";
  if (x2 > x1) o.line.endArrowType = "triangle"; else o.line.beginArrowType = "triangle";
  s.addShape(pres.shapes.LINE, o);
}

// ================================================================== PAGE 1
{
  const s = pres.addSlide();
  frame(s, "Full-Duplex Voice Agent on a Raspberry Pi", "Iris · Pipecat + 5 own MCP servers on a Pi 4", 1);

  // ---- 1 SYSTEM DIAGRAM
  section(s, 1, "SYSTEM DIAGRAM", "the one diagram that explains your build", 0.67, 1.42);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.67, y: 1.8, w: 5.97, h: 2.2, rectRadius: 0.06, fill: { color: "F8F9FB" }, line: { color: BOXL, width: 1 } });
  // on-RPi boundary
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.77, y: 1.9, w: 3.98, h: 2.0, rectRadius: 0.05, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1.25, dashType: "dash" } });
  T(s, "ON THE RASPBERRY PI 4 (8 GB)", { x: 0.85, y: 1.93, w: 3.0, h: 0.18, fontSize: 7.5, bold: true, color: NAVY });
  T(s, "CLOUD APIs", { x: 4.88, y: 1.93, w: 1.66, h: 0.18, fontSize: 7.5, bold: true, color: ORANGE });

  // devices
  node(s, 0.85, 2.3, 0.78, 0.48, ["Mic", "USB headset"], PIF, TEAL);
  node(s, 0.85, 3.2, 0.78, 0.48, ["Speaker", "USB headset"], PIF, TEAL);
  // harness
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.79, y: 2.16, w: 1.6, h: 1.66, rectRadius: 0.05, fill: { color: NAVY }, line: { color: NAVY } });
  T(s, [
    { text: "Pipecat 1.15 harness", options: { bold: true, fontSize: 8, color: "FFFFFF", breakLine: true } },
    { text: "MCP client", options: { fontSize: 7, color: "AEB8C8", breakLine: true, paraSpaceAfter: 4 } },
    { text: "WebRTC AEC", options: { fontSize: 7.5, color: "FFFFFF", breakLine: true } },
    { text: "Silero VAD + Smart Turn v3", options: { fontSize: 7.5, color: "FFFFFF", breakLine: true } },
    { text: "STT → LLM context → TTS", options: { fontSize: 7.5, color: "FFFFFF", breakLine: true, paraSpaceAfter: 4 } },
    { text: "barge-in: stop TTS + LLM", options: { fontSize: 7.5, bold: true, color: ORANGE } },
  ], { x: 1.83, y: 2.18, w: 1.52, h: 1.62, align: "center", valign: "middle" });
  // own MCP servers
  node(s, 3.53, 2.15, 1.13, 0.36, ["asr-mcp :8101"], "FFFFFF", NAVY);
  T(s, "LLM: direct API", { x: 3.53, y: 2.5, w: 1.13, h: 0.24, fontSize: 7, italic: true, color: MUTED, align: "center", valign: "middle" });
  node(s, 3.53, 2.93, 1.13, 0.36, ["tts-mcp :8102"], "FFFFFF", NAVY);
  node(s, 3.53, 3.38, 1.13, 0.44, ["assist · home · agent", "MCP :8103-8105"], "FFFFFF", NAVY, 7);
  // cloud
  node(s, 4.88, 2.15, 1.66, 0.36, ["Deepgram nova-3"], CLOUDF, ORANGE);
  node(s, 4.88, 2.54, 1.66, 0.36, ["gpt-4.1-mini", "OpenRouter"], CLOUDF, ORANGE, 7);
  node(s, 4.88, 2.93, 1.66, 0.36, ["Deepgram Aura-2"], CLOUDF, ORANGE);
  node(s, 4.88, 3.38, 1.66, 0.44, ["Open-Meteo · Tavily", "Gemini 2.5 Flash (agent)"], CLOUDF, ORANGE, 7);
  // wires
  arrow(s, 1.63, 2.54, 1.79, TEAL);            // mic -> harness
  arrow(s, 1.79, 3.44, 1.63, TEAL);            // harness -> speaker
  arrow(s, 3.39, 2.33, 3.53);                  // harness -> asr
  arrow(s, 3.39, 2.72, 4.88, ORANGE, true);    // harness -> LLM API
  arrow(s, 3.39, 3.11, 3.53);                  // harness -> tts
  arrow(s, 3.39, 3.6, 3.53);                   // harness -> tools
  arrow(s, 4.66, 2.33, 4.88, ORANGE, true);
  arrow(s, 4.66, 3.11, 4.88, ORANGE, true);
  arrow(s, 4.66, 3.6, 4.88, ORANGE, true);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.67, y: 4.08, w: 5.97, h: 0.3, rectRadius: 0.05, fill: { color: NAVY }, line: { color: NAVY } });
  T(s, "mic -> AEC+VAD -> asr-mcp -> LLM(+MCP tools) -> tts-mcp -> speaker", { x: 0.67, y: 4.08, w: 5.97, h: 0.3, fontFace: MONO, fontSize: 9, color: "FFFFFF", align: "center", valign: "middle" });
  T(s, "barge-in: speech detected -> stop TTS + LLM -> listen (0.38 s)", { x: 0.67, y: 4.42, w: 5.97, h: 0.22, fontFace: MONO, fontSize: 9, bold: true, color: ORANGE, align: "center" });

  T(s, dashList([
    "Mic -> WebRTC AEC -> Silero VAD (on RPi) -> asr-mcp -> agent -> tts-mcp -> speaker",
    "Harness with MCP client: Pipecat 1.15 (streamable-HTTP MCP; Smart Turn v3 decides end of turn)",
    "LLM is a commercial API (gpt-4.1-mini via OpenRouter); every tool comes from MY MCP servers",
    "All 5 MCP servers built by me and running on the RPi: asr, tts, assist, home, agent",
    "Barge-in cancels TTS playback and the LLM stream: playback stops 0.38 s after speech starts",
    "Real tasks: weather, web search, timers/alarms on the TM1637, lights/OLED/music, background agent builds a web game",
    "On-RPi vs cloud: dashed box = runs on the Pi; orange = cloud APIs",
  ], 11, 5), { x: 6.96, y: 1.72, w: 5.71, h: 2.95, valign: "top" });

  // ---- 2 PROMPT
  section(s, 2, "PROMPT", "kept in the repo, versioned (assistant/main.py)", 0.67, 4.78);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.67, y: 5.12, w: 5.97, h: 1.8, rectRadius: 0.06, fill: { color: BOXF }, line: { color: BOXL, width: 1 } });
  const P = [
    'You are Iris, a calm, quick voice agent on a RPi;',
    '  light dry wit, calls the user "boss".',
    'Short, natural sentences; at most two per turn.',
    'If the user speaks, stop and listen; answer what',
    '  they just said, not the old topic.',
    'Confirm before costly or irreversible actions.',
    'Tools via MCP: weather, search, timers, home;',
    '  delegate_task for long jobs. Never read markdown',
    '  or URLs aloud.',
  ];
  T(s, [{ text: 'SYSTEM PROMPT (v3, PROMPT_VERSION="v3") - excerpt', options: { bold: true, color: TEAL, fontSize: 9.5, breakLine: true, paraSpaceAfter: 4 } }]
      .concat(P.map((l, i) => ({ text: l, options: { fontSize: 9.5, color: INK, breakLine: i < P.length - 1 } }))),
    { x: 0.85, y: 5.2, w: 5.7, h: 1.66, fontFace: MONO, valign: "top" });

  // ---- 3 CONTEXT ENGINEERING
  section(s, 3, "CONTEXT ENGINEERING", "", 6.96, 4.78);
  T(s, "GOES IN:", { x: 6.96, y: 5.14, w: 2.7, h: 0.2, fontSize: 10, bold: true, color: TEAL });
  T(s, dashList([
    "system prompt v3 + persona + 25 tool schemas (~1.2k tokens)",
    "ASR final per VAD / Smart Turn segment",
    "VAD turn events (user-turn aggregation)",
    "MCP tool results + async agent results",
    "memory: notes / reminders via tools (no RAG)",
  ], 9, 2), { x: 6.96, y: 5.38, w: 2.75, h: 1.6, valign: "top" });
  T(s, "MANAGED:", { x: 9.92, y: 5.14, w: 2.7, h: 0.2, fontSize: 10, bold: true, color: TEAL });
  T(s, dashList([
    "replies capped at 400 tokens; fillers spoken, kept out of context",
    "stable prefix (prompt + tools) never edited: cache-friendly TTFT",
    "idle -> listen -> think -> speak (Pipecat frames)",
    "on barge-in: drop queued TTS audio; echo filter drops self-heard text",
    [{ text: "logged per run: " }, { text: "1.2k -> 5.2k tokens", options: { bold: true } }, { text: " over 30 turns (no trim yet)" }],
  ], 9, 2), { x: 9.92, y: 5.38, w: 2.75, h: 1.6, valign: "top" });
}

// ================================================================== PAGE 2
{
  const s = pres.addSlide();
  frame(s, "MCP Services, Evaluation & Model Serving", "build + run servers on the RPi", 2);

  // ---- 4 MCP SERVICES (+6 model serving)
  section(s, 4, "MCP SERVICES", "+ section 6: model serving (one table, one truth)", 0.67, 1.38, 9);
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } });
  const rows = [
    ["ASR", "asr-mcp · streamable HTTP :8101", "transcribe(audio_b64) -> text · warm()", "Deepgram nova-3 (+ keyterms)", "RPi; final on VAD end; kept warm while user talks"],
    ["TTS", "tts-mcp · HTTP + SSE :8102", "synthesize_stream(text) -> 100 ms PCM chunks", "Deepgram Aura-2 (pandora)", "RPi; chunks as MCP progress; cancel on barge-in"],
    ["LLM", "Pipecat 1.15 (harness MCP client)", "function calling: 25 MCP tools + delegate_task", "OpenAI gpt-4.1-mini (OpenRouter)", "commercial API; streaming tokens; max 400"],
    ["VAD", "in-process (Pipecat)", "VADProcessor · Smart Turn v3 · WebRTC AEC", "Silero VAD v5 + Smart Turn v3", "RPi on-device, 16 kHz; AEC removes own voice"],
    ["Task", "assist-mcp · HTTP :8103", "time, weather, web_search, timer, alarm, reminders, notes", "Open-Meteo · Tavily", "RPi; due timers polled and spoken aloud"],
    ["Task", "home-mcp · HTTP :8104", "lights, colour, brightness, thermostat, OLED, timer, music", "n/a (GPIO / I2C / mpv)", "RPi; drives ME2 OLED + TM1637 displays"],
    ["Task", "agent-mcp · HTTP :8105", "run_task(task) -> spoken summary + page link", "Gemini 2.5 Flash (OpenRouter)", "RPi; async background job; pages on :8080"],
  ];
  s.addTable([[H("Component"), H("MCP server / transport"), H("Tools"), H("Model"), H("Served where / notes")]]
    .concat(rows.map((r, i) => r.map((c, j) => ({ text: c, options: Object.assign({ fill: { color: i % 2 ? ROWALT : "FFFFFF" } }, j === 0 ? { bold: true } : {}) })))),
    { x: 0.67, y: 1.74, w: 12.0, colW: [0.9, 2.2, 3.55, 2.25, 3.1], rowH: 0.235, fontFace: "Arial", fontSize: 9, color: INK,
      border: { type: "solid", pt: 0.75, color: BOXL }, valign: "middle", margin: [0, 0.06, 0, 0.06] });

  // ---- 5 EVALUATION
  section(s, 5, "EVALUATION & BENCHMARK", "standardized latency method (classwork) · King's fdbench", 0.67, 3.73, 9);
  T(s, dashList([
    "One clock: RPi monotonic; timestamp every stage (fdbench Pipecat marks)",
    "VAD end -> ASR final -> LLM 1st token -> TTS 1st audio -> speaker",
    "E2E = speaker - VAD end; barge-in = stop - speech start",
    "10 phrases x 3 runs = 30 turns; mean + p95; JSON log per run",
    "Full-Duplex-Bench v1/v1.5 (40 trials): stayed silent in 2/2 pauses, yielded to the interrupt, took 2/2 false turns",
  ], 10, 4), { x: 0.67, y: 4.08, w: 5.6, h: 1.38, valign: "top" });

  const m = [
    ["E2E latency", "end of speech -> audio out", "< 4.0 s", "4.08 s", "5.65 s", false],
    ["ASR final", "end of speech -> transcript", "< 0.8 s", "1.36 s", "1.58 s", false],
    ["LLM TTFT", "transcript -> first token", "< 1.5 s", "1.72 s", "3.02 s", false],
    ["TTS TTFB", "text -> first audio chunk", "< 0.6 s", "0.84 s", "1.05 s", false],
    ["Barge-in", "speech -> playback stopped", "< 0.8 s", "0.38 s", "0.38 s", true],
    ["WER", "ASR accuracy on test set (30)", "< 15 %", "6.2 %", "-", true],
  ];
  s.addTable([[H("Metric"), H("Definition"), H("Target"), H("Mean"), H("p95")]]
    .concat(m.map((r, i) => {
      const f = { fill: { color: i % 2 ? ROWALT : "FFFFFF" } };
      return [{ text: r[0], options: f }, { text: r[1], options: f }, { text: r[2], options: f },
              { text: r[3], options: Object.assign({ bold: true, color: r[5] ? OK : BAD }, f) }, { text: r[4], options: f }];
    })),
    { x: 6.55, y: 4.06, w: 6.12, colW: [1.1, 2.42, 0.85, 0.85, 0.9], rowH: 0.205, fontFace: "Arial", fontSize: 9, color: INK,
      border: { type: "solid", pt: 0.75, color: BOXL }, valign: "middle", margin: [0, 0.06, 0, 0.06] });

  // ---- DEMO / RESULTS / CHALLENGES boxes
  const by = 5.58, bh = 1.4;
  const box = (x, w, title) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w, h: bh, rectRadius: 0.06, fill: { color: BOXF }, line: { color: BOXL, width: 1 } });
    T(s, title, { x: x + 0.14, y: by + 0.08, w: w - 0.28, h: 0.2, fontSize: 10, bold: true, color: TEAL });
  };
  box(0.67, 3.35, "DEMO (2-3 min, no push-to-talk)");
  const D = [
    ['"Weather in Davao?"', " -> live get_weather MCP call"],
    ['"Set a 2-minute timer"', " -> countdown on the TM1637"],
    ["Long answer, I cut in", " -> Iris stops in ~0.4 s"],
    ['"Build me a flappy bird game"', " -> agent runs, we keep chatting"],
    ["Slow API:", ' "One sec..." filler + retry; game on :8080'],
  ];
  T(s, D.flatMap((d, i) => [
    { text: `${i + 1} `, options: { bold: true, color: ORANGE, fontSize: 8.5 } },
    { text: d[0], options: { bold: true, fontSize: 8.5 } },
    { text: d[1], options: Object.assign({ fontSize: 8.5, paraSpaceAfter: 3 }, i < D.length - 1 ? { breakLine: true } : {}) },
  ]), { x: 0.81, y: by + 0.33, w: 3.1, h: bh - 0.4, valign: "top" });

  box(4.17, 4.2, "RESULTS (mean s, 30 turns)");
  s.addChart(pres.charts.BAR, [
    { name: "Run 1 (02:02)", labels: ["E2E", "ASR", "LLM", "TTS"], values: [4.26, 1.57, 1.54, 1.02] },
    { name: "Run 2 + warm()", labels: ["E2E", "ASR", "LLM", "TTS"], values: [4.08, 1.36, 1.72, 0.84] },
    { name: "Target", labels: ["E2E", "ASR", "LLM", "TTS"], values: [4.0, 0.8, 1.5, 0.6] },
  ], { x: 4.22, y: by + 0.26, w: 2.3, h: 1.2, barDir: "col", barGapWidthPct: 60, chartColors: ["A3ACBA", NAVY, ORANGE],
       catAxisLabelFontSize: 7, valAxisLabelFontSize: 7, catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
       catAxisLabelFontFace: "Arial", valAxisLabelFontFace: "Arial", valAxisMinVal: 0, valAxisMaxVal: 4.5, valAxisMajorUnit: 1,
       valGridLine: { color: "D9DEE6", size: 0.5 }, catGridLine: { style: "none" },
       showLegend: true, legendPos: "b", legendFontSize: 6.5, legendFontFace: "Arial", legendColor: INK,
       objectName: "Latency chart: run 1 vs run 2 vs target" });
  T(s, [
    { text: "Raspberry Pi 4B · 8 GB RAM", options: { bold: true, breakLine: true } },
    { text: "heatsink · peak 62 °C · never throttled", options: { breakLine: true } },
    { text: "Wi-Fi to cloud; 4 ms link to the laptop", options: { breakLine: true } },
    { text: "2026-11-07 · fdbench 0.1.0", options: { breakLine: true } },
    { text: "pipecat 1.15 · nova-3 · gpt-4.1-mini", options: { breakLine: true } },
    { text: "Aura-2 · Silero v5 · Smart Turn v3", options: { breakLine: true } },
    { text: "E2E median 3.86 s (under target)", options: { bold: true, color: OK } },
  ], { x: 6.55, y: by + 0.34, w: 1.78, h: bh - 0.42, fontSize: 7.5, valign: "top", paraSpaceAfter: 1.5 });

  box(8.52, 4.15, "CHALLENGES & REFERENCES");
  const C = [
    ["Slow ASR/TTS after idle (Wi-Fi power save)", " -> warm() pings while the user talks", " -> ASR 1.57->1.36 s, TTS 1.02->0.84 s"],
    ["TTS waited for the whole clip", " -> stream 100 ms chunks as MCP progress", " -> TTFB 1-7 s -> 0.84 s"],
    ["Iris heard herself, cut herself off", " -> WebRTC AEC + echo text filter", " -> echo -57 dB (offline test)"],
    ["Open: answers TV / side talk (2/2)", " -> next: addressee check", ""],
  ];
  T(s, C.flatMap((c, i) => [
    { text: `${i + 1} `, options: { bold: true, color: ORANGE } },
    { text: c[0], options: { bold: true } },
    { text: c[1] },
    { text: c[2], options: Object.assign({ bold: true, color: i < 3 ? OK : BAD }, { breakLine: true }) },
  ]).concat([{ text: "refs: modelcontextprotocol.io · docs.pipecat.ai · fdbench · repo + logs: footer link", options: { color: MUTED, italic: true } }]),
    { x: 8.66, y: by + 0.31, w: 3.9, h: bh - 0.34, fontSize: 7.5, valign: "top", paraSpaceAfter: 1 });
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote " + f));
