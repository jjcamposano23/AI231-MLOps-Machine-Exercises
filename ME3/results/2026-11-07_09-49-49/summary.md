# Full-Duplex Voice Agent on a Raspberry Pi

Student 201435283 · run 2026-11-07_09-49-49 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 4.08 s | 5.65 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 1.36 s | 1.58 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 1.72 s | 3.02 s | 26 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.84 s | 1.05 s | 30 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.38 s | 0.38 s | 2 |
| WER | ASR accuracy on test set | < 15 % | 6.2 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 4 Model B Rev 1.4 · 8 GB
- Cooling: heatsink · peak 62 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 4 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0, ASR Deepgram nova-3 via asr-mcp, LLM openai/gpt-4.1-mini via OpenRouter, TTS Deepgram aura-2-pandora via tts-mcp, VAD Silero v5 + Smart Turn v3, AEC WebRTC APM

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate, usage) | 3186 | 5243 | 1197 → 5243 |
| Messages in context | 56 | 103 | 1 → 103 |
