# Full-Duplex Voice Agent on a Raspberry Pi

Student 201435283 · run 2026-11-07_02-02-45 · 2026-11-07

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 4.26 s | 5.43 s | 29 |
| ASR final | end of speech → transcript | < 0.80 s | 1.57 s | 1.70 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 1.54 s | 2.94 s | 30 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 1.02 s | 1.06 s | 30 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.44 s | 0.44 s | 1 |
| WER | ASR accuracy on test set | < 15 % | 1.8 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 4 Model B Rev 1.4 · 8 GB
- Cooling: heatsink · peak 46 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: Wi-Fi, 6 ms round trip
- Date: 2026-11-07
- Model versions: harness pipecat 1.15.0, ASR Deepgram nova-3 via asr-mcp, LLM openai/gpt-4.1-mini via OpenRouter, TTS Deepgram aura-2-pandora via tts-mcp, VAD Silero v5 + Smart Turn v3, AEC WebRTC APM

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate, usage) | 1756 | 3857 | 1199 → 2622 |
| Messages in context | 47 | 96 | 1 → 96 |
