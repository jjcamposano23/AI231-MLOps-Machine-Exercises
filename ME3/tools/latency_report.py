"""Latency report from live sessions (logs/latency.jsonl, written by assistant/observers.py).

Standardized method (course template section 5): one clock (the Pi's time.monotonic()), a timestamp at every stage
(VAD end -> ASR final -> LLM 1st token -> TTS 1st audio -> speaker), N phrases x M runs, mean + p95, JSON log.

    python tools/latency_report.py                 # all sessions
    python tools/latency_report.py --session abc   # one session
    python tools/latency_report.py --json out.json # also save the summary
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS_MS = {"e2e_ms": 4000, "asr_final_ms": 800, "llm_ttft_ms": 1500, "tts_ttfb_ms": 600, "barge_in_ms": 800}
ROWS = [("e2e_ms", "E2E latency", "end of speech -> audio out"),
        ("asr_final_ms", "ASR final", "end of speech -> transcript"),
        ("llm_ttft_ms", "LLM TTFT", "transcript -> first token"),
        ("tts_ttfb_ms", "TTS TTFB", "text -> first audio chunk"),
        ("barge_in_ms", "Barge-in", "speech -> playback stopped")]


def pct(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "logs" / "latency.jsonl"))
    ap.add_argument("--session", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = [json.loads(l) for l in Path(a.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.session:
        recs = [r for r in recs if r.get("session") == a.session]
    out = {}
    print(f"{len(recs)} records from {a.log}\n")
    print(f"{'Metric':<12} {'Definition':<30} {'Target':>7} {'Mean':>7} {'p95':>7} {'n':>4}")
    for key, name, definition in ROWS:
        vals = [r[key] for r in recs if r.get(key) is not None]
        mean = sum(vals) / len(vals) if vals else None
        p95 = pct(vals, 95)
        out[key] = {"metric": name, "definition": definition, "target_s": TARGETS_MS[key] / 1000,
                    "mean_s": None if mean is None else round(mean / 1000, 3),
                    "p95_s": None if p95 is None else round(p95 / 1000, 3), "n": len(vals)}
        f = lambda x: "-" if x is None else f"{x / 1000:.2f}s"  # noqa: E731
        print(f"{name:<12} {definition:<30} {TARGETS_MS[key] / 1000:>6.1f}s {f(mean):>7} {f(p95):>7} {len(vals):>4}")
    tools = [t["ms"] for r in recs for t in r.get("tools", []) if t.get("ms") is not None]
    if tools:
        print(f"\ntool calls: n={len(tools)}, mean {sum(tools) / len(tools) / 1000:.2f}s, "
              f"p95 {pct(tools, 95) / 1000:.2f}s")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
