"""QC gate + manifest builder for the collective dataset.

Scans data/<label>/*.wav, checks format/duration/level and naming, optionally
normalizes to 16 kHz mono, then writes a manifest CSV and a summary.

    # check only (no changes):
    python validate_dataset.py --data data

    # check AND fix in place (convert to 16 kHz mono, downmix, resample):
    python validate_dataset.py --data data --fix

Flags per clip: bad sample rate, not mono, too short/long, clipping, (near) silent,
filename not spk<ID>_<label>_..., or label not matching its folder.
"""
import argparse
import csv
import os
import re

import librosa
import numpy as np
import soundfile as sf

from config import LABELS, SR

NAME_RE = re.compile(r"^spk(?P<spk>[^_]+)_(?P<label>.+)_\d+\.wav$")


def analyze(path, label, sr_target, min_dur, max_dur, fix):
    issues = []
    audio, sr = sf.read(path, always_2d=True)     # (N, C)
    ch = audio.shape[1]
    mono = audio.mean(axis=1)

    if sr != sr_target:
        issues.append(f"sr={sr}")
    if ch != 1:
        issues.append(f"channels={ch}")

    dur = len(mono) / sr
    if dur < min_dur:
        issues.append(f"short={dur:.2f}s")
    if dur > max_dur:
        issues.append(f"long={dur:.2f}s")

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    clip_frac = float(np.mean(np.abs(mono) > 0.99)) if mono.size else 0.0
    if peak < 0.02 or rms < 0.005:
        issues.append("silent")
    if clip_frac > 0.005:
        issues.append(f"clip={clip_frac:.1%}")

    m = NAME_RE.match(os.path.basename(path))
    spk = m.group("spk") if m else ""
    if not m:
        issues.append("badname")
    elif m.group("label") != label:
        issues.append("label!=folder")

    if fix and (sr != sr_target or ch != 1):
        y = mono.astype(np.float32)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sf.write(path, y, sr_target, subtype="PCM_16")
        sr, dur = sr_target, len(y) / sr_target
        issues.append("FIXED")

    return dict(sr=sr, dur=round(dur, 3), peak=round(peak, 3), rms=round(rms, 4),
                clip_frac=round(clip_frac, 4), speaker=spk,
                ok=all(i in ("FIXED",) for i in issues) or not issues,
                issues=";".join(issues))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--fix", action="store_true", help="normalize to 16kHz mono in place")
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--min-dur", type=float, default=0.4)
    ap.add_argument("--max-dur", type=float, default=3.0)
    args = ap.parse_args()

    rows, per_class, per_speaker, flagged = [], {}, {}, 0
    for label in LABELS:
        folder = os.path.join(args.data, label)
        if not os.path.isdir(folder):
            print(f"[warn] missing class folder: {folder}")
            per_class[label] = 0
            continue
        wavs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
        per_class[label] = len(wavs)
        for w in wavs:
            try:
                r = analyze(os.path.join(folder, w), label, args.sr,
                            args.min_dur, args.max_dur, args.fix)
            except Exception as e:                       # unreadable file
                r = dict(sr=0, dur=0, peak=0, rms=0, clip_frac=0, speaker="",
                         ok=False, issues=f"unreadable:{e}")
            r.update(filename=w, label=label)
            rows.append(r)
            per_speaker[r["speaker"]] = per_speaker.get(r["speaker"], 0) + 1
            hard = [i for i in r["issues"].split(";") if i and i != "FIXED"]
            if hard:
                flagged += 1
                print(f"  [flag] {label}/{w}: {r['issues']}")

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        cols = ["filename", "label", "speaker", "sr", "dur", "peak", "rms",
                "clip_frac", "ok", "issues"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    total = len(rows)
    print("\n=== per-class counts ===")
    for label in LABELS:
        n = per_class.get(label, 0)
        bar = "#" * min(40, n)
        print(f"  {label:16} {n:5}  {bar}")
    print(f"\nspeakers: {sum(1 for s in per_speaker if s)}  "
          f"(min/clip-count per speaker: "
          f"{min([v for k,v in per_speaker.items() if k], default=0)})")
    print(f"total clips: {total} | flagged: {flagged} | manifest -> {args.manifest}")
    if per_class:
        counts = [v for v in per_class.values() if v]
        if counts and max(counts) > 3 * max(1, min(counts)):
            print("[warn] class imbalance > 3x -- record more of the thin classes.")


if __name__ == "__main__":
    main()
