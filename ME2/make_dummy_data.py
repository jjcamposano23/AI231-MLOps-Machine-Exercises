"""Generate a SYNTHETIC data/ tree so the whole pipeline can be tested before
real recordings exist.

    python make_dummy_data.py --out data --speakers 12 --per-class-per-speaker 8

Each class gets a distinct synthetic signature (so the model can actually learn
it and the benchmark shows > chance accuracy -- proving the chain works). This is
NOT real speech; swap the folder contents for real clips when they arrive.

Filenames follow the real convention (spk<ID>_<label>_<idx>.wav) and data/_silence_
doubles as the noise folder for `train.py --noise data/_silence_`.
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf

from config import LABELS, SR


def synth(label, ci, sr, dur, rng, spk_shift):
    N = int(sr * dur)
    t = np.arange(N) / sr

    if label == "_silence_":
        return (0.01 * rng.standard_normal(N)).astype(np.float32)

    if label == "_unknown_":                    # broadband + random weak tones
        x = 0.6 * rng.standard_normal(N)
        for _ in range(3):
            f = rng.uniform(300, 3500)
            x += 0.3 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        return (0.3 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)

    # structured command class: harmonic stack + class-specific tremolo
    base = 180 + 55 * ci
    sig = np.zeros(N)
    for k, amp in [(1, 1.0), (2, 0.5), (3, 0.33)]:
        sig += amp * np.sin(2 * np.pi * base * k * spk_shift * t + rng.uniform(0, 6.28))
    sig *= 1 + 0.5 * np.sin(2 * np.pi * (2 + ci) * t)          # tremolo signature
    sig += 0.02 * rng.standard_normal(N)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    onset = rng.integers(0, int(0.2 * N))                      # small time jitter
    sig = np.roll(sig, onset)
    return (0.3 * sig).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--speakers", type=int, default=12)
    ap.add_argument("--per-class-per-speaker", type=int, default=8)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    total = 0
    for ci, label in enumerate(LABELS):
        folder = os.path.join(args.out, label)
        os.makedirs(folder, exist_ok=True)
        for s in range(1, args.speakers + 1):
            # per-speaker pitch shift, stable across that speaker's clips
            spk_shift = 1 + 0.04 * ((s - args.speakers / 2) / args.speakers)
            for i in range(args.per_class_per_speaker):
                rng = np.random.default_rng(
                    (args.seed, ci, s, i).__hash__() & 0xFFFFFFFF)
                y = synth(label, ci, args.sr, args.dur, rng, spk_shift)
                fname = f"spk{s:02d}_{label}_{i:04d}.wav"
                sf.write(os.path.join(folder, fname), y, args.sr, subtype="PCM_16")
                rows.append([fname, f"spk{s:02d}", label])
                total += 1

    with open(os.path.join(args.out, "metadata_dummy.csv"), "w",
              newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "speaker_id", "label"])
        wr.writerows(rows)

    print(f"wrote {total} synthetic clips across {len(LABELS)} classes "
          f"({args.speakers} speakers) -> {args.out}/")
    print("Now run:  python train.py --data data --noise data/_silence_ --epochs 3 --workers 0")


if __name__ == "__main__":
    main()
