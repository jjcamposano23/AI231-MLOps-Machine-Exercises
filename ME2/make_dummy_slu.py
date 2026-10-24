"""Generate a tiny Option-B-SCHEMA synthetic dataset (manifest + wavs) so the SLU
pipeline can be tested before downloading the real 17k-clip dataset.

    python make_dummy_slu.py --out dummy_optionb --speakers 6 --variations 2

Writes <out>/manifest.csv with Option B's columns and <out>/<label>/*.wav. Audio
is synthetic (signature per intent AND slot value, so both heads can learn) --
NOT real speech. Point train_slu.py --root at <out> to smoke-test.
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf

from config import INTENTS, SLOT_SPEC, SR


def synth(ii, si, sr, dur, rng, spk_shift, noisy):
    N = int(sr * dur)
    t = np.arange(N) / sr
    base = 180 + 40 * ii                          # intent signature
    sig = np.zeros(N)
    for k, a in [(1, 1.0), (2, 0.5), (3, 0.33)]:
        sig += a * np.sin(2 * np.pi * base * k * spk_shift * t + rng.uniform(0, 6.28))
    sig *= 1 + 0.5 * np.sin(2 * np.pi * (2 + 3 * si) * t)   # slot signature
    if noisy:
        sig += 0.15 * rng.standard_normal(N)
    else:
        sig += 0.02 * rng.standard_normal(N)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return (0.3 * sig).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dummy_optionb")
    ap.add_argument("--speakers", type=int, default=6)
    ap.add_argument("--variations", type=int, default=2)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--dur", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # (intent, slot_value) combos: slotted intents expand to their 3 values.
    combos = []
    for intent in INTENTS:
        if intent in SLOT_SPEC:
            for v in SLOT_SPEC[intent][1]:
                combos.append((intent, v))
        else:
            combos.append((intent, ""))

    rows, total = [], 0
    for intent, sv in combos:
        slot_name = SLOT_SPEC[intent][0] if intent in SLOT_SPEC else ""
        label = intent if not sv else f"{intent}_{sv.replace(' ', '')}"
        folder = os.path.join(args.out, label)
        os.makedirs(folder, exist_ok=True)
        ii = INTENTS.index(intent)
        si = 0 if not sv else SLOT_SPEC[intent][1].index(sv) + 1
        for s in range(1, args.speakers + 1):
            split = "test" if s == args.speakers else "val" if s == args.speakers - 1 else "train"
            spk_shift = 1 + 0.04 * ((s - args.speakers / 2) / args.speakers)
            for v in range(1, args.variations + 1):
                for cond in ("clean", "noisy"):
                    rng = np.random.default_rng(
                        (args.seed, ii, si, s, v, cond).__hash__() & 0xFFFFFFFF)
                    y = synth(ii, si, args.sr, args.dur, rng, spk_shift, cond == "noisy")
                    fname = f"{label}_s{s}_v{v}_{cond}.wav"
                    sf.write(os.path.join(folder, fname), y, args.sr, subtype="PCM_16")
                    rows.append([f"{label}/{fname}", label, intent, f"s{s}", split,
                                 f"v{v}", cond, f"{intent} {sv}".strip(),
                                 slot_name, sv, round(len(y) / args.sr, 3)])
                    total += 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "label", "intent", "speaker", "split", "phrase_id",
                     "variant_id", "transcript", "slot", "slot_value", "duration_sec"])
        wr.writerows(rows)

    print(f"wrote {total} clips across {len(combos)} intent/slot combos "
          f"({args.speakers} speakers) -> {args.out}/")
    print(f"Now:  python train_slu.py --root {args.out} --epochs 3 --workers 0")


if __name__ == "__main__":
    main()
