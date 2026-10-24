"""Guided recorder for the collective VCM dataset.

Enforces the format and naming the training pipeline expects, so every
classmate's contributions merge cleanly.

    # list your input devices first if needed:
    python record.py --list-devices

    # then record (pick a UNIQUE speaker id, e.g. your student number):
    python record.py --speaker 07 --takes 3 --device-name "USB mic"

Saves 16 kHz mono WAVs to  data/<label>/spk<ID>_<label>_<NNN>.wav
and appends one row per clip to  metadata.csv.

Controls per clip:  [Enter]=record, s=skip, q=quit   then   [Enter]=keep, r=redo
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import LABELS, SR

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prompts():
    with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as f:
        return json.load(f)["prompts"]


def next_index(folder, prefix):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix)] if os.path.isdir(folder) else []
    return len(existing) + 1


def record_clip(sr, dur):
    for n in (3, 2, 1):
        print(f"  recording in {n}...", end="\r", flush=True)
        time.sleep(0.6)
    print("  ● REC ...          ", end="\r", flush=True)
    audio = sd.rec(int(dur * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  done.               ")
    return audio[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", help="UNIQUE speaker id (e.g. student number)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--takes", type=int, default=3, help="takes per phrasing")
    ap.add_argument("--dur", type=float, default=2.0, help="clip length (seconds)")
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--classes", nargs="*", help="subset of labels to record")
    ap.add_argument("--device-name", default="", help="free-text mic label for metadata")
    ap.add_argument("--accent", default="", help="free-text accent tag for metadata")
    ap.add_argument("--input-device", type=int, default=None, help="sounddevice input index")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.speaker:
        ap.error("--speaker is required (use a unique id)")
    if args.input_device is not None:
        sd.default.device = (args.input_device, None)

    spk = f"spk{args.speaker}"
    prompts = load_prompts()
    labels = args.classes or LABELS

    print("=" * 60)
    print(f"Speaker: {spk} | {args.takes} takes x {args.dur}s @ {args.sr} Hz")
    consent = input("I consent to my voice being used for this class dataset "
                    "(anonymized). Type 'yes' to continue: ").strip().lower()
    if consent != "yes":
        print("No consent -> exiting."); return

    meta_path = os.path.join(HERE, "metadata.csv")
    new_meta = not os.path.exists(meta_path)
    meta = open(meta_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(meta)
    if new_meta:
        writer.writerow(["filename", "speaker_id", "label", "phrase",
                         "device", "accent", "sr", "duration_s", "timestamp"])

    try:
        for label in labels:
            folder = os.path.join(args.data, label)
            os.makedirs(folder, exist_ok=True)
            phrases = prompts.get(label, []) or ["(stay silent / ambient noise)"]
            print("\n" + "-" * 60 + f"\nCLASS: {label}")
            for phrase in phrases:
                for _ in range(args.takes):
                    print(f'\nSay: "{phrase}"')
                    cmd = input("  [Enter]=record  s=skip  q=quit > ").strip().lower()
                    if cmd == "q":
                        raise KeyboardInterrupt
                    if cmd == "s":
                        continue
                    while True:
                        audio = record_clip(args.sr, args.dur)
                        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                        print(f"  peak level: {peak:.2f}"
                              + ("  [!] silent?" if peak < 0.02 else "")
                              + ("  [!] clipping" if peak > 0.98 else ""))
                        again = input("  [Enter]=keep  r=redo  s=skip > ").strip().lower()
                        if again == "r":
                            continue
                        if again == "s":
                            audio = None
                        break
                    if audio is None:
                        continue
                    idx = next_index(folder, f"{spk}_{label}_")
                    fname = f"{spk}_{label}_{idx:04d}.wav"
                    sf.write(os.path.join(folder, fname), audio, args.sr, subtype="PCM_16")
                    writer.writerow([fname, spk, label, phrase, args.device_name,
                                     args.accent, args.sr, args.dur,
                                     datetime.now().isoformat(timespec="seconds")])
                    meta.flush()
                    print(f"  saved {label}/{fname}")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        meta.close()
        print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
