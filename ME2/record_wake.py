"""Record wake-word training data ON THE PI (so it matches demo mic + room).

Positives: say your wake phrase ("Hey <name>") each time it prompts.
Negatives: say your COMMANDS, random talk, or stay quiet / let room noise in.
Saved as 16 kHz wav:  ~/wake_data/pos/*.wav  and  ~/wake_data/neg/*.wav

    python record_wake.py --name Nova --pos 80 --neg 160

Then copy ~/wake_data to your laptop and run train_wake.py there (torch won't run
on the Pi). Tip: record positives at different distances / volumes / tones, and
include your actual command phrases in the negatives so the wake word doesn't
fire on them.
"""
import argparse
import os
import time

import soundfile as sf
import webrtcvad

from config import SR
from pi_assistant import listen_once          # reuse the same VAD capture


def record_set(vad, n, folder, prompt):
    os.makedirs(folder, exist_ok=True)
    print(f"\n== {prompt} ==  (need {n}; Ctrl-C to stop early)")
    i = 0
    while i < n:
        res = listen_once(vad, timeout=15)
        if res is None:
            print("  (nothing heard, still waiting...)")
            continue
        wav, ms = res
        sf.write(os.path.join(folder, f"{int(time.time() * 1000)}.wav"), wav, SR)
        i += 1
        print(f"  [{i}/{n}]  {ms:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="my VCM")
    ap.add_argument("--pos", type=int, default=80)
    ap.add_argument("--neg", type=int, default=160)
    ap.add_argument("--out", default=os.path.expanduser("~/wake_data"))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--vad", type=int, default=2)
    args = ap.parse_args()

    if args.device is not None:
        import sounddevice as sd
        sd.default.device = (args.device, None)
    vad = webrtcvad.Vad(args.vad)

    input(f'\nPOSITIVES — say "Hey {args.name}" each prompt. Press Enter to start...')
    record_set(vad, args.pos, os.path.join(args.out, "pos"), f'say "Hey {args.name}"')
    input('\nNEGATIVES — say your COMMANDS / random words / stay quiet. Press Enter...')
    record_set(vad, args.neg, os.path.join(args.out, "neg"), "say anything that is NOT the wake word")
    print(f"\nDone. Copy '{args.out}' to your laptop, then run train_wake.py")


if __name__ == "__main__":
    main()
