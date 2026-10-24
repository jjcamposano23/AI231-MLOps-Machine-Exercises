"""Record REAL command samples on the Pi mic to fine-tune the VCM to your voice.

Saves clips + manifest.csv (path,intent,slot_value) under ~/cmd_data, which
train_slu.py can fold into the gold training (heavily oversampled) so the model
adapts to your voice + this mic -- the fix for the real-mic gap.

    python record_commands.py --takes 10 --device 6
    # (edit COMMANDS below to a shorter set if you're tight on time)

Then: copy ~/cmd_data to your laptop, scp to the DGX, and on the DGX:
    python train_slu.py --hf-dataset /data/ai231 --real ~/cmd_data --real-oversample 60 \
        --noise ~/noise --music-as-noise ~/music --aug-voice --label-smoothing 0.0 \
        --arch dscnn --width 64 --epochs 50 --batch 128 --workers 8 --run-name goldreal
"""
import argparse
import csv
import os
import time

import soundfile as sf
import webrtcvad

from config import SR
from pi_assistant import listen_once

COMMANDS = [
    ("turn on the lights", "LIGHT_ON", ""),
    ("turn off the lights", "LIGHT_OFF", ""),
    ("play music", "PLAY_MUSIC", ""),
    ("stop", "STOP", ""),
    ("pause", "PAUSE", ""),
    ("next", "NEXT", ""),
    ("volume up", "VOLUME_UP", ""),
    ("volume down", "VOLUME_DOWN", ""),
    ("what's the weather", "WEATHER", ""),
    ("what time is it", "TIME", ""),
    ("call mom", "CALL", ""),
    ("send a message", "MESSAGE", ""),
    ("list my reminders", "LIST_REMINDERS", ""),
    ("set a timer for 30 seconds", "TIMER", "30 seconds"),
    ("set an alarm for 6 AM", "ALARM", "6 AM"),
    ("set the temperature to 22 degrees", "TEMPERATURE", "22 degrees"),
    ("set the brightness to 60 percent", "BRIGHTNESS", "60 percent"),
    ("change the color to red", "COLOR", "red"),
    ("remind me to drink water", "CREATE_REMINDER", "drink water"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("~/cmd_data"))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--vad", type=int, default=2)
    args = ap.parse_args()

    if args.device is not None:
        import sounddevice as sd
        sd.default.device = (args.device, None)
    vad = webrtcvad.Vad(args.vad)
    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    rows = []
    for prompt, intent, slot in COMMANDS:
        input(f'\n>>> say "{prompt}"  {args.takes}x  (vary tone/distance). Enter to start...')
        got = 0
        while got < args.takes:
            res = listen_once(vad, timeout=15)
            if res is None:
                print("  (nothing heard, waiting...)")
                continue
            wav, ms = res
            name = f"clips/{intent}_{int(time.time() * 1000)}.wav"
            sf.write(os.path.join(args.out, name), wav, SR)
            rows.append((name, intent, slot))
            got += 1
            print(f"  [{got}/{args.takes}]  {ms:.0f} ms")

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "intent", "slot_value"])
        w.writerows(rows)
    print(f"\ndone: {len(rows)} clips in {args.out}. Copy it to your laptop, scp to the DGX.")


if __name__ == "__main__":
    main()
