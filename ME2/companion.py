"""Laptop companion for the Pi VCM demo.

Receives UDP events "intent|slot" (or "WAKE") from pi_assistant on the Pi and,
on THIS laptop:
  * WAKE          -> wake chime (your Bluetooth speaker)
  * every command -> SPEAKS a response ("Turning on the lights", the weather, the
                     time, etc.) out your laptop's default output (the SonicGo)
  * PLAY_MUSIC / PAUSE / STOP / NEXT / VOLUME_* -> Spotify via media keys
  * CALL / MESSAGE -> your phone via adb (only if --phone is given; see notes)

Setup (laptop, me2env):
    pip install pynput sounddevice numpy pyttsx3
    python companion.py --port 9009 [--phone +639XXXXXXXXX]
Then on the Pi:  pi_assistant.py ... --companion 9009

Phone note: adb makes the phone place an OUTGOING call / open a pre-filled SMS
(Android + USB + USB-debugging required). To make the phone *ring* (incoming) you
need an external sender like Twilio -- ask if you want that instead.
"""
import argparse
import os
import shutil
import socket
import subprocess

import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller, Key

from config import response_text

_kb = Controller()
ADB = "adb"

MEDIA = {
    "PLAY_MUSIC": Key.media_play_pause, "PAUSE": Key.media_play_pause,
    "STOP": Key.media_play_pause, "NEXT": Key.media_next,
    "VOLUME_UP": Key.media_volume_up, "VOLUME_DOWN": Key.media_volume_down,
}
PHONE = None


def chime():
    sr = 44100
    t = np.linspace(0, 0.12, int(sr * 0.12), False)
    sig = np.concatenate([0.3 * np.sin(2 * np.pi * 784 * t),
                          0.3 * np.sin(2 * np.pi * 1175 * t)]).astype(np.float32)
    try:
        sd.play(sig, sr); sd.wait()
    except Exception as e:
        print("  [chime]", e)


def speak(text):
    if not text:
        return
    safe = str(text).replace("'", "")                  # PowerShell single-quote string
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Speak('{safe}')"], timeout=15)
    except Exception as e:
        print("  [tts]", e)


def tap(key):
    _kb.press(key); _kb.release(key)


def phone_call():
    if not PHONE:
        return
    try:
        subprocess.run([ADB, "shell", "am", "start", "-a", "android.intent.action.CALL",
                        "-d", f"tel:{PHONE}"], timeout=8)
    except Exception as e:
        print("  [phone call]", e)


def phone_text(body="Hello from Iris your voice assistant"):
    if not PHONE:
        return
    # one quoted string so the phone's shell keeps the spaced body as a single token
    cmd = (f"am start -a android.intent.action.SENDTO -d sms:{PHONE} "
           f"--es sms_body '{body}' --ez exit_on_sent true")
    try:
        subprocess.run([ADB, "shell", cmd], timeout=8)
    except Exception as e:
        print("  [phone text]", e)


def main():
    global PHONE, ADB
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--phone", default=None, help="phone number for CALL/MESSAGE via adb")
    ap.add_argument("--adb", default=None, help="full path to adb.exe if it isn't on PATH")
    args = ap.parse_args()
    PHONE = args.phone
    ADB = args.adb or shutil.which("adb") or "adb"
    if PHONE and not (shutil.which(ADB) or os.path.isfile(ADB)):
        print(f"  [warn] adb not found ('{ADB}') -- CALL/MESSAGE will fail. "
              f"Run from a terminal where `adb version` works, or pass --adb <path to adb.exe>.")

    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)       # dual-stack IPv4 + IPv6
    try:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    except Exception:
        pass
    sock.bind(("::", args.port))
    print(f"companion on UDP {args.port}  (chime + voice + Spotify"
          + (f" + phone {PHONE})" if PHONE else ")"))
    print("keep Spotify open; Ctrl-C to stop.")
    try:
        while True:
            data, _ = sock.recvfrom(1024)
            msg = data.decode(errors="ignore").strip()
            if msg == "WAKE":
                chime()
                continue
            intent, _, slot = msg.partition("|")
            print(">>", intent, slot)
            if intent in MEDIA:
                tap(MEDIA[intent])
            if intent == "CALL":
                phone_call()
            elif intent == "MESSAGE":
                phone_text()
            speak(response_text(intent, slot))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
