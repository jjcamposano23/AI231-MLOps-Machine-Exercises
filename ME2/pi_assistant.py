"""Always-listening VCM assistant for the live benchmark (Pi) or a laptop demo.

Runs the SLU VCM (ONNX) on microphone audio and LOGS one JSON line per decision to
~/vcm_benchmark/<student>_<timestamp>.log in the format Ailene's vcm-benchmark reads:
    {"intent": "TIMER", "slot": "30 seconds", "infer_ms": 4.2, "audio_ms": 1500.0}

Out-of-scope / low-confidence inputs are logged as intent "OUT_OF_SCOPE" (rejected).

    # on the Pi for the benchmark (wake-gated: each clip is [wake][pause][command]):
    python pi_assistant.py --onnx slu.int8.onnx --student <student-id> --wake
    # laptop demo (classify every utterance, optionally drive the ESP32 panel):
    python pi_assistant.py --onnx slu.int8.onnx --no-wake --port COM6
"""
import argparse
import collections
import datetime
import json
import os
import socket
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import webrtcvad

from config import (INTENTS, SLOT_VALUES, SR, CONF_THRESHOLD, OOS, NO_SLOT,
                    SLOT_SPEC, SLOT2IDX, INTENT2IDX)
from features import wav_to_input

FRAME_MS = 30
FRAME_LEN = SR * FRAME_MS // 1000
OOS_IDX = INTENT2IDX[OOS]


def classify(sess, in_name, out_names, wav, ignore_oos=False):
    out = sess.run(None, {in_name: wav_to_input(wav)})
    il = out[out_names.index("intent_logits")][0]
    sl = out[out_names.index("slot_logits")][0]
    if ignore_oos:
        il = il.copy(); il[OOS_IDX] = -1e9           # never let OUT_OF_SCOPE win the argmax
    pi = np.exp(il - il.max()); pi /= pi.sum()
    ps = np.exp(sl - sl.max()); ps /= ps.sum()
    ii = int(pi.argmax())
    intent = INTENTS[ii]
    if intent in SLOT_SPEC:          # slotted intent -> best VALID value (never 'none')
        idxs = [SLOT2IDX[v] for v in SLOT_SPEC[intent][1]]
        si = max(idxs, key=lambda k: ps[k])
    else:
        si = SLOT2IDX[NO_SLOT]
    return intent, float(pi[ii]), SLOT_VALUES[si], float(ps[si])


def wake_prob(sess, in_name, wav):
    out = sess.run(None, {in_name: wav_to_input(wav)})[0][0]
    e = np.exp(out - out.max()); p = e / e.sum()
    return float(p[1])                                   # class 1 = WAKE


def play_chime():
    sr = 16000
    t = np.linspace(0, 0.1, int(sr * 0.1), False)
    sig = np.concatenate([0.25 * np.sin(2 * np.pi * 784 * t),
                          0.25 * np.sin(2 * np.pi * 1175 * t)]).astype(np.float32)
    try:
        sd.play(sig, sr); sd.wait()
    except Exception:
        pass


def segments(vad, ratio=0.9, silence_frames=8):
    """Yield (utterance float32, audio_ms) from the mic using triggered VAD."""
    ring = collections.deque(maxlen=silence_frames)
    triggered, voiced = False, []
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME_LEN) as st:
        while True:
            block, _ = st.read(FRAME_LEN)
            frame = block[:, 0]
            sp = vad.is_speech(frame.tobytes(), SR)
            if not triggered:
                ring.append((frame, sp))
                if sum(s for _, s in ring) > ratio * ring.maxlen:
                    triggered = True; voiced = [f for f, _ in ring]; ring.clear()
            else:
                voiced.append(frame); ring.append((frame, sp))
                if sum(not s for _, s in ring) > ratio * ring.maxlen:
                    aud = np.concatenate(voiced).astype(np.float32) / 32768.0
                    yield aud, len(aud) / SR * 1000.0
                    triggered, voiced = False, []; ring.clear()


def listen_once(vad, timeout=6.0, ratio=0.9, silence_frames=8):
    """Capture ONE utterance from the mic (used after a wake). Returns
    (float32 audio, audio_ms), or None if nothing is spoken before `timeout`."""
    ring = collections.deque(maxlen=silence_frames)
    triggered, voiced = False, []
    start = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME_LEN) as st:
        while True:
            block, _ = st.read(FRAME_LEN)
            frame = block[:, 0]
            sp = vad.is_speech(frame.tobytes(), SR)
            if not triggered:
                ring.append((frame, sp))
                if sum(s for _, s in ring) > ratio * ring.maxlen:
                    triggered = True; voiced = [f for f, _ in ring]; ring.clear()
                elif time.time() - start > timeout:
                    return None
            else:
                voiced.append(frame); ring.append((frame, sp))
                if sum(not s for _, s in ring) > ratio * ring.maxlen:
                    aud = np.concatenate(voiced).astype(np.float32) / 32768.0
                    return aud, len(aud) / SR * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="slu.int8.onnx")
    ap.add_argument("--student", default="student")
    ap.add_argument("--wake", dest="wake", action="store_true",
                    help="wake-gated: 1st segment = wake word, next = command")
    ap.add_argument("--no-wake", dest="wake", action="store_false",
                    help="classify every utterance (simple demo)")
    ap.set_defaults(wake=True)
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--ignore-oos", dest="ignore_oos", action="store_true",
                    help="never predict OUT_OF_SCOPE; pick the best in-scope intent, reject only by --threshold")
    ap.add_argument("--logdir", default=os.path.expanduser("~/vcm_benchmark"))
    ap.add_argument("--port", default=None, help="optional ESP32 serial port for the panel")
    ap.add_argument("--gpio", action="store_true", help="drive LEDs on the Pi's own GPIO (stand-alone demo)")
    ap.add_argument("--wake-gpio", type=int, default=None, dest="wake_gpio",
                    help="KY-038 sound-sensor wake on this BCM GPIO pin (e.g. 6); takes priority over --wake")
    ap.add_argument("--oled", action="store_true", help="show the recognized command on an I2C SSD1306 OLED")
    ap.add_argument("--strip", type=int, default=None, help="WS2812 strip DIN on this BCM GPIO (e.g. 18); needs sudo")
    ap.add_argument("--tm1637", default=None, help="TM1637 4-digit display on 'CLK,DIO' BCM pins, e.g. 20,21")
    ap.add_argument("--wake-word", default=None, dest="wake_word",
                    help="ONNX wake-word model: say the wake phrase, hear a chime, then the command")
    ap.add_argument("--wake-threshold", type=float, default=0.7, dest="wake_threshold")
    ap.add_argument("--no-chime", dest="chime", action="store_false")
    ap.set_defaults(chime=True)
    ap.add_argument("--device", type=int, default=None, help="input device index (see sd.query_devices)")
    ap.add_argument("--companion", default=None,
                    help="send events (WAKE + intents) to companion.py on your laptop "
                         "for the chime + Spotify; value is 'port' (host auto from SSH) or 'host:port'")
    ap.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3])
    args = ap.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    wk = wk_in = None
    if args.wake_word:
        wk = ort.InferenceSession(args.wake_word, providers=["CPUExecutionProvider"])
        wk_in = wk.get_inputs()[0].name
        print(f"wake-word model: {args.wake_word}")

    os.makedirs(args.logdir, exist_ok=True)
    logpath = os.path.join(args.logdir, f"{args.student}_{datetime.datetime.now():%Y%m%d-%H%M%S}.log")
    log = open(logpath, "a", buffering=1)
    print(f"logging to {logpath}")

    ser = None
    if args.port:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=1)
        print(f"ESP32 panel -> {args.port}")

    panel = None
    if args.gpio:
        try:
            from pi_gpio import Panel
            panel = Panel()
            print("Pi GPIO panel active")
        except Exception as e:
            print(f"[warn] GPIO panel off ({e})")

    oled = None
    if args.oled:
        try:
            from pi_oled import OLED
            oled = OLED()
            print("OLED display active")
        except Exception as e:
            print(f"[warn] OLED off ({e})")

    strip = None
    if args.strip is not None:
        try:
            from pi_strip import StripPanel
            strip = StripPanel(pin=args.strip)
            print(f"WS2812 strip on GPIO{args.strip}")
        except Exception as e:
            print(f"[warn] strip off ({e})")

    tm = None
    if args.tm1637:
        try:
            from pi_tm1637 import TM1637
            clk, dio = (int(x) for x in args.tm1637.split(","))
            tm = TM1637(clk=clk, dio=dio)
            print(f"TM1637 on CLK{clk}/DIO{dio}")
        except Exception as e:
            print(f"[warn] TM1637 off ({e})")

    comp = csock = None
    if args.companion:
        if ":" in args.companion:
            chost, cport = args.companion.rsplit(":", 1)
        else:
            ssh = os.environ.get("SSH_CLIENT", "")
            chost = ssh.split()[0] if ssh else "127.0.0.1"
            cport = args.companion
        comp = (chost, int(cport))
        fam = socket.AF_INET6 if ":" in chost else socket.AF_INET
        csock = socket.socket(fam, socket.SOCK_DGRAM)
        print(f"companion -> {comp[0]}:{comp[1]}  (chime + Spotify on your laptop)")

    def tell_companion(msg):
        if csock:
            try:
                csock.sendto(msg.encode(), comp)
            except Exception:
                pass

    vad = webrtcvad.Vad(args.vad)

    def decide(wav, audio_ms):
        t0 = time.perf_counter()
        intent, ci, slot, cs = classify(sess, in_name, out_names, wav, ignore_oos=args.ignore_oos)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if intent == OOS or ci < args.threshold:            # reject
            intent, slot = OOS, ""
        elif slot == NO_SLOT:
            slot = ""
        rec = {"intent": intent, "slot": slot,
               "infer_ms": round(infer_ms, 1), "audio_ms": round(audio_ms, 1)}
        print(">>", json.dumps(rec), f"(intent {ci:.2f})")
        print(json.dumps(rec), file=log, flush=True)
        if ser and intent != OOS:
            ser.write((f"{intent}|{slot}\n").encode())
        if panel:
            panel.handle(intent, slot, ci)
        if oled:
            oled.handle(intent, slot, ci)
        if strip:
            strip.handle(intent, slot, ci)
        if tm:
            tm.handle(intent, slot, ci)
        tell_companion(f"{intent}|{slot}")               # laptop: narration + Spotify + phone

    gate = None
    try:
        if args.wake_word:                                  # spoken wake word -> chime -> command
            print("listening for the wake word...  (say it, wait for the chime, then your command)")
            armed = False
            for wav, audio_ms in segments(vad):
                if not armed:
                    p = wake_prob(wk, wk_in, wav)
                    print(f"  heard {audio_ms:.0f} ms | wake prob {p:.2f}")
                    if p >= args.wake_threshold:
                        print("  [wake!]")
                        print("\a", end="", flush=True)       # terminal bell fallback
                        tell_companion("WAKE")                # laptop plays the chime -> BT speaker
                        if oled:
                            oled.show("Listening...", "say a command")
                        if args.chime:
                            play_chime()                      # Pi speaker, if one is attached
                        armed = True
                else:
                    decide(wav, audio_ms)
                    armed = False
        elif args.wake_gpio is not None:                    # KY-038 hardware wake
            from pi_wake import WakeGate
            gate = WakeGate(pin=args.wake_gpio)
            print(f"KY-038 wake on GPIO{args.wake_gpio} - make a sound, then say a command...")
            while True:
                gate.wait_for_sound()
                print("  [wake] listening for command...")
                res = listen_once(vad)
                if res is None:
                    print("  (no command heard)")
                    continue
                decide(*res)
        elif args.wake:                                     # speech turn-taking wake
            print("listening...  (say anything to wake, pause, then your command)")
            awaiting_command = False
            for wav, audio_ms in segments(vad):
                if not awaiting_command:
                    awaiting_command = True
                    print("  [wake detected]")
                    continue
                decide(wav, audio_ms)
                awaiting_command = False
        else:                                               # always-on: classify every utterance
            print("listening...")
            for wav, audio_ms in segments(vad):
                decide(wav, audio_ms)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        log.close()
        if ser:
            ser.close()
        if panel:
            panel.close()
        if oled:
            oled.close()
        if strip:
            strip.close()
        if tm:
            tm.close()
        if gate:
            gate.close()


if __name__ == "__main__":
    main()
