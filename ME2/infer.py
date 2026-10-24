"""Real-time demo: mic -> VAD -> VCM -> intent -> Arduino over serial.

    python infer.py --onnx vcm.int8.onnx --port COM3
    python infer.py --onnx vcm.int8.onnx --no-serial     # print only, no Arduino

The laptop is the on-device VCM host (no cloud, no LLM). It listens on the mic,
segments each utterance with WebRTC VAD, classifies it, rejects low-confidence /
non-command audio, and sends a one-line command to the Arduino actuator.

NOTE: the core model predicts INTENT only. Numeric slots (timer seconds, dim %,
degrees) are sent as placeholders below; wire in the optional slot model to fill
real values. This is called out so the demo behaviour is honest.
"""
import argparse
import collections

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import webrtcvad

from config import LABELS, REJECT, SR, CONF_THRESHOLD
from features import wav_to_input

FRAME_MS = 30
FRAME_LEN = SR * FRAME_MS // 1000            # samples per VAD frame (480 @16k)

# Intent -> Arduino serial command (newline-terminated on send).
# Values marked "placeholder" await the optional slot model.
INTENT2CMD = {
    "play_music":      "MUSIC",
    "query_search":    "QUERY",
    "lights_onoff":    "LIGHTS_TOGGLE",
    "lights_dim":      "DIM:50",       # placeholder %
    "set_timer":       "TIMER:10",     # placeholder seconds
    "set_alarm":       "ALARM",
    "set_thermostat":  "THERMO:24",    # placeholder degrees C
    "media_control":   "MEDIA",
    "reminders":       "REMINDER",
    "calls_messaging": "CALL",
}


def classify(sess, in_name, wav_np, threshold):
    x = wav_to_input(wav_np)
    logits = sess.run(None, {in_name: x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(p.argmax())
    if p[i] < threshold:
        return "_unknown_", float(p[i])
    return LABELS[i], float(p[i])


def utterances(vad, threshold_ratio=0.9, silence_frames=8):
    """Yield float32 utterances from the mic using triggered VAD."""
    ring = collections.deque(maxlen=silence_frames)
    triggered, voiced = False, []
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=FRAME_LEN) as stream:
        print("listening... (Ctrl+C to stop)")
        while True:
            block, _ = stream.read(FRAME_LEN)
            frame = block[:, 0]
            is_speech = vad.is_speech(frame.tobytes(), SR)
            if not triggered:
                ring.append((frame, is_speech))
                if sum(s for _, s in ring) > threshold_ratio * ring.maxlen:
                    triggered = True
                    voiced = [f for f, _ in ring]
                    ring.clear()
            else:
                voiced.append(frame)
                ring.append((frame, is_speech))
                if sum(not s for _, s in ring) > threshold_ratio * ring.maxlen:
                    samples = np.concatenate(voiced).astype(np.float32) / 32768.0
                    yield samples
                    triggered, voiced = False, []
                    ring.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="vcm.int8.onnx")
    ap.add_argument("--port", default=None, help="Arduino serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3],
                    help="WebRTC VAD aggressiveness (higher = stricter)")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    ser = None
    if not args.no_serial and args.port:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"serial -> {args.port} @ {args.baud}")

    vad = webrtcvad.Vad(args.vad)
    try:
        for wav in utterances(vad):
            intent, conf = classify(sess, in_name, wav, args.threshold)
            if intent in REJECT:
                print(f"  (rejected: {intent} {conf:.2f})")
                continue
            cmd = INTENT2CMD.get(intent, "UNKNOWN")
            print(f">> {intent} ({conf:.2f}) -> {cmd}")
            if ser:
                ser.write((cmd + "\n").encode())
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if ser:
            ser.close()


if __name__ == "__main__":
    main()
