"""Phase 1 demo host: mic -> VAD -> SLU model -> serial to the ESP32 panel.

    python infer_slu.py --onnx slu.int8.onnx --port COM6
    python infer_slu.py --onnx slu.int8.onnx --no-serial      # print only

The laptop runs the VCM (slu.int8.onnx, two heads: intent + slot). For each
spoken utterance it decodes intent + slot and sends one line to the ESP32:

    INTENT|slot_value\n     e.g.  TEMPERATURE|26 degrees   COLOR|red   LIGHT_ON|

Close the Arduino Serial Monitor first — only one program can hold COM6.
"""
import argparse
import collections
import datetime

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import webrtcvad

from config import INTENTS, SLOT_VALUES, SR, CONF_THRESHOLD, NO_SLOT
from features import wav_to_input

FRAME_MS = 30
FRAME_LEN = SR * FRAME_MS // 1000


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def classify(sess, in_name, out_names, wav):
    x = wav_to_input(wav)
    out = sess.run(None, {in_name: x})
    pi = softmax(out[out_names.index("intent_logits")][0])
    ps = softmax(out[out_names.index("slot_logits")][0])
    ii, si = int(pi.argmax()), int(ps.argmax())
    return INTENTS[ii], float(pi[ii]), SLOT_VALUES[si], float(ps[si])


def utterances(vad, ratio=0.9, silence_frames=8):
    """Yield float32 utterances from the mic using triggered VAD."""
    ring = collections.deque(maxlen=silence_frames)
    triggered, voiced = False, []
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=FRAME_LEN) as stream:
        print("listening... (Ctrl+C to stop)")
        while True:
            block, _ = stream.read(FRAME_LEN)
            frame = block[:, 0]
            speech = vad.is_speech(frame.tobytes(), SR)
            if not triggered:
                ring.append((frame, speech))
                if sum(s for _, s in ring) > ratio * ring.maxlen:
                    triggered = True
                    voiced = [f for f, _ in ring]; ring.clear()
            else:
                voiced.append(frame); ring.append((frame, speech))
                if sum(not s for _, s in ring) > ratio * ring.maxlen:
                    yield np.concatenate(voiced).astype(np.float32) / 32768.0
                    triggered, voiced = False, []; ring.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="slu.int8.onnx")
    ap.add_argument("--port", default=None, help="ESP32 serial port, e.g. COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3])
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    ser = None
    if not args.no_serial and args.port:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"serial -> {args.port}")

    vad = webrtcvad.Vad(args.vad)
    try:
        for wav in utterances(vad):
            intent, ci, slot, cs = classify(sess, in_name, out_names, wav)
            if ci < args.threshold:
                print(f"  (rejected: {intent} {ci:.2f})")
                continue
            if intent == "TIME":                       # laptop supplies the clock
                slot = datetime.datetime.now().strftime("%H:%M")
            elif slot == NO_SLOT:
                slot = ""
            line = f"{intent}|{slot}"
            print(f">> {line}   (intent {ci:.2f}, slot {cs:.2f})")
            if ser:
                ser.write((line + "\n").encode())
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if ser:
            ser.close()


if __name__ == "__main__":
    main()
