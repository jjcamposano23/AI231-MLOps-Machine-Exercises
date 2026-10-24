"""Validate a VCM against the held-out speaker-independent test split.

    python benchmark.py --data data --onnx vcm.int8.onnx

Reports: accuracy, macro-F1, per-class report, confusion matrix (PNG),
false-accept rate on _unknown_/_silence_, latency, model size.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torchaudio
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from config import LABELS, REJECT, SR, CONF_THRESHOLD
from data import make_splits, scan_dataset
from features import wav_to_input


def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0).numpy().astype(np.float32)       # mono float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--onnx", default="vcm.int8.onnx")
    ap.add_argument("--out", default="confusion_matrix.png")
    args = ap.parse_args()

    _, _, test_items = make_splits(scan_dataset(args.data), by_speaker=True)
    print(f"test clips: {len(test_items)}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    y_true, y_pred, latencies = [], [], []
    for path, label in test_items:
        x = wav_to_input(load_wav(path))
        t0 = time.perf_counter()
        logits = sess.run(None, {in_name: x})[0][0]
        latencies.append((time.perf_counter() - t0) * 1000)   # ms
        p = np.exp(logits - logits.max()); p /= p.sum()
        pred = int(p.argmax())
        if p[pred] < CONF_THRESHOLD:
            pred = LABELS.index("_unknown_")
        y_true.append(label); y_pred.append(pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\n=== Accuracy / F1 ===")
    print(f"accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"macro-F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\n" + classification_report(y_true, y_pred, labels=range(len(LABELS)),
                                        target_names=LABELS, zero_division=0))

    # False-accept rate: reject-class clips that were classified as a real command.
    reject_idx = {LABELS.index(n) for n in REJECT}
    mask = np.isin(y_true, list(reject_idx))
    if mask.any():
        fired = ~np.isin(y_pred[mask], list(reject_idx))
        print(f"false-accept rate (non-commands firing): {fired.mean():.4f}")

    lat = np.array(latencies)
    print("\n=== Efficiency (this machine) ===")
    print(f"latency  : mean {lat.mean():.1f} ms | p95 {np.percentile(lat,95):.1f} ms")
    print(f"RTF      : {lat.mean()/1000/2.0:.4f}  (compute / 2.0s audio)")
    print(f"model    : {os.path.getsize(args.onnx)/1024:.1f} KB")

    # Confusion matrix figure.
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
