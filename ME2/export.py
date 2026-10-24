"""Export a trained checkpoint to ONNX and int8-quantize it (static QDQ).

Works for any checkpoint this repo produces:
    train.py           -> VCM        (network under 'net.')
    train_distill.py   -> DistillVCM (network under 'student.')
    qat.py             -> QAT-tuned VCM ('net.')

    python export.py --ckpt checkpoints/vcm-best.ckpt --data data

Outputs vcm.onnx (fp32) and vcm.int8.onnx (static int8, QLinearConv -> runs on
the CPU / Raspberry Pi). Static quantization needs a few real clips to calibrate
activation ranges; we sample them from --data (falls back to random if absent).

NOTE: CNNs must use STATIC (calibrated) quantization, not dynamic -- dynamic
quant emits ConvInteger, which ONNX Runtime's CPU provider does not implement.
"""
import argparse
import os

import numpy as np
import torch

from config import N_CLASSES, N_MELS, N_FRAMES, SR
from features import wav_to_input
from model import build_model, count_params


def load_net(ckpt_path):
    """Rebuild the deployable network from any repo checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    sd = ckpt["state_dict"]

    net = build_model(hp.get("arch", "dscnn"),
                      n_classes=hp.get("n_classes", N_CLASSES),
                      width=hp.get("width", 64))
    prefix = "student." if any(k.startswith("student.") for k in sd) else "net."
    net_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    net.load_state_dict(net_sd)
    net.eval()
    print(f"loaded {prefix[:-1]} | arch={hp.get('arch','dscnn')} "
          f"width={hp.get('width',64)} | params {count_params(net):,}")
    return net


def calibration_samples(data_root, n=64):
    """Yield up to n {'mel': tensor} dicts from real clips for calibration."""
    samples = []
    try:
        import random
        import soundfile as sf
        from data import scan_dataset
        items = scan_dataset(data_root)
        random.Random(0).shuffle(items)
        for path, _ in items[:n]:
            wav, sr = sf.read(path, always_2d=True)
            mono = wav.mean(axis=1).astype(np.float32)
            if sr != SR:
                import librosa
                mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
            samples.append({"mel": wav_to_input(mono)})
    except Exception as e:
        print(f"[warn] no calibration data ({e}); using random calibration")
    if not samples:
        samples = [{"mel": np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)}
                   for _ in range(8)]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data", help="clips to calibrate int8 ranges")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--onnx", default="vcm.onnx")
    ap.add_argument("--int8", default="vcm.int8.onnx")
    args = ap.parse_args()

    net = load_net(args.ckpt)

    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        net, dummy, args.onnx,
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "frames"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {args.onnx}  ({os.path.getsize(args.onnx)/1024:.1f} KB)")

    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, samples):
            self._it = iter(samples)

        def get_next(self):
            return next(self._it, None)

    samples = calibration_samples(args.data, args.calib_n)
    print(f"calibrating int8 on {len(samples)} clips...")

    prep = args.onnx.replace(".onnx", ".prep.onnx")
    quant_pre_process(args.onnx, prep)
    quantize_static(
        prep, args.int8, Reader(samples),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    )
    os.remove(prep)
    print(f"wrote {args.int8}  ({os.path.getsize(args.int8)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
