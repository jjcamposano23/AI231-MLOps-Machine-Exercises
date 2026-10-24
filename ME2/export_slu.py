"""Export the SLU multi-head VCM to ONNX and int8-quantize it (static QDQ).

The SLU model has TWO outputs (intent logits, slot logits), so this is separate
from export.py (single-output KWS).

    python export_slu.py --ckpt checkpoints_slu/slu-best.ckpt --root path/to/OptionB

Outputs slu.onnx (fp32) and slu.int8.onnx (static int8 -> QLinearConv, runs on
CPU / Pi / ESP via ORT). Calibration clips are sampled from the dataset manifest.

At inference the two outputs are:
    intent_logits: (B, 19)   -> argmax over config.INTENTS
    slot_logits:   (B, 19)   -> argmax over config.SLOT_VALUES  ('none' = no slot)
"""
import argparse
import os

import numpy as np
import torch

from config import INTENTS, SLOT_VALUES, N_MELS, N_FRAMES, SR
from features import wav_to_input
from model import count_params
from train_slu import SLUVCM


def load_slu_net(ckpt_path):
    net = SLUVCM.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False).net.eval()
    print(f"loaded SLUNet | params {count_params(net):,}")
    return net


def calibration_samples(root, manifest=None, n=64):
    """Sample {'mel': tensor} dicts from the dataset manifest for calibration."""
    samples = []
    try:
        import random
        import pandas as pd
        import soundfile as sf
        manifest = manifest or os.path.join(root, "manifest.csv")
        df = pd.read_csv(manifest)
        rows = df.sample(min(n, len(df)), random_state=0)
        for _, r in rows.iterrows():
            wav, sr = sf.read(os.path.join(root, r["path"]), always_2d=True)
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
    ap.add_argument("--root", default=None, help="OptionB folder for calibration clips")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--onnx", default="slu.onnx")
    ap.add_argument("--int8", default="slu.int8.onnx")
    args = ap.parse_args()

    net = load_slu_net(args.ckpt)

    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        net, dummy, args.onnx,
        input_names=["mel"], output_names=["intent_logits", "slot_logits"],
        dynamic_axes={"mel": {0: "batch", 3: "frames"},
                      "intent_logits": {0: "batch"}, "slot_logits": {0: "batch"}},
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

    samples = calibration_samples(args.root, args.manifest, args.calib_n)
    print(f"calibrating int8 on {len(samples)} clips...")

    prep = args.onnx.replace(".onnx", ".prep.onnx")
    quant_pre_process(args.onnx, prep)
    quantize_static(prep, args.int8, Reader(samples),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    os.remove(prep)
    print(f"wrote {args.int8}  ({os.path.getsize(args.int8)/1024:.1f} KB)")

    # --- verify both outputs load and decode ---
    import onnxruntime as ort
    sess = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"mel": np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)})
    names = [o.name for o in sess.get_outputs()]
    intent = INTENTS[int(np.argmax(out[names.index("intent_logits")]))]
    slot = SLOT_VALUES[int(np.argmax(out[names.index("slot_logits")]))]
    print(f"int8 runs on CPU | outputs {names} | sample decode -> "
          f"intent={intent}, slot={slot}")


if __name__ == "__main__":
    main()
