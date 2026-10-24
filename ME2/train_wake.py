"""Train a tiny FROM-SCRATCH wake-word model (WAKE vs background) -> wake.onnx.

Binary keyword spotter: class 1 = the wake phrase ("Hey <name>"), class 0 = anything
else. Reuses the same peak-normalized log-mel front-end as the main VCM, so the Pi's
NumPy features.py produces identical inputs. Run on your LAPTOP (torch); the Pi only
runs the exported wake.onnx.

    python train_wake.py --data wake_data --noise ~/noise --epochs 40
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from config import SR, N_MELS, N_FRAMES, CLIP_LEN
from features import logmel                     # torch log-mel (peak-norm) = Pi's numpy one


def load_wav(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w


class WakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc(self.c(x).flatten(1))


class DS(torch.utils.data.Dataset):
    def __init__(self, files, labels, noises, train):
        self.files, self.labels, self.noises, self.train = files, labels, noises, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        w = torch.from_numpy(load_wav(self.files[i])).float().unsqueeze(0)
        if self.train:
            w = torch.roll(w, random.randint(-1600, 1600), -1)
            if self.noises and random.random() < 0.6:
                nz = random.choice(self.noises)
                if nz.shape[-1] > CLIP_LEN:
                    o = random.randint(0, nz.shape[-1] - CLIP_LEN); nz = nz[..., o:o + CLIP_LEN]
                snr = random.uniform(5, 20)
                w = w + nz * (w.norm() / (nz.norm() + 1e-6)) * (10 ** (-snr / 20))
        return logmel(w), self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wake_data")
    ap.add_argument("--noise", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="wake.onnx")
    args = ap.parse_args()

    pos = glob.glob(os.path.join(args.data, "pos", "*.wav"))
    neg = glob.glob(os.path.join(args.data, "neg", "*.wav"))
    files, labels = pos + neg, [1] * len(pos) + [0] * len(neg)
    print(f"pos {len(pos)}  neg {len(neg)}")
    if not pos or not neg:
        raise SystemExit("need wavs in both wake_data/pos and wake_data/neg")

    noises = []
    if args.noise:
        for f in glob.glob(os.path.join(os.path.expanduser(args.noise), "*.wav")):
            try:
                noises.append(torch.from_numpy(load_wav(f)).float().unsqueeze(0))
            except Exception:
                pass

    idx = list(range(len(files))); random.seed(0); random.shuffle(idx)
    nval = max(2, len(idx) // 6); va = set(idx[:nval])
    mk = lambda keep, tr: DS([files[i] for i in idx if (i in va) != keep],
                             [labels[i] for i in idx if (i in va) != keep], noises, tr)
    tr = torch.utils.data.DataLoader(mk(False, True), batch_size=32, shuffle=True)
    vl = torch.utils.data.DataLoader(mk(True, False), batch_size=32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = WakeNet().to(dev)
    w = torch.tensor([1.0, max(1.0, len(neg) / max(1, len(pos)))], dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(args.epochs):
        net.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
        net.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for x, y in vl:
                p = net(x.to(dev)).argmax(1).cpu()
                tp += int(((p == 1) & (y == 1)).sum()); fp += int(((p == 1) & (y == 0)).sum())
                fn += int(((p == 0) & (y == 1)).sum())
        rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp)
        print(f"ep{ep + 1:2d}  wake recall {rec:.2f}  precision {prec:.2f}")

    net.eval().cpu()
    dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(net, dummy, args.out, input_names=["mel"], output_names=["logits"],
                      dynamic_axes={"mel": {0: "b", 3: "f"}, "logits": {0: "b"}}, opset_version=17)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
