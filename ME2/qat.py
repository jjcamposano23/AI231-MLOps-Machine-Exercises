"""Quantization-Aware Training (QAT) fine-tune for the DS-CNN student.

QAT inserts "fake-quant" ops during a short fine-tune so the weights adapt to
int8 rounding. You then int8-quantize as usual, but keep more accuracy than
post-training quantization (PTQ) alone.

    python qat.py --data data --noise data/_silence_ \
        --ckpt checkpoints/vcm-distilled.ckpt --epochs 10

Output:
    checkpoints/vcm-qat.ckpt  -- QAT-adapted fp32 weights, saved in the same
                                 VCM format as train.py. Deploy it exactly like
                                 any other checkpoint:
        python export.py    --ckpt checkpoints/vcm-qat.ckpt
        python benchmark.py --onnx vcm.int8.onnx        # measures int8 acc + size

We keep BN UNFUSED during QAT so the fine-tuned weights map cleanly back onto a
plain DS-CNN; export.py's int8 step then benefits from the QAT-adapted weights.

NOTE: eager-mode QAT here targets arch='dscnn'. For BC-ResNet use PTQ (export.py)
or FX-graph QAT -- its broadcast-add / SiLU need graph-mode handling.
"""
import argparse
import os

import torch
import torch.ao.quantization as tq
from torch import nn

from config import N_CLASSES
from data import build_loaders
from export import load_net
from model import DSCNN, count_params


class QuantDSCNN(nn.Module):
    """Wrap a DS-CNN with quant/dequant stubs for eager-mode QAT."""

    def __init__(self, net: DSCNN):
        super().__init__()
        self.quant = tq.QuantStub()
        self.net = net
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.net(self.quant(x)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _channels(net: DSCNN) -> int:
    return net.classifier.in_features         # recover c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise", default=None)
    ap.add_argument("--ckpt", required=True, help="trained/distilled DS-CNN checkpoint")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = load_net(args.ckpt)
    if not isinstance(net, DSCNN):
        raise SystemExit("qat.py supports arch='dscnn' only (see file header).")
    c = _channels(net)
    print(f"student params: {count_params(net):,} | c={c} | device={device}")

    tr, va, _, counts = build_loaders(args.data, args.noise, args.batch, args.workers)
    print(f"train {counts[0]}  val {counts[1]}  test {counts[2]}")
    print(f"fp32 val acc: {evaluate(net.to(device), va, device):.4f}")

    # --- prepare QAT (no fusion, so weight names stay plain DS-CNN) ---
    qmodel = QuantDSCNN(net).to(device)
    qmodel.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(qmodel.train(), inplace=True)

    # --- short fine-tune with fake-quant active ---
    opt = torch.optim.AdamW(qmodel.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(args.epochs):
        qmodel.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce(qmodel(x), y).backward()
            opt.step()
        print(f"epoch {epoch+1}/{args.epochs}  val_acc(fakequant) "
              f"{evaluate(qmodel, va, device):.4f}")

    # --- copy QAT-adapted weights back into a clean DS-CNN ---
    clean = DSCNN(n_classes=N_CLASSES, c=c)
    src = qmodel.net.state_dict()                 # plain names (weight/bn), + fq buffers
    clean_sd = clean.state_dict()
    matched = {k: src[k] for k in clean_sd
               if k in src and src[k].shape == clean_sd[k].shape}
    clean.load_state_dict(matched, strict=False)
    print(f"copied {len(matched)}/{len(clean_sd)} tensors into clean DS-CNN")

    os.makedirs(args.out, exist_ok=True)
    qat_ckpt = os.path.join(args.out, "vcm-qat.ckpt")
    torch.save({"state_dict": {f"net.{k}": v for k, v in clean.state_dict().items()},
                "hyper_parameters": {"arch": "dscnn", "width": c,
                                     "n_classes": N_CLASSES}},
               qat_ckpt)
    print(f"\nsaved {qat_ckpt}")
    print("Next:\n"
          f"  python export.py    --ckpt {qat_ckpt}\n"
          "  python benchmark.py --onnx vcm.int8.onnx")


if __name__ == "__main__":
    main()
