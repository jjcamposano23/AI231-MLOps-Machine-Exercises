"""Measure OUT_OF_SCOPE rejection the way pi_assistant actually deploys it:
reject when the predicted intent is OUT_OF_SCOPE *or* the max confidence < thr.

Argmax-only benchmarks understate reject performance -- a from-scratch classifier
tends to output LOW confidence on genuinely out-of-scope speech, so the threshold
catches it. This sweeps thresholds so we can pick one for the live demo:
  OOS-recall   = fraction of out-of-scope clips correctly rejected
  OOS-prec     = of everything rejected, fraction that was truly out-of-scope
  in-scope-keep= fraction of real commands NOT wrongly rejected (want this high)

    python eval_reject.py --ckpt checkpoints_slu/gold-best.ckpt --hf-dataset /data/ai231
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import INTENT2IDX, OOS
from data import load_master, HFSLUDataset
from train_slu import SLUVCM

OOS_IDX = INTENT2IDX[OOS]


@torch.no_grad()
def collect(net, split):
    loader = DataLoader(HFSLUDataset(split, train=False), batch_size=64)
    confs, preds, trues = [], [], []
    for feat, yi, _ in loader:
        p = torch.softmax(net(feat)[0], 1)
        confs.append(p.max(1).values.numpy())
        preds.append(p.argmax(1).numpy())
        trues.append(yi.numpy())
    return np.concatenate(confs), np.concatenate(preds), np.concatenate(trues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf-dataset", dest="hf", required=True)
    args = ap.parse_args()
    net = SLUVCM.load_from_checkpoint(args.ckpt, map_location="cpu", strict=False).net.eval()

    ds = load_master(args.hf, "default")
    conf, pred, true = collect(net, ds["test"])
    is_oos = (true == OOS_IDX)
    try:                                   # add the dedicated negatives test set
        neg = load_master(args.hf, "synthetic_negatives")
        if "test" in neg:
            c2, p2, _ = collect(net, neg["test"])
            conf = np.concatenate([conf, c2])
            pred = np.concatenate([pred, p2])
            is_oos = np.concatenate([is_oos, np.ones(len(c2), bool)])
    except Exception as e:
        print(f"[warn] no synthetic_negatives test ({e})")

    n_oos, n_in = int(is_oos.sum()), int((~is_oos).sum())
    print(f"\nOOS clips: {n_oos} | in-scope clips: {n_in}\n")
    print(f"{'thr':>5} {'OOS-recall':>11} {'OOS-prec':>9} {'in-scope-keep':>14}")
    for thr in (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        reject = (pred == OOS_IDX) | (conf < thr)
        tp = int((reject & is_oos).sum())
        fp = int((reject & ~is_oos).sum())
        rec = tp / max(n_oos, 1)
        prec = tp / max(tp + fp, 1)
        keep = int((~reject & ~is_oos).sum()) / max(n_in, 1)
        print(f"{thr:5.2f} {rec:11.3f} {prec:9.3f} {keep:14.3f}")


if __name__ == "__main__":
    main()
