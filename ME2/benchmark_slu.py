"""Validate an SLU VCM on the manifest test split + optional noisy-SNR robustness.

    # clean only:
    python benchmark_slu.py --root . --manifest combined.csv --ckpt checkpoints_slu/B-best.ckpt --tag B
    # + robustness sweep (clean, 20/10/5 dB SNR) using a noise folder:
    python benchmark_slu.py --root . --manifest combined.csv --ckpt checkpoints_slu/C-best.ckpt --tag C --noise ~/noise

Reports intent acc + macro-F1, slot acc, JOINT exact-match, COMMAND macro
(per-intent recall of fully-correct commands, averaged = the balanced headline),
a per-intent report + confusion matrix, efficiency (params, MACs, latency), and
-- if --noise -- COMMAND macro vs SNR. Saves metrics_<tag>.json for compare_runs.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import (INTENTS, CLIP_S, CLIP_LEN, N_MELS, N_FRAMES, SR,
                    INTENT2IDX, slot_index)
from features import logmel, pad_or_trim
from model import count_params
from train_slu import SLUVCM

SNRS = [20, 10, 5]


def count_macs(net):
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            oh, ow = out.shape[2], out.shape[3]
            total[0] += oh * ow * m.out_channels * (m.in_channels // m.groups) \
                * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in net.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        net(torch.randn(1, 1, N_MELS, N_FRAMES))
    for h in hs:
        h.remove()
    return total[0]


def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0, keepdim=True)                      # (1, T) mono


def load_noise_bank(folder):
    clips = []
    for f in glob.glob(os.path.join(folder, "*.wav")):
        try:
            clips.append(load_wav(f))
        except Exception:
            pass
    return clips


def rand_noise(clips):
    w = random.choice(clips)
    T = w.shape[-1]
    if T > CLIP_LEN:
        o = random.randint(0, T - CLIP_LEN)
        w = w[..., o:o + CLIP_LEN]
    return pad_or_trim(w)


@torch.no_grad()
def eval_split(net, rows, root, noise=None, snr=None, timed=False):
    yi_t, yi_p, ys_t, ys_p, lat = [], [], [], [], []
    for _, r in rows.iterrows():
        wav = pad_or_trim(load_wav(os.path.join(root, str(r["path"]))))
        if noise and snr is not None:
            nz = rand_noise(noise)
            scale = (wav.norm() / (nz.norm() + 1e-6)) * (10 ** (-snr / 20))
            wav = wav + nz * scale
        feat = logmel(wav).unsqueeze(0)                  # (1,1,40,F)
        t0 = time.perf_counter()
        li, ls = net(feat)
        if timed:
            lat.append((time.perf_counter() - t0) * 1000)
        yi_p.append(int(li.argmax(1))); yi_t.append(INTENT2IDX[str(r["intent"])])
        ys_p.append(int(ls.argmax(1))); ys_t.append(slot_index(r.get("slot_value")))
    return (np.array(yi_t), np.array(yi_p), np.array(ys_t), np.array(ys_p), np.array(lat))


def metrics_of(yi_t, yi_p, ys_t, ys_p):
    correct = (yi_t == yi_p) & (ys_t == ys_p)
    return dict(
        intent_acc=float((yi_t == yi_p).mean()),
        slot_acc=float((ys_t == ys_p).mean()),
        joint_acc=float(correct.mean()),
        intent_macro_f1=float(f1_score(yi_t, yi_p, average="macro")),
        command_macro=float(np.mean([correct[yi_t == c].mean() for c in np.unique(yi_t)])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--noise", default=None, help="noise folder -> robustness at 20/10/5 dB")
    ap.add_argument("--test-source", default="all", choices=["all", "real", "synthetic"],
                    help="filter test rows by the manifest `source` column")
    ap.add_argument("--hf-dataset", default=None, dest="hf", help="eval on the GOLD master dataset")
    ap.add_argument("--hf-split", default="test", help="HF split to evaluate (test / holdout)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    sfx = "" if args.test_source == "all" else f"_{args.test_source}"
    out_png = args.out or f"confusion_{args.tag}{sfx}.png"
    out_json = args.out_json or f"metrics_{args.tag}{sfx}.json"

    net = SLUVCM.load_from_checkpoint(args.ckpt, map_location="cpu", strict=False).net.eval()
    params, macs = count_params(net), count_macs(net)

    if args.hf:                                              # GOLD master dataset split
        from torch.utils.data import DataLoader
        from data import load_master, HFSLUDataset
        ds = load_master(args.hf)
        split = ds[args.hf_split] if args.hf_split in ds else ds["test"]
        print(f"[{args.tag}] HF split '{args.hf_split}': {len(split)} clips")
        loader = DataLoader(HFSLUDataset(split, train=False), batch_size=1)
        yi_t, yi_p, ys_t, ys_p, lat = [], [], [], [], []
        with torch.no_grad():
            for feat, yi, ys in loader:
                t0 = time.perf_counter(); li, ls = net(feat); lat.append((time.perf_counter()-t0)*1000)
                yi_p.append(int(li.argmax(1))); yi_t.append(int(yi))
                ys_p.append(int(ls.argmax(1))); ys_t.append(int(ys))
        yi_t, yi_p = np.array(yi_t), np.array(yi_p)
        ys_t, ys_p = np.array(ys_t), np.array(ys_p); lat = np.array(lat)
        n_test = len(split)
    else:                                                   # manifest path
        if not args.root:
            ap.error("provide --hf-dataset OR --root")
        import pandas as pd
        manifest = args.manifest or os.path.join(args.root, "manifest.csv")
        te = pd.read_csv(manifest)
        te = te[te["split"].astype(str).str.lower() == "test"]
        if args.test_source != "all" and "source" in te.columns:
            te = te[te["source"].astype(str).str.lower() == args.test_source]
        print(f"[{args.tag}] test clips: {len(te)} (source={args.test_source})")
        yi_t, yi_p, ys_t, ys_p, lat = eval_split(net, te, args.root, timed=True)
        n_test = len(te)

    m = metrics_of(yi_t, yi_p, ys_t, ys_p)

    metrics = {"run": args.tag, "n_test": int(n_test), **m,
               "params": int(params), "macs": int(macs),
               "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.percentile(lat, 95)),
               "rtf": float(lat.mean() / 1000 / CLIP_S),
               "robustness": {"clean": m["command_macro"]}}

    print(f"\n=== [{args.tag}] SLU metrics (clean test) ===")
    print(f"intent {m['intent_acc']:.4f} | macro-F1 {m['intent_macro_f1']:.4f} | slot {m['slot_acc']:.4f}")
    print(f"JOINT {m['joint_acc']:.4f} | COMMAND macro (headline) {m['command_macro']:.4f}")
    print(f"params {params:,} | MACs {macs/1e6:.2f} M | latency {lat.mean():.1f} ms | RTF {metrics['rtf']:.4f}")

    if args.noise and not args.hf:
        noise = load_noise_bank(args.noise)
        if noise:
            print(f"\n=== [{args.tag}] robustness (COMMAND macro vs SNR, noise={len(noise)} clips) ===")
            print(f"  clean : {m['command_macro']:.4f}")
            for snr in SNRS:
                r = metrics_of(*eval_split(net, te, args.root, noise=noise, snr=snr)[:4])
                metrics["robustness"][f"snr{snr}"] = r["command_macro"]
                print(f"  {snr:2d} dB : {r['command_macro']:.4f}")
        else:
            print(f"[warn] no .wav in {args.noise}; skipping robustness")

    print("\n--- per-intent (clean) ---")
    print(classification_report(yi_t, yi_p, labels=range(len(INTENTS)),
                                target_names=INTENTS, zero_division=0))

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {out_json}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(yi_t, yi_p, labels=range(len(INTENTS)))
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=INTENTS, yticklabels=INTENTS)
    plt.title(f"Run {args.tag} — intent confusion (clean test)")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
