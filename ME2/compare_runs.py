"""Aggregate the A/B/C/D runs into a comparison table + figures.

    python compare_runs.py --runs A B C D

Reads metrics_<run>.json (from benchmark_slu.py) and each run's CSV-logger
metrics.csv (train + validation curves) and writes:
  comparison.csv         side-by-side metrics table
  comparison_bars.png    headline test metrics per run
  validation_curves.png  val JOINT + val intent accuracy vs epoch (all runs)
  confusion_grid.png     the four intent confusion matrices
Prints the best run by COMMAND macro (the balanced headline).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def metrics_csv(run, out_dir):
    c = sorted(glob.glob(os.path.join(out_dir, run, "version_*", "metrics.csv")))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--out-dir", default="checkpoints_slu", help="where CSV logs live")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1. summary table ----
    rows = [json.load(open(f"metrics_{r}.json")) for r in args.runs
            if os.path.exists(f"metrics_{r}.json")]
    if not rows:
        raise SystemExit("no metrics_<run>.json found — run benchmark_slu.py first")
    tab = pd.DataFrame(rows)
    cols = ["run", "intent_acc", "intent_macro_f1", "slot_acc", "joint_acc",
            "command_macro", "params", "macs", "latency_ms_mean", "rtf"]
    tab = tab[[c for c in cols if c in tab.columns]]
    tab.to_csv("comparison.csv", index=False)
    print(tab.to_string(index=False))
    best = tab.loc[tab["command_macro"].idxmax(), "run"]
    print(f"\nBEST by command_macro (headline): Run {best}")

    # ---- 2. headline bar chart ----
    show = [m for m in ["intent_acc", "joint_acc", "command_macro", "intent_macro_f1"]
            if m in tab.columns]
    x, w = np.arange(len(tab)), 0.2
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(show):
        plt.bar(x + i * w, tab[m], w, label=m)
    plt.xticks(x + w * (len(show) - 1) / 2, tab["run"])
    plt.ylim(0, 1); plt.ylabel("accuracy"); plt.legend()
    plt.title("A/B/C/D — test metrics"); plt.tight_layout()
    plt.savefig("comparison_bars.png", dpi=150); plt.close()

    # ---- robustness curves: command_macro vs SNR ----
    if any("robustness" in r for r in rows):
        order = ["clean", "snr20", "snr10", "snr5"]
        labs = ["clean", "20 dB", "10 dB", "5 dB"]
        plt.figure(figsize=(7, 5))
        for r in rows:
            rob = r.get("robustness", {})
            xs = [i for i, k in enumerate(order) if rob.get(k) is not None]
            ys = [rob[order[i]] for i in xs]
            if ys:
                plt.plot(xs, ys, marker="o", label=r["run"])
        plt.xticks(range(len(order)), labs); plt.ylim(0, 1)
        plt.xlabel("test condition"); plt.ylabel("command macro")
        plt.title("Robustness: command accuracy vs noise SNR"); plt.legend()
        plt.tight_layout(); plt.savefig("robustness_curves.png", dpi=150); plt.close()

    # ---- 3. train + validation curves ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for r in args.runs:
        p = metrics_csv(r, args.out_dir)
        if not p:
            continue
        df = pd.read_csv(p)
        if "val_joint" in df:
            v = df.dropna(subset=["val_joint"])
            ax[0].plot(v["epoch"], v["val_joint"], marker=".", label=f"{r} val")
        if "train_joint" in df:
            t = df.dropna(subset=["train_joint"])
            ax[0].plot(t["epoch"], t["train_joint"], ls="--", alpha=0.5, label=f"{r} train")
        if "val_intent" in df:
            vi = df.dropna(subset=["val_intent"])
            ax[1].plot(vi["epoch"], vi["val_intent"], marker=".", label=r)
    ax[0].set_title("JOINT accuracy (train dashed / val solid)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("joint acc"); ax[0].legend(fontsize=8)
    ax[1].set_title("validation intent accuracy")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("intent acc"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("validation_curves.png", dpi=150); plt.close()

    # ---- 4. confusion-matrix grid ----
    imgs = [f"confusion_{r}.png" for r in args.runs if os.path.exists(f"confusion_{r}.png")]
    if imgs:
        n = len(imgs); ncol = 2; nrow = (n + 1) // 2
        fig, axg = plt.subplots(nrow, ncol, figsize=(16, 7 * nrow))
        axg = np.array(axg).reshape(-1)
        for a, im in zip(axg, imgs):
            a.imshow(plt.imread(im)); a.axis("off")
        for a in axg[len(imgs):]:
            a.axis("off")
        plt.tight_layout(); plt.savefig("confusion_grid.png", dpi=120); plt.close()

    print("\nsaved: comparison.csv, comparison_bars.png, validation_curves.png, "
          "robustness_curves.png, confusion_grid.png")


if __name__ == "__main__":
    main()
