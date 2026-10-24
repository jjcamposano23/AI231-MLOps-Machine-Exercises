"""Plot learning curves from a Lightning CSVLogger metrics.csv.

    python plot_curves.py gold_metrics.csv

Train & val metrics are logged on separate rows, so we collapse per epoch
(mean ignores the NaNs). Saves learning_curves.png (intent / slot / joint).
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "gold_metrics.csv"
g = pd.read_csv(csv).groupby("epoch").mean(numeric_only=True)

panels = [("intent", "intent accuracy"), ("slot", "slot accuracy"), ("joint", "joint accuracy")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (key, title) in zip(axes, panels):
    for split, style in (("train", "--"), ("val", "-")):
        col = f"{split}_{key}"
        if col in g:
            ax.plot(g.index, g[col], style, marker=".", label=split)
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend()
fig.suptitle("Gold VCM — learning curves (train vs val)")
fig.tight_layout()
fig.savefig("learning_curves.png", dpi=150)
print("saved learning_curves.png")
