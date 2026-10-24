"""Train the SLU multi-head VCM (Option B) FROM SCRATCH.

Shared backbone -> intent head (19) + slot head (19 incl 'none'). Loss = the two
cross-entropies. Tracks intent / slot / joint (both-correct) accuracy. A CSV
logger records per-epoch train AND validation metrics for plotting curves.

Knobs for the A/B/C/D experiment runs:
  --train-source {all,synthetic,real}   filter TRAIN rows by manifest `source`
  --noise <folder>                      mix ambient noise into training
  --music-as-noise <folder>             mix music (quieter) into training
  --class-weight                        inverse-frequency class weights (lifts macro-F1)
  --run-name <name>                     names the checkpoint + CSV log dir

    python train_slu.py --root . --manifest combined.csv --run-name B \
        --arch dscnn --width 64 --epochs 60
"""
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config import N_INTENTS, N_SLOTS, INTENT2IDX, slot_index
from data import build_loaders_manifest
from model import build_slu_model, count_params


class SLUVCM(pl.LightningModule):
    def __init__(self, arch="dscnn", width=64, n_intents=N_INTENTS, n_slots=N_SLOTS,
                 slot_weight=1.0, intent_cw=None, slot_cw=None, lr=3e-3, max_epochs=60,
                 label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["intent_cw", "slot_cw"])
        self.net = build_slu_model(arch, width=width, n_intents=n_intents, n_slots=n_slots)
        self.ce_i = torch.nn.CrossEntropyLoss(weight=intent_cw, label_smoothing=label_smoothing)
        self.ce_s = torch.nn.CrossEntropyLoss(weight=slot_cw, label_smoothing=label_smoothing)
        self.intent_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_intents)
        self.slot_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_slots)

    def forward(self, x):
        return self.net(x)

    def _shared(self, batch, tag):
        x, yi, ys = batch
        li, ls = self.net(x)
        loss = self.ce_i(li, yi) + self.hparams.slot_weight * self.ce_s(ls, ys)
        pi, ps = li.argmax(1), ls.argmax(1)
        joint = ((pi == yi) & (ps == ys)).float().mean()
        self.log(f"{tag}_loss", loss, prog_bar=(tag == "train"), on_epoch=True, on_step=False)
        self.log(f"{tag}_intent", self.intent_acc(li, yi), prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{tag}_slot", self.slot_acc(ls, ys), prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{tag}_joint", joint, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, b, _):
        return self._shared(b, "train")

    def validation_step(self, b, _):
        return self._shared(b, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return [opt], [sch]


def class_weights(manifest, train_source):
    """Inverse-frequency class weights for intent + slot heads (from TRAIN rows)."""
    df = pd.read_csv(manifest)
    tr = df[df["split"].astype(str).str.lower() == "train"]
    if train_source != "all" and "source" in df.columns:
        tr = tr[tr["source"].astype(str).str.lower() == train_source]
    ii = tr["intent"].map(INTENT2IDX).dropna().astype(int).to_numpy()
    si = tr["slot_value"].fillna("").map(slot_index).astype(int).to_numpy()

    def inv_freq(idx, n):
        c = np.bincount(idx, minlength=n).astype(float)
        w = c.sum() / (n * np.maximum(c, 1.0))          # rare classes -> larger weight
        return torch.tensor(w / w.mean(), dtype=torch.float32)   # normalize to mean 1

    return inv_freq(ii, N_INTENTS), inv_freq(si, N_SLOTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="manifest-mode: folder with manifest.csv")
    ap.add_argument("--manifest", default=None, help="defaults to <root>/manifest.csv")
    ap.add_argument("--hf-dataset", default=None, dest="hf",
                    help="GOLD master dataset: local dir (/data/ai231/...) or HF name")
    ap.add_argument("--supplemental", default=None, help="supplemental dataset (train-only)")
    ap.add_argument("--neg-oversample", type=int, default=1, dest="neg_oversample",
                    help="repeat synthetic_negatives (OUT_OF_SCOPE) N times to fix reject-class imbalance")
    ap.add_argument("--real", default=None, help="folder of YOUR real mic recordings (manifest.csv) to adapt to")
    ap.add_argument("--real-oversample", type=int, default=60, dest="real_oversample",
                    help="repeat the real recordings N times so they carry weight against the gold data")
    ap.add_argument("--train-source", default="all", choices=["all", "synthetic", "real"])
    ap.add_argument("--noise", default=None, help="folder of noise .wav")
    ap.add_argument("--music-as-noise", default=None, dest="music", help="folder of music .wav")
    ap.add_argument("--aug-voice", action="store_true",
                    help="pitch + speed perturbation (simulate other voices -> unseen-speaker robustness)")
    ap.add_argument("--class-weight", action="store_true", help="inverse-freq class weights (intent + slot)")
    ap.add_argument("--class-weight-intent-only", action="store_true", dest="cw_intent_only",
                    help="class-weight the intent head only (slot head unweighted)")
    ap.add_argument("--arch", default="dscnn", choices=["dscnn", "bcresnet"])
    ap.add_argument("--width", type=float, default=64)
    ap.add_argument("--slot-weight", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1, dest="label_smoothing",
                    help="0.0 = crisper/higher confidence (easier thresholding for the live demo)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--out", default="checkpoints_slu")
    args = ap.parse_args()

    intent_cw = slot_cw = None
    if args.hf:                                             # GOLD master dataset path
        from data import build_loaders_hf
        tr, va, counts = build_loaders_hf(args.hf, args.batch, args.workers, noise=args.noise,
                                          music=args.music, aug_voice=args.aug_voice,
                                          supplemental=args.supplemental,
                                          neg_oversample=args.neg_oversample,
                                          real_root=args.real, real_oversample=args.real_oversample)
        print(f"[{args.run_name}] HF master dataset  train {counts[0]}  val {counts[1]}")
        if args.class_weight or args.cw_intent_only:
            print(f"[{args.run_name}] (class-weighting skipped for --hf-dataset path)")
    else:                                                  # manifest path
        if not args.root:
            ap.error("provide --hf-dataset OR --root")
        manifest = args.manifest or f"{args.root}/manifest.csv"
        tr, va, _, counts = build_loaders_manifest(args.root, manifest, args.batch,
                                                   args.workers, noise=args.noise,
                                                   music=args.music, train_source=args.train_source,
                                                   aug_voice=args.aug_voice)
        print(f"[{args.run_name}] source={args.train_source}  train {counts[0]}  val {counts[1]}  test {counts[2]}")
        if args.class_weight or args.cw_intent_only:
            intent_cw, slot_cw = class_weights(manifest, args.train_source)
            if args.cw_intent_only:
                slot_cw = None
            print(f"[{args.run_name}] class weighting: intent"
                  f"{' only' if args.cw_intent_only else ' + slot'} "
                  f"(intent w {intent_cw.min():.2f}-{intent_cw.max():.2f})")

    model = SLUVCM(arch=args.arch, width=args.width, slot_weight=args.slot_weight,
                   intent_cw=intent_cw, slot_cw=slot_cw, lr=args.lr, max_epochs=args.epochs,
                   label_smoothing=args.label_smoothing)
    print(f"[{args.run_name}] arch={args.arch} width={args.width} params={count_params(model.net):,}")

    ckpt = ModelCheckpoint(dirpath=args.out, filename=f"{args.run_name}-best",
                           monitor="val_joint", mode="max", save_top_k=1)
    logger = CSVLogger(save_dir=args.out, name=args.run_name)   # -> <out>/<run_name>/version_x/metrics.csv
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto",
                         callbacks=[ckpt], logger=logger, log_every_n_steps=20)
    trainer.fit(model, tr, va)
    print(f"[{args.run_name}] best: {ckpt.best_model_path}  val_joint={ckpt.best_model_score}")
    print(f"[{args.run_name}] curves: {logger.log_dir}/metrics.csv")


if __name__ == "__main__":
    main()
