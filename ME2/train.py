"""Train the VCM (student or teacher).

    # student (default tiny DS-CNN):
    python train.py --data data --noise data/_silence_ --epochs 60

    # a stronger teacher for distillation (wider BC-ResNet):
    python train.py --data data --noise data/_silence_ --arch bcresnet --width 6 \
        --out checkpoints_teacher

Produces checkpoints/ with the best (val_acc) checkpoint. A test split is held
out speaker-independently; run benchmark.py on it afterwards.
"""
import argparse

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

from config import N_CLASSES
from data import build_loaders
from model import build_model, count_params


class VCM(pl.LightningModule):
    def __init__(self, arch="dscnn", width=64, n_classes=N_CLASSES,
                 lr=3e-3, max_epochs=60):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_model(arch, n_classes=n_classes, width=width)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, acc, tag):
        x, y = batch
        logits = self.net(x)
        loss = self.loss(logits, y)
        acc(logits, y)
        self.log(f"{tag}_loss", loss, prog_bar=(tag == "train"))
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, self.train_acc, "train")

    def validation_step(self, batch, _):
        return self._step(batch, self.val_acc, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return [opt], [sch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dataset root (folder per class)")
    ap.add_argument("--noise", default=None, help="folder of noise wavs for augmentation")
    ap.add_argument("--arch", default="dscnn", choices=["dscnn", "bcresnet"])
    ap.add_argument("--width", type=float, default=64,
                    help="dscnn: channel count c;  bcresnet: width multiplier tau")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    tr, va, _, counts = build_loaders(args.data, args.noise, args.batch, args.workers)
    print(f"train {counts[0]}  val {counts[1]}  test {counts[2]}")

    model = VCM(arch=args.arch, width=args.width, lr=args.lr, max_epochs=args.epochs)
    print(f"arch={args.arch} width={args.width} | params: {count_params(model.net):,}")

    ckpt = ModelCheckpoint(dirpath=args.out, filename="vcm-best",
                           monitor="val_acc", mode="max", save_top_k=1)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto",
                         callbacks=[ckpt], log_every_n_steps=10)
    trainer.fit(model, tr, va)
    print(f"best checkpoint: {ckpt.best_model_path}  (val_acc={ckpt.best_model_score})")


if __name__ == "__main__":      # required on Windows for num_workers > 0
    main()
