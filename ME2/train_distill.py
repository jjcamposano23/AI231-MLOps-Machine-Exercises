"""Knowledge distillation: train a tiny student to mimic a strong teacher.

Step 1 -- train a teacher (once, on the DGX):
    python train.py --data data --noise data/_silence_ --arch bcresnet --width 6 \
        --epochs 80 --out checkpoints_teacher

Step 2 -- distill into the tiny deployable student:
    python train_distill.py --data data --noise data/_silence_ \
        --teacher checkpoints_teacher/vcm-best.ckpt \
        --arch dscnn --width 64 --epochs 80

Loss = alpha * CE(student, labels)
     + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))

The T^2 factor keeps the soft-label gradients on the same scale as the hard-label
gradients (Hinton et al., 2015). Best student checkpoint is saved for export.py.
"""
import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

from config import N_CLASSES
from data import build_loaders
from model import build_model, count_params
from train import VCM               # to load the teacher checkpoint


class DistillVCM(pl.LightningModule):
    def __init__(self, teacher_ckpt, arch="dscnn", width=64, n_classes=N_CLASSES,
                 temperature=4.0, alpha=0.5, lr=3e-3, max_epochs=80):
        super().__init__()
        # Don't pickle the (large) teacher into student hparams/checkpoints.
        self.save_hyperparameters(ignore=["teacher_ckpt"])

        self.student = build_model(arch, n_classes=n_classes, width=width)

        teacher = VCM.load_from_checkpoint(teacher_ckpt, map_location="cpu").net
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher       # registered -> Lightning moves it to device

        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def training_step(self, batch, _):
        x, y = batch
        with torch.no_grad():
            self.teacher.eval()
            t_logits = self.teacher(x)
        s_logits = self.student(x)

        T, a = self.hparams.temperature, self.hparams.alpha
        hard = self.ce(s_logits, y)
        soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                        F.softmax(t_logits / T, dim=1),
                        reduction="batchmean") * (T * T)
        loss = a * hard + (1 - a) * soft
        self.log_dict({"train_loss": loss, "hard": hard, "soft": soft}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.student(x)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.student.parameters(),
                                lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return [opt], [sch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--noise", default=None)
    ap.add_argument("--teacher", required=True, help="teacher .ckpt from train.py")
    ap.add_argument("--arch", default="dscnn", choices=["dscnn", "bcresnet"])
    ap.add_argument("--width", type=float, default=64)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on hard-label CE; (1-alpha) on soft KL")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    tr, va, _, counts = build_loaders(args.data, args.noise, args.batch, args.workers)
    print(f"train {counts[0]}  val {counts[1]}  test {counts[2]}")

    model = DistillVCM(teacher_ckpt=args.teacher, arch=args.arch, width=args.width,
                       temperature=args.temperature, alpha=args.alpha,
                       lr=args.lr, max_epochs=args.epochs)
    print(f"student arch={args.arch} width={args.width} | "
          f"params: {count_params(model.student):,}")
    print(f"teacher params: {count_params(model.teacher):,}")

    ckpt = ModelCheckpoint(dirpath=args.out, filename="vcm-distilled",
                           monitor="val_acc", mode="max", save_top_k=1)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto",
                         callbacks=[ckpt], log_every_n_steps=10)
    trainer.fit(model, tr, va)
    print(f"best student: {ckpt.best_model_path}  (val_acc={ckpt.best_model_score})")
    print("Next: python export.py --ckpt " + ckpt.best_model_path)


if __name__ == "__main__":
    main()
