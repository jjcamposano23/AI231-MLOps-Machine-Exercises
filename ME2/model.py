"""Model factory for the VCM.

Backbones expose a shared embedding via `.features(x)` and `.embedding_dim`, so
the same encoder drives either a single-head classifier (KWS) or the two-head
SLU model (intent + slot). Both train FROM SCRATCH (random init, no pretrained
weights) -- required for this exercise.

  * "dscnn"    -- tiny depthwise-separable CNN (~21K params at c=64).
  * "bcresnet" -- broadcasted-residual net (bcresnet.py); higher acc/param.
  * SLUNet     -- backbone + intent head + slot head (the Option B model).
"""
import torch.nn as nn
import torch.nn.functional as F

from config import N_CLASSES, N_INTENTS, N_SLOTS


# --------------------------------------------------------------------- DS-CNN --
class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 conv, each BN + ReLU. Separate ReLU modules
    (act1/act2) so torch.ao.quantization can fuse conv+bn+relu for QAT."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class DSCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES, c=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DSConv(c, c), DSConv(c, c), DSConv(c, c, stride=2), DSConv(c, c),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(c, n_classes)
        self.embedding_dim = c

    def features(self, x):              # (B,1,F,T) -> (B, c)
        return self.pool(self.blocks(self.stem(x))).flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.features(x)))


def fuse_dscnn(model: "DSCNN"):
    """In-place conv+bn(+relu) fusion for eager-mode QAT. Call in eval() first."""
    import torch.ao.quantization as tq
    tq.fuse_modules(model.stem, ["0", "1", "2"], inplace=True)
    for blk in model.blocks:
        tq.fuse_modules(blk, [["dw", "bn1", "act1"], ["pw", "bn2", "act2"]], inplace=True)
    return model


# ------------------------------------------------------------------- SLU head --
class SLUNet(nn.Module):
    """Shared backbone -> (intent logits, slot logits). The Option B VCM."""

    def __init__(self, backbone, n_intents=N_INTENTS, n_slots=N_SLOTS):
        super().__init__()
        self.backbone = backbone
        d = backbone.embedding_dim
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(d, n_intents)
        self.slot_head = nn.Linear(d, n_slots)

    def forward(self, x):
        z = self.dropout(self.backbone.features(x))
        return self.intent_head(z), self.slot_head(z)


# ---------------------------------------------------------------- model factory --
def build_model(arch="dscnn", n_classes=N_CLASSES, width=64):
    """Single-head classifier (KWS). arch='dscnn'->width=c; 'bcresnet'->width=tau."""
    if arch == "dscnn":
        return DSCNN(n_classes=n_classes, c=int(width))
    if arch == "bcresnet":
        from bcresnet import BCResNet
        return BCResNet(n_classes=n_classes, tau=float(width))
    raise ValueError(f"unknown arch: {arch!r}")


def build_slu_model(arch="dscnn", width=64, n_intents=N_INTENTS, n_slots=N_SLOTS):
    """Two-head SLU model (intent + slot) on a from-scratch backbone."""
    backbone = build_model(arch, n_classes=1, width=width)
    # SLUNet uses backbone.features() only -> drop the backbone's own classifier/head
    # so it has no unused parameters (which otherwise break multi-GPU DDP).
    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    if hasattr(backbone, "head"):
        backbone.head = nn.Identity()
    return SLUNet(backbone, n_intents=n_intents, n_slots=n_slots)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
