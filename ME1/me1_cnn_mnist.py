"""
AI 231 (MLOps) - Machine Exercise 1
====================================
A 3-layer CNN for MNIST classification with every layer/operation implemented
from scratch using torch tensors + einops/einsum. No CNN libraries
(no torch.nn.Conv2d, no F.conv2d, no nn.MaxPool2d, no nn.Linear).

- Convolution        -> torch.Tensor.unfold (sliding-window view) + einsum
- Max pooling        -> einops.reduce (max)
- Flatten            -> einops.rearrange
- Linear / FC        -> einsum
- ReLU, log-softmax, cross-entropy -> plain tensor ops

torch's autograd is used only to backprop through the einsum/tensor graph, and
torch.optim.Adam is used as the optimizer -- neither is a "CNN library"; the
network itself is built entirely from the primitives above.

Author: Jomar Christian Camposano
"""

import torch
import torch.optim as optim
from torch import einsum
from einops import rearrange, reduce
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # safe for headless HPC nodes; notebook overrides this
import matplotlib.pyplot as plt

SEED = 0
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ----------------------------------------------------------------------------
# Data (torchvision is used ONLY to fetch/normalize MNIST, not for modeling)
# ----------------------------------------------------------------------------
def get_loaders(batch_size=BATCH_SIZE):
    tfm = transforms.Compose([
        transforms.ToTensor(),                 # -> float tensor (1, 28, 28) in [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_ds, test_ds, train_loader, test_loader


# ----------------------------------------------------------------------------
# Layers / operations -- all from scratch with einops / einsum
# ----------------------------------------------------------------------------
def conv2d(x, weight, bias, stride=1):
    """2D valid convolution.
    x:      (B, Cin, H, W)
    weight: (Cout, Cin, K, K)
    bias:   (Cout,)
    returns (B, Cout, OH, OW)
    Patches are extracted with a strided sliding-window view (Tensor.unfold),
    then the multiply-accumulate is a single einsum contraction.
    """
    K = weight.shape[-1]
    # (B, Cin, OH, OW, K, K)
    patches = x.unfold(2, K, stride).unfold(3, K, stride)
    out = einsum("b c p q i j, o c i j -> b o p q", patches, weight)
    out = out + rearrange(bias, "o -> 1 o 1 1")
    return out


def relu(x):
    return torch.clamp(x, min=0.0)


def maxpool2d(x, k=2):
    """Non-overlapping max pool via an einops max-reduction.
    einops.reduce needs the spatial axes to divide evenly, so we drop the
    remainder first -- this matches standard max pooling with ceil_mode=False.
    """
    H, W = x.shape[-2], x.shape[-1]
    x = x[..., : H - H % k, : W - W % k]
    return reduce(x, "b c (h ph) (w pw) -> b c h w", "max", ph=k, pw=k)


def linear(x, weight, bias):
    """Fully connected layer. x:(B,I) weight:(O,I) bias:(O,)"""
    return einsum("b i, o i -> b o", x, weight) + bias


def log_softmax(logits):
    z = logits - logits.max(dim=1, keepdim=True).values      # numerical stability
    return z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))


def cross_entropy(logits, targets):
    logp = log_softmax(logits)
    nll = -logp[torch.arange(logits.shape[0], device=logits.device), targets]
    return nll.mean()


# ----------------------------------------------------------------------------
# Model: 3 convolutional layers + 1 linear classifier head
#   28x28  -conv1(1->8) ->26  -pool-> 13
#          -conv2(8->16)->11  -pool-> 5
#          -conv3(16->32)->3   (flatten 32*3*3=288)  -linear-> 10
# ----------------------------------------------------------------------------
def he_init(shape, fan_in):
    return torch.randn(shape, device=device) * (2.0 / fan_in) ** 0.5


class ScratchCNN:
    def __init__(self):
        self.w1 = he_init((8, 1, 3, 3), 1 * 3 * 3).requires_grad_()
        self.b1 = torch.zeros(8, device=device, requires_grad=True)
        self.w2 = he_init((16, 8, 3, 3), 8 * 3 * 3).requires_grad_()
        self.b2 = torch.zeros(16, device=device, requires_grad=True)
        self.w3 = he_init((32, 16, 3, 3), 16 * 3 * 3).requires_grad_()
        self.b3 = torch.zeros(32, device=device, requires_grad=True)
        self.wfc = he_init((10, 32 * 3 * 3), 32 * 3 * 3).requires_grad_()
        self.bfc = torch.zeros(10, device=device, requires_grad=True)

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2,
                self.w3, self.b3, self.wfc, self.bfc]

    def forward(self, x):
        x = maxpool2d(relu(conv2d(x, self.w1, self.b1)))   # (B, 8, 13, 13)
        x = maxpool2d(relu(conv2d(x, self.w2, self.b2)))   # (B, 16, 5, 5)
        x = relu(conv2d(x, self.w3, self.b3))              # (B, 32, 3, 3)
        x = rearrange(x, "b c h w -> b (c h w)")           # (B, 288)
        return linear(x, self.wfc, self.bfc)               # (B, 10)

    __call__ = forward


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def train(model, train_loader, test_loader, epochs=EPOCHS):
    opt = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}/{epochs}  train_loss={running/seen:.4f}  test_acc={acc*100:.2f}%")
    return model


@torch.no_grad()
def evaluate(model, loader):
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


# ----------------------------------------------------------------------------
# Visualization: 16 random test images in a 4x4 grid with GT + prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def plot_samples(model, test_ds, out_path="sample_predictions.png"):
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=True)
    imgs, labels = next(iter(loader))
    preds = model(imgs.to(device)).argmax(dim=1).cpu()

    # undo Normalize purely for display
    disp = imgs * 0.3081 + 0.1307

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(disp[i, 0], cmap="gray")
        ok = labels[i].item() == preds[i].item()
        ax.set_title(f"GT: {labels[i].item()}  Pred: {preds[i].item()}",
                     color="green" if ok else "red", fontsize=10)
        ax.axis("off")
    fig.suptitle("MNIST - 3-layer scratch CNN (einops/einsum)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    return fig


def main():
    train_ds, test_ds, train_loader, test_loader = get_loaders()
    model = ScratchCNN()
    train(model, train_loader, test_loader, EPOCHS)
    final_acc = evaluate(model, test_loader)
    print(f"\nFINAL TEST ACCURACY: {final_acc*100:.2f}%")
    plot_samples(model, test_ds)


if __name__ == "__main__":
    main()
