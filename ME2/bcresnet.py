"""BC-ResNet -- Broadcasted Residual Learning for keyword spotting.

Kim, Chang, Kim, Ha, "Broadcasted Residual Learning for Efficient Keyword
Spotting", Interspeech 2021. Best accuracy-per-parameter tiny KWS baseline.

Each BC-ResBlock mixes a 2-D frequency-depthwise path with a 1-D temporal path
obtained by averaging over frequency (the "broadcast"), so most compute is 1-D.
Width scales with `tau`. Input/output match model.py's interface exactly:
    (B, 1, N_MELS, N_FRAMES) -> (B, n_classes)
"""
import torch
import torch.nn as nn

from config import N_CLASSES


class SubSpectralNorm(nn.Module):
    """Split the frequency axis into `sub_bands` groups and BN each separately.

    Falls back to plain BatchNorm when the current frequency size is not
    divisible by sub_bands (keeps deep stages from crashing).
    """

    def __init__(self, channels, sub_bands=5, eps=1e-5):
        super().__init__()
        self.S = sub_bands
        self.bn_grouped = nn.BatchNorm2d(channels * sub_bands, eps=eps)
        self.bn_plain = nn.BatchNorm2d(channels, eps=eps)

    def forward(self, x):                       # x: (N, C, F, T)
        N, C, F, T = x.shape
        if self.S > 1 and F % self.S == 0:
            x = x.view(N, C * self.S, F // self.S, T)
            x = self.bn_grouped(x)
            return x.view(N, C, F, T)
        return self.bn_plain(x)


class BCResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, freq_stride=1, sub_bands=5, temp_dilation=1):
        super().__init__()
        self.identity = (in_ch == out_ch) and (freq_stride == 1)

        # 1x1 transition when channels change.
        self.expand = None
        if in_ch != out_ch:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        # f2: frequency-depthwise (3x1) + Sub-Spectral Norm.
        self.freq_dw = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1),
                                 stride=(freq_stride, 1), padding=(1, 0),
                                 groups=out_ch, bias=False)
        self.ssn = SubSpectralNorm(out_ch, sub_bands)

        # f1: temporal-depthwise (1x3, dilated) + BN + swish + 1x1 + dropout.
        self.temp_dw = nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3),
                                 padding=(0, temp_dilation), dilation=(1, temp_dilation),
                                 groups=out_ch, bias=False)
        self.temp_bn = nn.BatchNorm2d(out_ch)
        self.swish = nn.SiLU(inplace=True)
        self.temp_pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.dropout = nn.Dropout2d(0.1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.expand is not None:
            x = self.expand(x)
        identity = x

        aux2d = self.ssn(self.freq_dw(x))       # (N, C, F', T)

        out1d = aux2d.mean(dim=2, keepdim=True)  # broadcast: average over frequency
        out1d = self.temp_dw(out1d)
        out1d = self.swish(self.temp_bn(out1d))
        out1d = self.dropout(self.temp_pw(out1d))

        out = aux2d + out1d                      # (N,C,F',T) + (N,C,1,T)
        if self.identity:
            out = out + identity
        return self.act(out)


class BCResNet(nn.Module):
    # (out_channels_base, n_blocks, freq_stride, temporal_dilation) per stage.
    STAGES = [(16, 2, 1, 1), (24, 2, 2, 2), (32, 4, 2, 4), (48, 4, 1, 8)]

    def __init__(self, n_classes=N_CLASSES, tau=1.0, sub_bands=5):
        super().__init__()
        w = lambda c: max(1, int(round(c * tau)))
        c0 = w(self.STAGES[0][0])

        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
        )

        blocks, cin = [], c0
        for base_c, n_blk, fstride, dil in self.STAGES:
            cout = w(base_c)
            for b in range(n_blk):
                blocks.append(BCResBlock(cin, cout,
                                         freq_stride=(fstride if b == 0 else 1),
                                         sub_bands=sub_bands, temp_dilation=dil))
                cin = cout
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Conv2d(cin, n_classes, 1)
        self.embedding_dim = cin                 # for SLU / shared-encoder use

    def features(self, x):                       # (N, 1, F, T) -> (N, C)
        x = self.blocks(self.stem(x))
        return x.mean(dim=(2, 3))                # collapse frequency + time

    def forward(self, x):                        # (N, 1, F, T)
        x = self.blocks(self.stem(x))
        x = x.mean(dim=2, keepdim=True)          # collapse frequency -> (N, C, 1, T)
        x = self.head(x)                         # (N, n_classes, 1, T)
        return x.mean(dim=(2, 3))                # collapse time -> (N, n_classes)


if __name__ == "__main__":                       # quick self-test
    from model import count_params
    net = BCResNet(tau=1.0)
    y = net(torch.randn(2, 1, 40, 201))
    print("out", tuple(y.shape), "| params", f"{count_params(net):,}")
