import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()
        if not mid_ch:
            mid_ch = out_ch

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(True),
            nn.Conv2d(mid_ch, out_ch, 3, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Downscaling with max pooling then double conv"""

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, f_in=(64, 128, 256, 512, 1024)):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_in = f_in
        self.f_out = list(reversed(f_in))

        self.inc = InConv(in_channels, self.f_in[0])

        downs = list()
        for i in range(len(self.f_in) - 1):
            downs.append(Down(self.f_in[i], self.f_in[i + 1]))
        self.downs = nn.ModuleList(downs)

        ups = list()
        for i in range(len(self.f_out) - 1):
            ups.append(Up(self.f_out[i], self.f_out[i + 1]))
        self.ups = nn.ModuleList(ups)

        self.outc = OutConv(self.f_out[-1], self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)

        downs = [x1]
        for i in range(len(self.f_in) - 1):
            x1 = self.downs[i](x1)
            downs.append(x1)

        ups = list()
        for i in range(len(self.f_out) - 1):
            x1 = self.ups[i](x1, downs[len(downs) - i - 2])
            ups.append(x1)

        logits = self.outc(x1)
        return logits


if __name__ == '__main__':
    net = Unet(3, 3)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 256, 256)
    print(net(x).shape)
