import torch
from torch import nn
from nets.swin import SwinTransformer
import torch.nn.functional as F

class Encode(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encode, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class Decode(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decode, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=2, dilation=2)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=2, dilation=2)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encode1 = Encode(3, 64)  # 128 128 64
        self.encode2 = Encode(64, 128)  # 64 64 128
        self.backbone = SwinTransformer(img_size=64, patch_size=4, in_chans=128,
                                           embed_dim=96, depths=[6, 4], num_heads=[3, 6])  # 8 8 192
        self.decode1 = Decode(192, 512)  # 16 16 512
        self.decode2 = Decode(512, 256)  # 32 32 256
        self.decode3 = Decode(256, 128)  # 64 64 128
        self.decode4 = Decode(128, 64)  # 128 128 64
        self.decode5 = Decode(64, 64)  # 128 128 64
        self.decode6 = Decode(3, 64)  # 128 128 64

    def forward(self, x):
        x1 = self.encode1(x)  # 128 128 64
        x2 = self.encode2(x1)  # 64 64 128
        x3 = self.backbone(x2)  # 8 8 192
        x4 = F.interpolate(x3, size=x3.shape[-1] * 2, mode='bilinear', align_corners=True)
        x4 = self.decode1(x4)  # 16 16 512
        x5 = F.interpolate(x4, size=x4.shape[-1] * 2, mode='bilinear', align_corners=True)
        x5 = self.decode2(x5)  # 32 32 256
        x6 = F.interpolate(x5, size=x5.shape[-1] * 2, mode='bilinear', align_corners=True)
        x6 = self.decode3(x6) + x2  # 64 64 128
        x7 = F.interpolate(x6, size=x6.shape[-1] * 2, mode='bilinear', align_corners=True)
        x7 = self.decode4(x7) + x1  # 128 128 64
        x8 = F.interpolate(x7, size=x7.shape[-1] * 2, mode='bilinear', align_corners=True)
        x8 = self.decode5(x8) + self.decode6(x)  # 128 128 64
        return x8