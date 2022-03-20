from torch import nn

import sys

sys.path.append('../')

from nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from nets.Tranformer import Transformer


class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes=20, backbone='resnet50', pretrain=False):
        super(CenterNet_Resnet50, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrain=pretrain)
            self.decoder = resnet50_Decoder(2048)
            self.head = resnet50_Head(channel=64, num_classes=num_classes)
        else:
            raise ValueError('backbone must be resnet50')

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


class CenterNet_Swin(nn.Module):
    def __init__(self, num_classes=20):
        super(CenterNet_Swin, self).__init__()
        self.backbone = Transformer()
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    model = CenterNet_Swin(num_classes=1)
    from torchsummary import summary

    summary(model, (3, 256, 256), device='cpu')
