# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F

from models.normal_models.layers import _FCNHead
from collections import OrderedDict


class FCN32s(nn.Module):
    def __init__(self, nb_classes, pretrained=False):
        super(FCN32s, self).__init__()

        self.nb_classes = nb_classes
        backbone = resnet.resnet18(pretrained=pretrained)

        # two-layer stem, output channels with 32 and 64
        self.stem0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True))
        ])) # down-sampling 2
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True)),
        ])) # down-sampling 4

        self.layer1 = backbone.layer1 # 64 64   down-sampling 8
        self.layer2 = backbone.layer2 # 64 128  down-sampling 16
        self.layer3 = backbone.layer3 # 128 256 down-sampling 32

        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False) # out down-sampling 16
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False)  # out down-sampling 8
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1, bias=False)   # out down-sampling 4
        #

        self.score_layer = _FCNHead(64, nb_classes)

        self._weight_init()

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x) #

        x1 = self.layer1(x) #  64
        x2 = self.layer2(x1) # 128
        x3 = self.layer3(x2) # 256
        #x4 = self.layer4(x3) #

        out = self.deconv3(x3)       # 128
        out = self.deconv2(out + x2) # 128 + 128 out 64
        out = self.deconv1(out + x1) # 64 + 64

        out = self.score_layer(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return out

    def _weight_init(self):

        for module in self.modules():
            module._weight_init()

