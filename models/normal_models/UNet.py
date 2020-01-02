# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

# for the comparison, normal FCN

import torch
import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.nn.functional as F
from models.normal_models.layers import _FCNHead
from collections import OrderedDict

class Unet_resnet18(nn.Module):
    def __init__(self, nb_classes, pretrained=False):
        super(Unet_resnet18, self).__init__()

        self.nb_classes = nb_classes
        #backbone = resnet.resnet18(pretrained=pretrained)

        # encoder block1, inc 3, outc 32, downsample outc 64
        self.layer1 = nn.Sequential(OrderedDict({
                'conv1': nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                'bn1': nn.BatchNorm2d(32),
                'relu': nn.ReLU(),
                'conv2': nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                'bn2': nn.BatchNorm2d(32),
            }))
        self.downsample1 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(32, 64, 1, 2, bias=False),
            'bn': nn.BatchNorm2d(64),
        }))

        # encoder block2, inc 64, outc 64, downsample outc 128
        self.layer2 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(64),
        }))
        self.downsample2 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(64, 128, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(128),
        }))

        # encoder block3, inc 128, outc 128, downsample outc 256
        self.layer3 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(128),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(128),
        }))
        self.downsample3 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(128, 256, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(256),
        }))

        # encoder block4, inc 256, outc 256, downsample outc 256
        self.layer4 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(256),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(256),
        }))
        self.downsample4 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(256, 256, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(256),
        }))
        # bilinear upsampled

        # decoder block 1, [256, 256] > 128
        self.decoder_layer1 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(512, 128, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(128),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(128),
        }))
        # bilinear upsampled

        # decoder block 2  [128, 128] > 64
        self.decoder_layer2 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(256, 64, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(64),
        }))
        # bilinear upsampled

        # decoder block 3  [64, 64]   > 32
        self.decoder_layer3 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(128, 32, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(32),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(32),
        }))
        # bilinear upsampled
        '''
        # decoder block 4  [32, 32]   > nb_classes
        self.decoder_layer4 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(32),
            'relu': nn.ReLU(),
            'conv2': nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(32),
        }))
        '''

        self.classifier = _FCNHead(64, nb_classes)

        self._weight_init()

    def forward(self, x):

        x1 = self.layer1(x) # downsample 2, out 32
        x1_downsample = self.downsample1(x1) # downsample 4, out 64

        x2 = self.layer2(x1_downsample) # downsample 4, out 64
        x2_downsample = self.downsample2(x2) # downsample 8, out 128

        x3 = self.layer3(x2_downsample) # downsample 8, out 128
        x3_downsample = self.downsample3(x3) # downsample 16, out 256

        x4 = self.layer4(x3_downsample) # downsample 16, out 256
        x4_downsample = self.downsample4(x4) # downsample 32, out 256

        x4_downsample = F.interpolate(x4_downsample, scale_factor=2, mode='bilinear', align_corners=True) # downsample 16, out 256
        x4 = torch.cat([x4, x4_downsample], dim=1) # downsample 16, out 512
        x4 = self.decoder_layer1(x4) # downsample 16, out 128
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True) # downsample 8, out 128

        x3 = torch.cat([x4, x3], dim=1) # downsample 8, out 256
        x3 = self.decoder_layer2(x3) # downsample 8, out 64
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True) # downsample 4, out 64

        x2 = torch.cat([x3, x2], dim=1) #downsample 4, outc 128
        x2 = self.decoder_layer3(x2) # downsample 4, outc 32
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1) # downsample 4 outc 64
        x1 = self.classifier(x1) # downsample 4 outc 6

        #x1 = self.decoder_layer4(x1)
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=True)

        #out = self.classifier(x1)
        return x1

    def _weight_init(self):

        for module in self.modules():
            module._weight_init()