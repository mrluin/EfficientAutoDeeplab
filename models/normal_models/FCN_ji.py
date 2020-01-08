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

class FCN_ji(nn.Module):
    def __init__(self, nb_classes, pretrained=False):
        super(FCN_ji, self).__init__()

        self.nb_classes = nb_classes
        #backbone = resnet.resnet18(pretrained=pretrained)

        # encoder block1, inc 3, outc 32, downsample outc 64
        self.layer1 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(64),
            'relu2': nn.ReLU(),
            }))
        '''
        self.downsample1 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(32, 64, 1, 2, bias=False),
            'bn': nn.BatchNorm2d(64),
        }))
        '''
        self.downsample1 = nn.MaxPool2d(2, 2, 0)

        # encoder block2, inc 64, outc 64, downsample outc 128
        self.layer2 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(128),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(128),
            'relu2': nn.ReLU(),
        }))
        '''
        self.downsample2 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(64, 128, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(128),
        }))
        '''
        self.downsample2 = nn.MaxPool2d(2, 2, 0)

        # encoder block3, inc 128, outc 128, downsample outc 256
        self.layer3 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(256),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(256),
            'relu2': nn.ReLU()
        }))
        '''
        self.downsample3 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(128, 256, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(256),
        }))
        '''
        self.downsample3 = nn.MaxPool2d(2, 2, 0)

        # encoder block4, inc 256, outc 256, downsample outc 256
        self.layer4 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(512),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(512),
            'relu2': nn.ReLU()
        }))
        '''
        self.downsample4 = nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(512, 512, 1, 2, bias=False),
            'bn' : nn.BatchNorm2d(256),
        }))
        '''
        self.downsample4 = nn.MaxPool2d(2, 2, 0)

        self.layer5 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            'bn1': nn.BatchNorm2d(512),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            'bn2': nn.BatchNorm2d(512),
            'relu2': nn.ReLU()
        }))
        # bilinear upsampled

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, 2, 1, 1, bias=False)

        self.up_layer1 = nn.Sequential(OrderedDict({
            'conv1':nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            'bn1': nn.BatchNorm2d(256),
            'relu1': nn.ReLU(),
        }))

        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, bias=False)
        self.up_layer2 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            'bn1': nn.BatchNorm2d(128),
            'relu1': nn.ReLU(),
        }))
        '''
        self.up_layer3 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'relu1': nn.ReLU(),
        }))

        self.up_layer4 = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'relu1': nn.ReLU(),
        }))
        '''

        self.classifier = _FCNHead(128, nb_classes)

        #self._weight_init()

    def forward(self, x):

        x1 = self.layer1(x)                  # downsample 2, out 64
        x1_downsample = self.downsample1(x1) # downsample 4, out 64

        x2 = self.layer2(x1_downsample)      # downsample 4, out 128
        x2_downsample = self.downsample2(x2) # downsample 8, out 128

        x3 = self.layer3(x2_downsample)      # downsample 8, out 256
        x3_downsample = self.downsample3(x3) # downsample 16, out 256

        x4 = self.layer4(x3_downsample)      # downsample 16, out 512
        x4_downsample = self.downsample4(x4) # downsample 32, out 512

        x5 = self.layer5(x4_downsample)      # downsample 32, out 512

        #x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True) # downsample 16, out 512
        x5 = self.deconv1(x5)
        x5 = x4 + x5                                               # downsample 16, out 512
        x4 = self.up_layer1(x5)                                    # downsample 16, out 256

        #x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True) # downsample 8, out 256
        x4 = self.deconv2(x4)
        x4 = x4 + x3                                               # downsample 8, out 256
        x3 = self.up_layer2(x4)                                    # downsample 8, out 128
        '''
        x3 = F.interpolate(x3, scale_factor=2, align_corners=True)
        x3 = x3 + x2
        x2 = self.up_layer3(x3)
        '''
        out = self.classifier(x3)                                    # downsample 8, out 2
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True) #
        return out

    def _weight_init(self):

        for module in self.modules():
            module._weight_init()