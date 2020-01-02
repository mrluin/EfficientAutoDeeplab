# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import torch
import torch.nn as nn
import torch.nn.init as init

from collections import OrderedDict

class ConvBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(ConvBnReLu, self).__init__()

        self.cbr = nn.Sequential(OrderedDict({
                    'conv': nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                    'bn': nn.BatchNorm2d(out_channels),
                    'relu': nn.ReLU(),
                }))
        self._weight_init()
    def forward(self, x):
        return self.cbr(x)

    def _weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=False):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.cbr = ConvBnReLu(in_channels, inter_channels)
        self.conv1x1 = nn.Conv2d(inter_channels, out_channels, 1, 1, 0, bias=False)

        self._weight_init()
    def forward(self, x):
        x = self.cbr(x)
        x = self.conv1x1(x)
        return x

    def _weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

