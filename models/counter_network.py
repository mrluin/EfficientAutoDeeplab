# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.my_modules import MyNetwork
from collections import OrderedDict
from modules.operations import Identity, Zero, MBInvertedConvLayer, SepConv, DilConv, ASPP
from models.gumbel_cells import GumbelCell, build_candidate_ops
from utils.common import get_prev_c
from models.counter_cells import CounterCell

''' the network for testing the capacity of the searched network '''
class CounterMBConvNet(MyNetwork):
    def __init__(self, nb_classes, search_space):
        super(CounterMBConvNet, self).__init__()
        self.nb_classes = nb_classes

        # three init stems
        self.stem0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        # remove 'relu' for self.stem1 # ('relu', nn.ReLU(inplace=True))
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
        ]))
        # change the order of the stem2
        self.stem2 = nn.Sequential(OrderedDict([
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
        ]))

        self.layer0_cell = CounterCell(0, 32, 1, 1, 64, 0, search_space, affine=True)
        self.layer1_cell = CounterCell(1, 32, 1, 1, 0, 0, search_space, affine=True)
        self.layer2_cell = CounterCell(2, 32, 1, 1, 0, 0, search_space, affine=True)
        self.layer3_cell = CounterCell(3, 32, 1, 1, 0, 0, search_space, affine=True)
        self.layer4_cell = CounterCell(4, 32, 1, 1, 0, 0, search_space, affine=True)
        self.layer5_cell = CounterCell(5, 32, 1, 1, 0, 0, search_space, affine=True)
        self.layer6_cell = CounterCell(6, 32, 1, 1, 0, 1, search_space, affine=True)
        self.layer7_cell = CounterCell(7, 32, 1, 1, 1, 1, search_space, affine=True)
        self.layer8_cell = CounterCell(8, 32, 1, 1, 1, 1, search_space, affine=True)
        self.layer9_cell = CounterCell(9, 32, 1, 1, 1, 1, search_space, affine=True)
        self.layer10_cell = CounterCell(10, 32, 1, 1, 1, 0, search_space, affine=True)
        self.layer11_cell = CounterCell(11, 32, 1, 1, 0, 0, search_space, affine=True)

        #self.aspp = ASPP(64, nb_classes, dilation=12, affine=True)
        self.final = nn.Conv2d(32, 2, 1, 1, 0, bias=False)

    def forward(self, x):

        size = x.size()[-2:]
        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stem2(x)

        x = self.layer0_cell(x)
        x = self.layer1_cell(x)
        x = self.layer3_cell(x)
        x = self.layer4_cell(x)
        x = self.layer5_cell(x)
        x = self.layer6_cell(x)
        x = self.layer7_cell(x)
        x = self.layer8_cell(x)
        x = self.layer9_cell(x)
        x = self.layer10_cell(x)
        x = self.layer11_cell(x)

        x = self.final(x)

        #return F.interpolate(self.aspp(x), size, mode='bilinear', align_corners=True)
        return F.interpolate(x, size, mode='bilinear', align_corners=True)



