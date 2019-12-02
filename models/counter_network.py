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


''' the network for testing the capacity of the searched network '''
class CounterMBConvNet(MyNetwork):
    def __init__(self, nb_classes):
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

        '''
        # 32, 64, 128, 256 for scale 0, 1, 2, 3
        MBConv_layer_scale0 = build_candidate_ops(['3x3_MBConv3'], in_channels=32, out_channels=32, stride=1, ops_order='act_weight_bn')
        MBconv_layer_scale1 = build_candidate_ops(['3x3_MBConv3'], in_channels=64, out_channels=64, stride=1, ops_order='act_weight_bn')
        MBconv_layer_scale2 = build_candidate_ops(['3x3_MBConv3'], in_channels=128, out_channels=128, stride=1, ops_order='act_weight_bn')
        MBconv_layer_scale3 = build_candidate_ops(['3x3_MBConv3'], in_channels=256, out_channels=256, stride=1, ops_order='act_weight_bn')

        MBconv_identity_scale0 = Identity(32, 32)
        MBconv_identity_scale1 = Identity(64, 64)
        MBconv_identity_scale2 = Identity(128, 128)
        MBconv_identity_scale3 = Identity(256, 256)
        
        # layer 0
        self.layer0_prev = ConvLayer(64, 32, 1, 1, False)
        self.layer0_MBlock = MobileInvertedResidualBlock(MBConv_layer_scale0, MBconv_identity_scale0)
        # layer 1
        self.layer1_prev_prev = ConvLayer(64, 32, 1, 1, False)
        self.layer1_prev = ConvLayer(64, 32, 1, 1, False)
        '''
        # scale 0
        self.layer0_cell = GumbelCell(0, 32, 1, 1, None, 64, 0, ['3x3_MBConv3'])
        self.layer1_cell = GumbelCell(0, 32, 1, 1, 64, 0, 0, ['3x3_MBConv3'])
        # scale 1
        self.layer2_cell = GumbelCell(0, 32, 1, 1, None, 0, 1, ['3x3_MBConv3'])
        self.layer3_cell = GumbelCell(0, 32, 1, 1, None, 1, 1, ['3x3_MBConv3'])
        self.layer4_cell = GumbelCell(0, 32, 1, 1, 1, 1, 1, ['3x3_MBConv3'])
        # scale 2
        self.layer5_cell = GumbelCell(0, 32, 1, 1, None, 1, 2, ['3x3_MBConv3'])
        self.layer6_cell = GumbelCell(0, 32, 1, 1, None, 2, 2, ['3x3_MBConv3'])
        self.layer7_cell = GumbelCell(0, 32, 1, 1, 2, 2, 2, ['3x3_MBConv3'])
        # scale 3
        self.layer8_cell = GumbelCell(0, 32, 1, 1, None, 2, 3, ['3x3_MBConv3'])
        self.layer9_cell = GumbelCell(0, 32, 1, 1, None, 3, 3, ['3x3_MBConv3'])
        # scale 2
        self.layer10_cell = GumbelCell(0, 32, 1, 1, None, 3, 2, ['3x3_MBConv3'])
        # scale 1
        self.layer11_cell = GumbelCell(0, 32, 1, 1, None, 2, 1, ['3x3_MBConv3'])


        self.aspp = ASPP(64, nb_classes, dilation=12)

    def forward(self, x):

        size = x.size()[-2:]
        x = self.stem0(x)
        x = self.stem1(x)
        out_stem2 = self.stem2(x)

        layer0_cell_out = self.layer0_cell(None, out_stem2)
        layer1_cell_out = self.layer1_cell(out_stem2, layer0_cell_out)
        layer2_cell_out = self.layer2_cell(None, layer1_cell_out)
        layer3_cell_out = self.layer3_cell(None, layer2_cell_out)
        layer4_cell_out = self.layer4_cell(layer2_cell_out, layer3_cell_out)
        layer5_cell_out = self.layer5_cell(None, layer4_cell_out)
        layer6_cell_out = self.layer6_cell(None, layer5_cell_out)
        layer7_cell_out = self.layer7_cell(layer5_cell_out, layer6_cell_out)
        layer8_cell_out = self.layer8_cell(None, layer7_cell_out)
        layer9_cell_out = self.layer9_cell(None, layer8_cell_out)
        layer10_cell_out = self.layer10_cell(None, layer9_cell_out)
        layer11_cell_out = self.layer11_cell(None, layer10_cell_out)

        return F.interpolate(self.aspp(layer11_cell_out), size, mode='bilinear', align_corners=True)




