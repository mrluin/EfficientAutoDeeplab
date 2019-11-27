'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from models.gumbel_super_network import GumbelAutoDeepLab
from modules.my_modules import MyNetwork, MyModule
from modules.operations import ASPP, FactorizedIncrease, ConvLayer, FactorizedReduce, MBInvertedConvLayer, Identity, \
    Zero, SepConv, DilConv
from run_manager import RunConfig
from utils.common import get_prev_c


__all__ = ['NewGumbelCell', 'NewGumbelAutoDeeplab']

def build_candidate_ops(candiate_ops, in_channels, out_channels, stride, ops_order):

    if candiate_ops is None:
        raise ValueError('Please specify a candidate set')

    # None zero layer
    name2ops = {
        'Identity': lambda inc, outc, s: Identity(inc, outc, ops_order=ops_order),
        'Zero': lambda inc, outc, s: Zero(s),
    }
    # add MBConv Layers
    name2ops.update({
        '3x3_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 1),
        '3x3_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 2),
        '3x3_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 3),
        '3x3_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 4),
        '3x3_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 5),
        '3x3_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 6),
        '5x5_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 1),
        '5x5_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 2),
        '5x5_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 3),
        '5x5_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 4),
        '5x5_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 5),
        '5x5_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 6),
        '7x7_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 1),
        '7x7_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 2),
        '7x7_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 3),
        '7x7_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 4),
        '7x7_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 5),
        '7x7_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 6),
        #===========================================================================
        '3x3_DWConv': lambda inc, outc, s: SepConv(inc, outc, 3, s),
        '5x5_DWConv': lambda inc, outc, s: SepConv(inc, outc, 5, s),
        '3x3_DilConv': lambda inc, outc, s: DilConv(inc, outc, 3, s, 2),
        '5x5_DilConv': lambda inc, outc, s: DilConv(inc, outc, 5, s, 2),
        '3x3_AvgPooling': lambda inc, outc, s: nn.AvgPool2d(3, stride=s, padding=1, count_include_pad=False),
        '3x3_MaxPooling': lambda inc, outc, s: nn.MaxPool2d(3, stride=s, padding=1),
    })
    return [
        name2ops[name](in_channels, out_channels, stride) for name in candiate_ops
    ]


class NewGumbelCell(MyModule):
    def __init__(self, layer, filter_multiplier, block_multiplier, steps,
                 prev_prev_scale, prev_scale, scale, genotype, conv_candidates):
        super(NewGumbelCell, self).__init__()
        index2scale = {
            -2: 1,
            -1: 2,
            0: 4,
            1: 8,
            2: 16,
            3: 32,
        }
        self.total_nodes = 2 + steps
        self.layer = layer
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.prev_prev_scale = prev_prev_scale
        self.prev_scale = prev_scale
        self.scale = scale
        self.genotype = genotype
        self.conv_candidates = conv_candidates

        self.outc = int(self.filter_multiplier * self.block_multiplier * self.scale / 4)

        # preprocess0 and preprocess1
        if index2scale.get(self.prev_scale) is not None:
            self.prev_c = int(self.filter_multiplier * self.block_multiplier * index2scale[self.prev_scale] / 4)
            if self.prev_scale == self.scale + 1:  # up
                self.preprocess1 = FactorizedIncrease(self.prev_c, self.outc)
            elif self.prev_scale == self.scale:  # same
                self.preprocess1 = ConvLayer(self.prev_c, self.outc, 1, 1, False)
            elif self.prev_scale == self.scale - 1:  # down
                self.preprocess1 = FactorizedReduce(self.prev_c, self.outc)
            else:
                raise ValueError('invalid relation between prev_scale and current scale')
        else:
            self.prev_c = self.prev_scale
            if scale == 0:
                self.preprocess1 = ConvLayer(self.prev_c, self.outc, 1, 1, False)
            elif self.scale == 1:
                self.preprocess1 = FactorizedReduce(self.prev_c, self.outc)
            else:
                raise ValueError('invalid scale value')
        if self.prev_prev_scale is None:
            self.prev_prev_c = None
            self.preprocess0 = None
        else:
            if index2scale.get(self.prev_prev_scale) is None:  # fixed
                self.prev_prev_c = self.prev_prev_scale
            else:
                self.prev_prev_c = int(
                    self.filter_multiplier * self.block_multiplier * index2scale[self.prev_prev_scale] / 4)
            # TODO: issue in scale of prev_prev_c, it is considered as next_scale by default
            self.preprocess0 = ConvLayer(self.prev_prev_c, self.outc, 1, 1, False)

        self.ops = nn.ModuleDict()

        for node_str , select_op_index in self.genotype:
            # operation name -> conv_candidates
            operation = build_candidate_ops([self.conv_candidates[select_op_index]], self.outc, self.outc,
                                            stride=1, ops_order='act_weight_bn')
            self.ops[node_str] = operation

        self.final_conv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)

    def forward(self, s0, s1):

        # node_str and operation_name in self.genotype
        # match node_str, perform related operation
        if s0 is not None:
            s0 = self.preprocess0(s0)
        else: assert self.prerpocess0 is None, 'inconsistency in s0 and preprocess0'
        s1 = self.preprocess1(s1)
        states = [s0, s1] # including None state

        for i in range(2, self.total_nodes):
            new_states = []
            for j in range(i): # all previous node for each node i
                node_str = '{:}<-{:}'.format(i,j)
                if node_str in self.genotype[:, 0]:
                    # the current edge is selected
                    related_hidden = states[j]
                    if related_hidden is None:
                        assert j == 0, 'inconsistency in None hidden and node index'
                        continue
                    new_state = self.ops[node_str](related_hidden)
                    new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.final_conv1x1(concat_feature)
        return concat_feature

class NewGumbelAutoDeeplab(MyNetwork):
    def __init__(self, nb_layers, filter_multiplier, block_multiplier, steps, nb_classes,
                 actual_path, cell_genotypes, conv_candidates):
        super(NewGumbelAutoDeeplab, self).__init__()

        self.nb_layers = nb_layers
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.nb_classes = nb_classes
        self.steps = steps
        self.actual_path = actual_path
        self.cell_genotypes = cell_genotypes

        # three init stems
        self.stem0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.stem2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.cells = nn.ModuleList()
        prev_prev_c = 32
        prev_c = 64
        inter_scale = [-1, 0]
        for layer in range(self.nb_layers):
            next_scale = self.actual_path[layer]
            cell_genotype = self.cell_genotypes[layer] # next_scale cell genotype
            if layer == 0:
                if next_scale == 0 or next_scale == 1:
                    prev_prev_scale = None
                    prev_scale = prev_c
                else: raise ValueError('invalid scale value in layer 0')
            elif layer == 1:
                if next_scale == 0:
                    prev_prev_scale = prev_c
                    prev_scale = inter_scale[-1]
                else:
                    if next_scale == inter_scale[-2]: prev_prev_scale = inter_scale[-2]
                    else: prev_prev_scale = None
                    prev_scale = inter_scale[-1]
            else:
                if next_scale == inter_scale[-2]: prev_prev_scale = inter_scale[-2]
                else: prev_prev_scale = None
                prev_scale = inter_scale[-1]
            self.cells.append(NewGumbelCell(layer, self.filter_multiplier, self.block_multiplier,
                                            self.steps, prev_prev_scale, prev_scale, next_scale, cell_genotype, conv_candidates))
            inter_scale.pop(0)
            inter_scale.append(next_scale)

        self.out0 = int(self.filter_multiplier * self.block_multiplier * 4 / 4)
        self.out1 = int(self.filter_multiplier * self.block_multiplier * 8 / 4)
        self.out2 = int(self.filter_multiplier * self.block_multiplier * 16 / 4)
        self.out3 = int(self.filter_multiplier * self.block_multiplier * 32 / 4)
        last_scale = self.actual_path[-1]
        if last_scale == 0:
            self.aspp = ASPP(self.out0 ,self.nb_classes, 24, self.nb_classes)
        elif last_scale == 1:
            self.aspp = ASPP(self.out1, self.nb_classes, 12, self.nb_classes)
        elif last_scale == 2:
            self.aspp = ASPP(self.out2, self.nb_classes, 6, self.nb_classes)
        elif last_scale == 3:
            self.aspp = ASPP(self.out3, self.nb_classes, 3, self.nb_classes)
        else:
            raise ValueError('invalid last_scale value {}'.format(last_scale))
    def forward(self, x):

        size = x.size()[2:]
        # chain-like structure, normal forward cells and last aspp
        inter_features = []
        x = self.stem0(x)
        x = self.stem1(x)
        inter_features.append((-1, None))
        x = self.stem2(x)
        inter_features.append((0, x))

        for layer in range(self.nb_layers):
            next_scale = self.actual_path[layer]
            prev_prev_feature, prev_feature = get_prev_c(inter_features, next_scale)
            _result = self.cells[layer](prev_prev_feature, prev_feature)
            inter_features.pop(0)
            inter_features.append((next_scale, _result))

        _result = self.aspp(inter_features[-1])
        return F.interpolate(_result, size=size, mode='bilinear', align_corners=True)

def get_new_model(super_network, run_config: RunConfig):
    nb_layers = run_config.nb_layers
    filter_multiplier = run_config.filter_multiplier
    block_multiplier = run_config.block_multiplier
    steps = run_config.steps
    nb_classes = run_config.nb_classes
    actual_path, cell_genotypes = super_network.network_cell_arch_decode()
    print('\t=> Construct New Model ... ...')
    normal_model = NewGumbelAutoDeeplab(nb_layers, filter_multiplier, block_multiplier, steps, nb_classes, actual_path, cell_genotypes)
    print('\t=> New Model Constructed Done ... ... Begin Testing')

    data = torch.zeros(1, 3, 512, 512).to('cuda:{}'.format(run_config.gpu_ids))
    normal_model(data)
    print('\t=> Testing Done.')
    return normal_model
