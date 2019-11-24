'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from modules.operations import *
from modules.my_modules import MyModule


__all__ = ['build_candidate_ops', 'GumbelCell', 'MixedOp']


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


class MixedOp(MyModule):
    def __init__(self, candidates_ops):
        super(MixedOp, self).__init__()
        self.candidate_ops = nn.ModuleList(candidates_ops)

    def forward(self, x, weight):
        return sum(w * op(x) for w, op in zip(weight, self.candidate_ops))

    def forward_gdas(self, x, weight, argmax):
        return sum(weight[_ie] * op(x) if _ie == argmax else weight[_ie] for _ie, op in enumerate(self.candidate_ops))

    # TODO: whether exists zero layer case
    def is_zero_layer(self):
        return False

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def get_flops(self):
        raise NotImplementedError


class GumbelCell(MyModule):
    def __init__(self, layer,
                 filter_multiplier, block_multiplier, steps,
                 prev_prev_scale, prev_scale, scale, conv_candidates,
                 ):
        super(GumbelCell, self).__init__()

        # todo add new attribute, for debugging
        self.layer = layer

        index2scale = {
            -2: 1,
            -1: 2,
            0: 4,
            1: 8,
            2: 16,
            3: 32,
        }
        self.steps = steps # nodes within each cell

        # todo add new attribute
        self.total_nodes = 2 + self.steps # exclude output node

        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.scale = scale
        self.conv_candidates = conv_candidates

        self.prev_prev_scale = prev_prev_scale
        self.prev_scale = prev_scale

        self.outc = int(self.filter_multiplier * self.block_multiplier * index2scale[self.scale] / 4)

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
            self.prev_c = self.prev_scale # fixed
            if self.scale == 0:
                self.preprocess1 = ConvLayer(self.prev_c, self.outc, 1, 1, False)
            elif self.scale == 1:
                self.preprocess1 = FactorizedReduce(self.prev_c, self.outc)
            else:
                raise ValueError('invalid scale value')

        if self.prev_prev_scale is None:
            self.prev_prev_c = None
            self.preprocess0 = None
        else:
            if index2scale.get(self.prev_prev_scale) is None: # fixed
                self.prev_prev_c = self.prev_prev_scale
            else:
                self.prev_prev_c = int(self.filter_multiplier * self.block_multiplier * index2scale[self.prev_prev_scale] / 4)
            # TODO: issue in scale of prev_prev_c, it is considered as next_scale by default
            self.preprocess0 = ConvLayer(self.prev_prev_c, self.outc, 1, 1, False)



        # todo, new attribute nn.ModuleDict()
        self.ops = nn.ModuleDict()
        # i::node_index, j::previous_node_index
        for i in range(2, self.total_nodes):
            for j in range(i):
                edge_str = '{:}<-{:}'.format(i, j)
                if j == 0 and self.prev_prev_scale is None: # for prev_prev_cell
                    mobile_inverted_conv = None
                    shortcut = None
                else:
                    mobile_inverted_conv = MixedOp(
                        build_candidate_ops(self.conv_candidates,
                        in_channels=self.outc, out_channels=self.outc, stride=1,
                        ops_order='act_weight_bn')) # normal MixedOp, ModuleList with weight
                    shortcut = Identity(self.outc, self.outc)
                if mobile_inverted_conv is None and shortcut is None:
                    inverted_residual_block = None
                else:
                    inverted_residual_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                self.ops[edge_str] = inverted_residual_block

        self.finalconv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)

        self.edge_keys = sorted(list(self.ops.keys()))
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.nb_edges = len(self.ops)

        self.cell_arch_parameters = nn.Parameter(torch.Tensor(self.nb_edges, self.n_choice))


    @property
    def n_choice(self):
        return len(self.conv_candidates)

    def forward(self, s0, s1, weights):
        # s0 and s1 are the output of prev_prev_cell and prev_cell, respectively.
        # weights is importance of operations, have been sorted.

        # s0 is none, self.prev_prev_scale is None, self.prev_prev_c is none, self.preprocess0 is None


        if s0 is not None:
            assert self.preprocess0 is not None, 'preprocess0 and s0 are inconsistent '
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1] # features of node0 and node1

        for i in range(2, self.total_nodes): # current node index
            new_states = []
            for j in range(i): # previous nodes
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]
                # self.ops has None operation when self.prev_prev_scale is none and j == 0
                if self.ops[edge_str] is None:
                    assert related_hidden is None, \
                        'None operation branch and None prev_prev_input are inconsistent'
                    continue
                weight = weights[branch_index]
                new_state = self.ops[edge_str](
                    related_hidden, weight
                )
                new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        # concatenation of last self.steps intermediate tensors
        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.finalconv1x1(concat_feature)

        return concat_feature

    def forward_gdas(self, s0, s1, hardwts, index):
        # s0 and s1 are the output of prev_prev_cell and prev_cell, respectively.
        # weights is importance of operations, have been sorted.

        # s0 is none, self.prev_prev_scale is None, self.prev_prev_c is none, self.preprocess0 is None

        #print(self.prev_prev_scale, self.prev_scale)

        if s0 is not None:
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(2, self.total_nodes): # node_index, excluding the first two node
            new_states = []
            for j in range(i): # edges for each node, j-th previous node
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]
                if self.ops[edge_str] is None or related_hidden is None:
                    # TODO: pay attention here, related to process of prev_prev_feature in sampled single_path in AutoDeepLab
                    #assert  related_hidden is None, 'inconsistent action of cell operations and prev_prev_cell'
                    continue
                weight = hardwts[branch_index]
                argmax = index[branch_index].item()
                new_state = self.ops[edge_str].forward_gdas(related_hidden, weight, argmax) # edge output of a node
                new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.finalconv1x1(concat_feature)

        return concat_feature


    def get_flops(self, s0, s1):
        # TODO: network flops should be calculated after derived!!!
        flop_preprocess0 = 0.
        if s0 is not None:
            flop_preprocess0, s0 = self.preprocess0.get_flops(s0)
        flop_preprcess1, s1 = self.preprocess1.get_flops(s1)
        states = [s0, s1]
        for i in range(2, self.total_nodes):
            new_states = []
            for j in range(i):
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]

    def forward_validate_test(self, s0, s1):
        # in validate phrase, use normal gdas_forward,
        # in testing phrase, it needs to derive the best network, then construct best network, perform normal forward
        raise NotImplementedError



    def get_flops(self):
        raise NotImplementedError

    def module_str(self):
        raise NotImplementedError