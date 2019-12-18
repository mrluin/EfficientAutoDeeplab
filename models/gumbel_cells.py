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


autodeeplab = [
    #'Zero'          , 'Identity'      ,
    #'3x3_MaxPooling', '3x3_AvgPooling',
    '3x3_DWConv'    , '5x5_DWConv'    ,
    '3x3_DilConv'   , '5x5_DilConv'
]
proxyless = [
    '3x3_MBConv3', '3x3_MBConv6',
    '5x5_MBConv3', '5x5_MBConv6',
    '7x7_MBConv3', '7x7_MBConv6',
    'Zero',  # 'Identity'
]
counter = [
    '3x3_DilConv'
]
# non-bottleneck architecture
# TODO: in sufficient training, use bottleneck layer by default
my_search_space = [
    '3x3_SepFacConv1', '5x5_SepFacConv1' ,
    '3x3_SepFacConv2', '5x5_SepFacConv2',
    '3x3_SepFacConv4', '5x5_SepFacConv4',
    #'3x3_SepFacConv8', '5x5_SepFacConv8',
    'Zero',
    #'Identity',
]


def build_candidate_ops(candiate_ops, in_channels, out_channels, stride, ops_order, affine=True):

    # learnable affine parameter is turn off in search phase.
    # learnable affine parameter is set to be learnable in retrain and test phase by default.
    if candiate_ops is None:
        raise ValueError('Please specify a candidate set')

    # None zero  layer
    # Identity   skip-connection
    name2ops = {
        'Identity': lambda inc, outc, s, affine: Identity(inc, outc, ops_order=ops_order, affine=affine),
        'Zero'    : lambda inc, outc, s, affine: Zero(s),
    }
    # add MBConv Layers
    name2ops.update({
        '3x3_MBConv1': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 1, affine=affine),
        '3x3_MBConv2': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 2, affine=affine),
        '3x3_MBConv3': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 3, affine=affine),
        '3x3_MBConv4': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 4, affine=affine),
        '3x3_MBConv5': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 5, affine=affine),
        '3x3_MBConv6': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 3, s, 6, affine=affine),
        '5x5_MBConv1': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 1, affine=affine),
        '5x5_MBConv2': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 2, affine=affine),
        '5x5_MBConv3': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 3, affine=affine),
        '5x5_MBConv4': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 4, affine=affine),
        '5x5_MBConv5': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 5, affine=affine),
        '5x5_MBConv6': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 5, s, 6, affine=affine),
        '7x7_MBConv1': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 1, affine=affine),
        '7x7_MBConv2': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 2, affine=affine),
        '7x7_MBConv3': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 3, affine=affine),
        '7x7_MBConv4': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 4, affine=affine),
        '7x7_MBConv5': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 5, affine=affine),
        '7x7_MBConv6': lambda inc, outc, s, affine: MBInvertedConvLayer(inc, outc, 7, s, 6, affine=affine),
        #===========================================================================
        '3x3_DWConv'    : lambda inc, outc, s, affine: SepConv(inc, outc, 3, s, affine=affine),
        '5x5_DWConv'    : lambda inc, outc, s, affine: SepConv(inc, outc, 5, s, affine=affine),
        '3x3_DilConv'   : lambda inc, outc, s, affine: DilConv(inc, outc, 3, s, 2, affine=affine),
        '5x5_DilConv'   : lambda inc, outc, s, affine: DilConv(inc, outc, 5, s, 2, affine=affine),
        '3x3_AvgPooling': lambda inc, outc, s, affine: nn.AvgPool2d(3, stride=s, padding=1, count_include_pad=False),
        '3x3_MaxPooling': lambda inc, outc, s, affine: nn.MaxPool2d(3, stride=s, padding=1),
        # ==========================================================================
        '3x3_FacConv1'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 3, s, 1, affine=affine),
        '3x3_FacConv2'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 3, s, 2, affine=affine),
        '3x3_FacConv4'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 3, s, 4, affine=affine),
        '3x3_FacConv8'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 3, s, 8, affine=affine),
        '3x3_FacConv16'   : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 3, s, 16, affine=affine),
        '5x5_FacConv1'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 5, s, 1, affine=affine),
        '5x5_FacConv2'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 5, s, 2, affine=affine),
        '5x5_FacConv4'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 5, s, 4, affine=affine),
        '5x5_FacConv8'    : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 5, s, 8, affine=affine),
        '5x5_FacConv16'   : lambda inc, outc, s, affine: FactorizedConvBlock(inc, outc, 5, s, 16, affine=affine),
        '3x3_SepConv1'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 3, s, 1, affine=affine),
        '3x3_SepConv2'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 3, s, 2, affine=affine),
        '3x3_SepConv4'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 3, s, 4, affine=affine),
        '3x3_SepConv8'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 3, s, 8, affine=affine),
        '3x3_SepConv16'   : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 3, s, 16, affine=affine),
        '5x5_SepConv1'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 5, s, 1, affine=affine),
        '5x5_SepConv2'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 5, s, 2, affine=affine),
        '5x5_SepConv4'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 5, s, 4, affine=affine),
        '5x5_SepConv8'    : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 5, s, 8, affine=affine),
        '5x5_SepConv16'   : lambda inc, outc, s, affine: SeparableConvBlock(inc, outc, 5, s, 16, affine=affine),
        '3x3_SepFacConv1' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 3, s, dilation=1, affine=affine),
        '3x3_SepFacConv2' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 3, s, dilation=2, affine=affine),
        '3x3_SepFacConv4' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 3, s, dilation=4, affine=affine),
        '3x3_SepFacConv8' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 3, s, dilation=8, affine=affine),
        '3x3_SepFacConv16': lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 3, s, dilation=16, affine=affine),
        '5x5_SepFacConv1' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 5, s, dilation=1, affine=affine),
        '5x5_SepFacConv2' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 5, s, dilation=2, affine=affine),
        '5x5_SepFacConv4' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 5, s, dilation=4, affine=affine),
        '5x5_SepFacConv8' : lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 5, s, dilation=8, affine=affine),
        '5x5_SepFacConv16': lambda inc, outc, s, affine: SepFacConvBlock(inc, outc, 5, s, dilation=16, affine=affine),
    })
    return [
        name2ops[name](in_channels, out_channels, stride, affine) for name in candiate_ops
    ]


class MixedOp(MyModule):
    def __init__(self, candidates_ops):
        super(MixedOp, self).__init__()
        self.candidate_ops = nn.ModuleList(candidates_ops)

    def forward(self, x, weight):
        return sum(w * op(x) for w, op in zip(weight, self.candidate_ops))

    def forward_single(self, x):
        assert len(self.candidate_ops) == 1, 'invalid len of self.candidate_ops in single_operation forward'
        return sum(op(x) for op in self.candidate_ops)

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


'''
# GumbelCell is used to search
# NewGumbelCell is used to construct derived network.
'''

class GumbelCell(MyModule):
    def __init__(self, layer,
                 filter_multiplier, block_multiplier, steps, scale, search_space,
                 ppc=None, pc=None, affine=True):
        super(GumbelCell, self).__init__()

        # todo add new attribute, affine parameter for bn, making searching phase more stable
        self.affine = affine
        # todo add new attribute, for debugging
        self.layer = layer
        # change index2scale to index2channel
        # index -2 and -1 is set by default
        # index 0, 1, 2, 3, 4 are calculated by int(filter_multiplier * block_multiplier * scale /4)
        self.index2scale = {
            0: 4,
            1: 8,
            2: 16,
            3: 32,
        }
        self.index2channel = {
            0: int(filter_multiplier * block_multiplier * self.index2scale[0] / 4),
            1: int(filter_multiplier * block_multiplier * self.index2scale[1] / 4),
            2: int(filter_multiplier * block_multiplier * self.index2scale[2] / 4),
            3: int(filter_multiplier * block_multiplier * self.index2scale[3] / 4),
        }
        self.steps = steps # nodes within each cell
        # todo add new attribute
        self.total_nodes = 2 + self.steps # exclude output node

        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.scale = scale

        self.search_space = search_space
        if self.search_space == 'autodeeplab':
            self.conv_candidates = autodeeplab
        elif self.search_space == 'proxyless':
            self.conv_candidates = proxyless
        elif self.search_space == 'counter': # used to debug
            self.conv_candidates = counter
        elif self.search_space == 'my_search_space':
            self.conv_candidates = my_search_space
        else:
            raise ValueError('search space {:} is not support'.format(self.search_space))
        #self.conv_candidates = conv_candidates
        #self.prev_prev_scale = prev_prev_scale
        #self.prev_scale = prev_scale
        self.outc = self.index2channel[self.scale]

        # TODO: do not need prev_prev_scale and prev_scale any more
        # 1. down same up link for prev_feature
        # 2. down same up, double down, and double up link for prev_prev_feature
        # 3. all the link operations are defined in __init__
        # 4. justification in forward() pass, and call the related link operation
        # 5. set prev_feature_channels and prev_prev_feature_channels specifically for output of stem0 and stem1

        # set types of link operation according to self.scale
        if self.scale == 0:
            # only has same and up link for prev_feature
            # only has same, up, and double up link for prev_prev_feature
            self.same_link_prev           = ConvLayer(self.outc if pc is None else pc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev             = FactorizedIncrease(int(self.outc*2) if pc is None else pc, self.outc, affine=affine)
            self.same_link_prev_prev      = ConvLayer(self.outc if ppc is None else ppc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev_prev        = FactorizedIncrease(int(self.outc*2) if ppc is None else ppc, self.outc, affine=affine)
            self.double_up_link_prev_prev = DoubleFactorizedIncrease(int(self.outc*4) if ppc is None else ppc, self.outc, affine=affine)
            # has down for prev_prev_feature in layer-0
            self.down_link_prev_prev      = FactorizedReduce(int(self.outc/2) if ppc is None else ppc, self.outc, affine=affine)
        elif self.scale == 1:
            # has down, same, up link for prev_feature
            # has down, same, up, and double up link for prev_prev_feature
            self.down_link_prev             = FactorizedReduce(int(self.outc/2) if pc is None else pc, self.outc, affine=affine)
            self.same_link_prev             = ConvLayer(self.outc if pc is None else pc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev               = FactorizedIncrease(int(self.outc*2) if pc is None else pc, self.outc, affine=affine)
            self.down_link_prev_prev        = FactorizedReduce(int(self.outc/2) if ppc is None else ppc, self.outc, affine=affine)
            self.same_link_prev_prev        = ConvLayer(self.outc if ppc is None else ppc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev_prev          = FactorizedIncrease(int(self.outc*2) if ppc is None else ppc, self.outc, affine=affine)
            self.double_up_link_prev_prev   = DoubleFactorizedIncrease(int(self.outc*4) if ppc is None else ppc, self.outc, affine=affine)
            # has double down link for prev_prev_feature
            self.double_down_link_prev_prev = DoubleFactorizedReduce(int(self.outc/4) if ppc is None else ppc, self.outc, affine=affine)
        elif self.scale == 2:
            # has down, same, up link for prev_feature
            # has ddown, same, up link for prev_prev_feature
            self.down_link_prev             = FactorizedReduce(int(self.outc/2) if pc is None else pc, self.outc, affine=affine)
            self.same_link_prev             = ConvLayer(self.outc if pc is None else pc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev               = FactorizedIncrease(int(self.outc*2) if pc is None else pc, self.outc, affine=affine)
            self.down_link_prev_prev        = FactorizedReduce(int(self.outc/2) if ppc is None else ppc, self.outc, affine=affine)
            self.double_down_link_prev_prev = DoubleFactorizedReduce(int(self.outc/4) if ppc is None else ppc, self.outc, affine=affine)
            self.same_link_prev_prev        = ConvLayer(self.outc if ppc is None else ppc, self.outc, 1, 1, False, affine=affine)
            self.up_link_prev_prev          = FactorizedIncrease(int(self.outc*2) if ppc is None else ppc, self.outc, affine=affine)
        elif self.scale == 3:
            # has down, same link for prev_feature
            # has ddown, down, and same for prev_prev_feature
            self.down_link_prev             = FactorizedReduce(int(self.outc/2) if pc is None else pc, self.outc, affine=affine)
            self.same_link_prev             = ConvLayer(self.outc if pc is None else pc, self.outc, 1, 1, False, affine=affine)
            self.double_down_link_prev_prev = DoubleFactorizedReduce(int(self.outc/4) if ppc is None else ppc, self.outc, affine=affine)
            self.down_link_prev_prev        = FactorizedReduce(int(self.outc/2) if ppc is None else ppc, self.outc, affine=affine)
            self.same_link_prev_prev        = ConvLayer(self.outc if ppc is None else ppc, self.outc, 1, 1, False, affine=affine)
        else:
            raise ValueError('invalid scale value {:}'.format(self.scale))

        # todo, new attribute nn.ModuleDict()
        self.ops = nn.ModuleDict()
        # i::node_index, j::previous_node_index
        if self.search_space == 'proxyless':
            for i in range(2, self.total_nodes):
                for j in range(i):
                    edge_str = '{:}<-{:}'.format(i, j)
                    #if j == 0 and self.prev_prev_scale is None:  # for prev_prev_cell
                    #    mobile_inverted_conv = None
                    #    shortcut = None
                    #else:
                    mobile_inverted_conv = MixedOp(
                        build_candidate_ops(self.conv_candidates,
                                            in_channels=self.outc, out_channels=self.outc, stride=1,
                                            ops_order='act_weight_bn', affine=self.affine))  # normal MixedOp, ModuleList with weight
                    shortcut = Identity(self.outc, self.outc)
                    #if mobile_inverted_conv is None and shortcut is None:
                    #    inverted_residual_block = None
                    #else:
                    inverted_residual_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                    self.ops[edge_str] = inverted_residual_block
        elif self.search_space == 'autodeeplab' or self.search_space == 'my_search_space':
            # TODO: have issue in search space of autodeeplab
            for i in range(2, self.total_nodes):
                for j in range(i):
                    edge_str = '{:}<-{:}'.format(i, j)
                    #if j == 0 and self.prev_prev_scale is None:
                    #    op = None
                    #else:
                    op = MixedOp(build_candidate_ops(self.conv_candidates, in_channels=self.outc, out_channels=self.outc, stride=1,
                                                ops_order='act_weight_bn', affine=self.affine))
                    self.ops[edge_str] = op
        else:
            raise ValueError('search space {:} is not supported'.format(self.search_space))

        self.finalconv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)

        self.edge_keys = sorted(list(self.ops.keys())) # 'sorted by {:}<-{:}'
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)} # {:}<-{:} : index
        self.nb_edges = len(self.ops)

        #self.cell_arch_parameters = nn.Parameter(torch.Tensor(self.nb_edges, self.n_choice))
        self.cell_arch_parameters = nn.Parameter(1e-3 * torch.randn(self.nb_edges, self.n_choice))

    @property
    def n_choice(self):
        return len(self.conv_candidates)

    def forward(self, s0, s1, weights):
        # s0 and s1 always exist!
        current_size = 512 / self.index2scale[self.scale]
        if s0 is not None:
            s0_size = s0[0].size()[2]
            if s0_size / current_size == 4.: # double down
                s0 = self.double_down_link_prev_prev(s0)
            elif s0_size / current_size == 2.: # down
                s0 = self.down_link_prev_prev(s0)
            elif s0_size / current_size == 1.: # same
                s0 = self.same_link_prev_prev(s0)
            elif s0_size / current_size == 1/2: # up
                s0 = self.up_link_prev_prev(s0)
            elif s0_size / current_size == 1/4: # double up
                s0 = self.double_up_link_prev_prev(s0)
            else:
                raise ValueError('invalid size relation s0_size / current_size = {:} in gdas_forward pass'.format(s0_size / current_size))
        if s1 is not None:
            s1_size = s1[0].size()[2]
            if s1_size / current_size == 2.: # down
                s1 = self.down_link_prev(s1)
            elif s1_size / current_size == 1.: # same
                s1 = self.same_link_prev(s1)
            elif s1_size / current_size == 1/2: # up
                s1 = self.up_link_prev(s1)
            else:
                raise ValueError('invalid size relation s1_size / current_size = {:} in gdas_forward pass'.format(s1_size / current_size))

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

        # s0 and s1 always exist!
        # check s0.shape and s1.shape [B, C, H, W]
        #print(s0)
        #print(s1)
        #print(self.layer)

        s0_size = s0[0].size()[2]
        s1_size = s1[0].size()[2]
        # TODO: change 512 to argument
        current_size = 512 / self.index2scale[self.scale]
        #print(s0_size, s1_size)
        #print(current_size)
        #print(s0.shape)
        #print(s1.shape)

        if s0_size / current_size == 4.: # double down
            s0 = self.double_down_link_prev_prev(s0)
        elif s0_size / current_size == 2.: # down
            s0 = self.down_link_prev_prev(s0)
        elif s0_size / current_size == 1.: # same
            s0 = self.same_link_prev_prev(s0)
        elif s0_size / current_size == 1/2: # up
            s0 = self.up_link_prev_prev(s0)
        elif s0_size / current_size == 1/4: # double up
            s0 = self.double_up_link_prev_prev(s0)
        else:
            raise ValueError('invalid size relation s0_size / current_size = {:} in gdas_forward pass'.format(s0_size / current_size))

        if s1_size / current_size == 2.: # down
            s1 = self.down_link_prev(s1)
        elif s1_size / current_size == 1.: # same
            s1 = self.same_link_prev(s1)
        elif s1_size / current_size == 1/2: # up
            s1 = self.up_link_prev(s1)
        else:
            raise ValueError('invalid size relation s1_size / current_size = {:} in gdas_forward pass'.format(s1_size / current_size))

        #s0 = self.preprocess0(s0)
        #s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(2, self.total_nodes): # node_index, excluding the first two node
            new_states = []
            for j in range(i): # edges for each node, j-th previous node
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]
                #if self.ops[edge_str] is None or related_hidden is None:
                #    continue
                weight = hardwts[branch_index]
                argmax = index[branch_index].item()
                new_state = self.ops[edge_str].forward_gdas(related_hidden, weight, argmax) # edge output of a node
                new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.finalconv1x1(concat_feature)
        #print(concat_feature.shape)
        return concat_feature


    def get_flops(self, s0, s1):
        # TODO: get rid of get_flops,
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


    def module_str(self):
        raise NotImplementedError