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
    Zero, SepConv, DilConv, MobileInvertedResidualBlock, DoubleFactorizedIncrease, DoubleFactorizedReduce
from run_manager import RunConfig
from utils.common import get_prev_c
from models.gumbel_cells import build_candidate_ops, proxyless, autodeeplab,counter, my_search_space
__all__ = ['NewGumbelCell', 'NewGumbelAutoDeeplab']

class NewGumbelCell(MyModule):
    def __init__(self, layer, filter_multiplier, block_multiplier, steps,
                 scale, genotype, search_space, ppc=None, pc=None, affine=True):
        super(NewGumbelCell, self).__init__()
        self.index2scale = {
            0: 4,
            1: 8,
            2: 16,
            3: 32,
        }
        if filter_multiplier == 16:
            # small setting
            self.index2channels = {
                0: 16,
                1: 32,
                2: 64,
                3: 128,
            }
        elif filter_multiplier == 32:
            # medium setting
            self.index2channels = {
                0: 32,
                1: 64,
                2: 128,
                3: 256,
            }
        elif filter_multiplier == 40:
            self.index2channels = {
                0: 40,
                1: 80,
                2: 160,
                3: 320,
            }
        elif filter_multiplier == 64:
            # large setting
            self.index2channels = {
                0: 64,
                1: 128,
                2: 256,
                3: 512,
            }
        elif filter_multiplier == 50:
            self.index2channels = {
                0: 50,
                1: 100,
                2: 200,
                3: 400,
            }
        else:
            raise ValueError('filter_multiplier {:} do not support for index2channels in new_gumbel_model'.format(filter_multiplier))

        self.total_nodes = 2 + steps
        self.layer = layer
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        #self.prev_prev_scale = prev_prev_scale
        #self.prev_scale = prev_scale
        self.scale = scale
        self.genotype = genotype
        self.search_space = search_space
        if self.search_space == 'autodeeplab':
            self.conv_candidates = autodeeplab
        elif self.search_space == 'proxyless':
            self.conv_candidates = proxyless
        elif self.search_space == 'counter':
            self.conv_candidates = counter
        elif self.search_space == 'my_search_space':
            self.conv_candidates = my_search_space
        else:
            raise ValueError('search space {:} is not support'.format(self.search_space))
        #self.conv_candidates = search_space_dict[self.search_space]
        #self.outc = int(self.filter_multiplier * self.block_multiplier * self.scale / 4)
        self.outc = self.index2channels[self.scale]

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

        self.ops = nn.ModuleDict()
        # fix issue of construct new_cell of proxyless search space
        if self.search_space == 'proxyless':
            for node_str, select_op_index in self.genotype[1][0]:
                conv_op = build_candidate_ops([self.conv_candidates[select_op_index]], self.outc, self.outc,
                                            stride=1, ops_order='act_weight_bn', affine=affine)
                shortcut = Identity(self.outc, self.outc)
                self.ops[node_str] = MobileInvertedResidualBlock(conv_op, shortcut)
        elif self.search_space == 'autodeeplab' or self.search_space == 'my_search_space':
            # TODO: have modification on genotypes pass ( cell_index, [[(edge_str), (operation)], ] )
            #print(self.genotype) # (cell_index, [[('edge_str', index), ('edge_str', index), ], ])
            for node_str, select_op_index in self.genotype[1][0]:
                # operation name -> conv_candidates
                operation = build_candidate_ops([self.conv_candidates[select_op_index]], self.outc, self.outc,
                                                stride=1, ops_order='act_weight_bn', affine=affine)

                self.ops[node_str] = operation[0]
        else:
            raise ValueError('search space {:} is not support'.format(self.search_space))
        self.final_conv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)


        self.included_edges = []
        for genotype_for_each_node in self.genotype[1]:
            for edge_str, _ in genotype_for_each_node:
                self.included_edges.append(edge_str)

        print(self.included_edges)
        
    def forward(self, s0, s1):

        s0_size = s0[0].size()[2]
        s1_size = s1[0].size()[2]
        # TODO: change 512 to argument
        current_size = 512 / self.index2scale[self.scale]
        # print(s0_size, s1_size)
        # print(current_size)

        if s0_size / current_size == 4.:  # double down
            s0 = self.double_down_link_prev_prev(s0)
        elif s0_size / current_size == 2.:  # down
            s0 = self.down_link_prev_prev(s0)
        elif s0_size / current_size == 1.:  # same
            s0 = self.same_link_prev_prev(s0)
        elif s0_size / current_size == 1 / 2:  # up
            s0 = self.up_link_prev_prev(s0)
        elif s0_size / current_size == 1 / 4:  # double up
            s0 = self.double_up_link_prev_prev(s0)
        else:
            raise ValueError('invalid size relation s0_size / current_size = {:} in gdas_forward pass'.format(
                s0_size / current_size))

        if s1_size / current_size == 2.:  # down
            s1 = self.down_link_prev(s1)
        elif s1_size / current_size == 1.:  # same
            s1 = self.same_link_prev(s1)
        elif s1_size / current_size == 1 / 2:  # up
            s1 = self.up_link_prev(s1)
        else:
            raise ValueError('invalid size relation s1_size / current_size = {:} in gdas_forward pass'.format(
                s1_size / current_size))

        states = [s0, s1] # including None state

        for i in range(2, self.total_nodes):
            new_states = []
            for j in range(i): # all previous node for each node i
                node_str = '{:}<-{:}'.format(i,j)
                if node_str in self.included_edges:
                    # the current edge is selected
                    related_hidden = states[j]
                    # forward :: MBConvResidualBlock -> MixedOp -> single_forward
                    # forward :: normal ConvBlock -> simple forward
                    new_state = self.ops[node_str](related_hidden)
                    new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.final_conv1x1(concat_feature)
        return concat_feature

class NewGumbelAutoDeeplab(MyNetwork):
    def __init__(self, nb_layers, filter_multiplier, block_multiplier, steps, nb_classes,
                 actual_path, cell_genotypes, search_space, affine=True):
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
            ('conv', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(16)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        # remove 'relu' for self.stem1 # ('relu', nn.ReLU(inplace=True))
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(16)),
        ]))
        # change the order of the stem2
        self.stem2 = nn.Sequential(OrderedDict([
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(16, self.filter_multiplier, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(self.filter_multiplier)),
        ]))

        self.cells = nn.ModuleList()
        prev_prev_c = 16
        prev_c = self.filter_multiplier
        for layer in range(self.nb_layers):
            next_scale = int(self.actual_path[layer])
            cell_genotype = self.cell_genotypes[layer] # next_scale cell genotype
            if layer == 0:
                cell = NewGumbelCell(layer, filter_multiplier, block_multiplier, steps, next_scale,
                                     cell_genotype, search_space, ppc=prev_prev_c, pc=prev_c, affine=affine)
            elif layer == 1:
                cell = NewGumbelCell(layer, filter_multiplier, block_multiplier, steps, next_scale,
                                     cell_genotype, search_space, ppc=prev_c, pc=None, affine=affine)
            else:
                cell = NewGumbelCell(layer, filter_multiplier, block_multiplier, steps, next_scale,
                                     cell_genotype, search_space, ppc=None, pc=None, affine=affine)

            self.cells.append(cell)

        self.out0 = int(self.filter_multiplier * self.block_multiplier * 4 / 4)
        self.out1 = int(self.filter_multiplier * self.block_multiplier * 8 / 4)
        self.out2 = int(self.filter_multiplier * self.block_multiplier * 16 / 4)
        self.out3 = int(self.filter_multiplier * self.block_multiplier * 32 / 4)
        last_scale = int(self.actual_path[-1])
        if last_scale == 0:
            self.aspp = ASPP(self.out0 ,self.nb_classes, 24, affine=affine)
        elif last_scale == 1:
            self.aspp = ASPP(self.out1, self.nb_classes, 12, affine=affine)
        elif last_scale == 2:
            self.aspp = ASPP(self.out2, self.nb_classes, 6, affine=affine)
        elif last_scale == 3:
            self.aspp = ASPP(self.out3, self.nb_classes, 3, affine=affine)
        else:
            raise ValueError('invalid last_scale value {}'.format(last_scale))

    def forward(self, x):

        size = x.size()[2:]
        # chain-like structure, normal forward cells and last aspp
        inter_features = []
        x = self.stem0(x)
        x = self.stem1(x)
        inter_features.append(x)
        x = self.stem2(x)
        inter_features.append(x)

        for layer in range(self.nb_layers):
            #next_scale = int(self.actual_path[layer])
            prev_prev_feature, prev_feature = inter_features[-2], inter_features[-1]
            _result = self.cells[layer](prev_prev_feature, prev_feature)
            inter_features.pop(0)
            inter_features.append(_result)

        _result = self.aspp(inter_features[-1])
        return F.interpolate(_result, size=size, mode='bilinear', align_corners=True)

def get_new_model(super_network, run_config: RunConfig):
    nb_layers = run_config.nb_layers
    filter_multiplier = run_config.filter_multiplier
    block_multiplier = run_config.block_multiplier
    search_space = run_config.search_space
    steps = run_config.steps
    nb_classes = run_config.nb_classes
    actual_path, cell_genotypes = super_network.network_cell_arch_decode()
    print('\t=> Construct New Model ... ...')
    normal_model = NewGumbelAutoDeeplab(nb_layers, filter_multiplier, block_multiplier, steps, nb_classes, actual_path, cell_genotypes, search_space)
    print('\t=> New Model Constructed Done ... ... Begin Testing')

    data = torch.zeros(1, 3, 512, 512).to('cuda:{}'.format(run_config.gpu_ids))
    normal_model(data)
    print('\t=> Testing Done.')
    return normal_model
