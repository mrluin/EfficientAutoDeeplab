# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.my_modules import MyNetwork
from models.gumbel_cells import proxyless, autodeeplab, my_search_space, GumbelCell
from collections import OrderedDict
from modules.operations import ASPP

class FixedNetwork(MyNetwork):
    def __init__(self,
                 filter_multiplier, block_multiplier, steps, nb_classes, nb_layers, search_space, logger, affine=False):
        super(FixedNetwork, self).__init__()
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        self.search_space = search_space
        if search_space == 'proxyless':
            self.conv_candidates = proxyless
        elif search_space == 'autodeeplab':
            self.conv_candidates = autodeeplab
        elif search_space == 'my_search_space':
            self.conv_candidates = my_search_space

        self.logger = logger
        self.affine = affine

        self.cells = nn.ModuleList()

        self.stem0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(16)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


        cell0 = GumbelCell(0, filter_multiplier, block_multiplier, steps, 1, search_space, ppc=16, pc=32, affine=affine)
        cell1 = GumbelCell(1, filter_multiplier, block_multiplier, steps, 1, search_space, ppc=32, pc=None, affine=affine)
        cell2 = GumbelCell(2, filter_multiplier, block_multiplier, steps, 1, search_space, ppc=None, pc=None, affine=affine)
        cell3 = GumbelCell(3, filter_multiplier, block_multiplier, steps, 1, search_space, ppc=None, pc=None, affine=affine)
        cell4 = GumbelCell(4, filter_multiplier, block_multiplier, steps, 2, search_space, ppc=None, pc=None, affine=affine)
        cell5 = GumbelCell(5, filter_multiplier, block_multiplier, steps, 2, search_space, ppc=None, pc=None, affine=affine)
        cell6 = GumbelCell(6, filter_multiplier, block_multiplier, steps, 2, search_space, ppc=None, pc=None, affine=affine)
        cell7 = GumbelCell(7, filter_multiplier, block_multiplier, steps, 2, search_space, ppc=None, pc=None, affine=affine)
        cell8 = GumbelCell(8, filter_multiplier, block_multiplier, steps, 3, search_space, ppc=None, pc=None, affine=affine)
        cell9 = GumbelCell(9, filter_multiplier, block_multiplier, steps, 2, search_space, ppc=None, pc=None, affine=affine)
        cell10 = GumbelCell(10, filter_multiplier, block_multiplier, steps, 1, search_space, ppc=None, pc=None, affine=affine)
        cell11 = GumbelCell(11, filter_multiplier, block_multiplier, steps, 0, search_space, ppc=None, pc=None, affine=affine)

        aspp_inc = int(filter_multiplier * block_multiplier * 4 / 4)

        # TODO: with or without ASPP
        #self.aspp = ASPP(aspp_inc, nb_classes, dilation=16, affine=affine)

        self.cells.append(cell0)
        self.cells.append(cell1)
        self.cells.append(cell2)
        self.cells.append(cell3)
        self.cells.append(cell4)
        self.cells.append(cell5)
        self.cells.append(cell6)
        self.cells.append(cell7)
        self.cells.append(cell8)
        self.cells.append(cell9)
        self.cells.append(cell10)
        self.cells.append(cell11)

        self.nb_edges = len([j for i in range(self.steps) for j in range(i+2)])
        self.nb_choices = len(self.conv_candidates)
        self.cell_arch_parameters = nn.Parameter(1e-3 * torch.randn(self.nb_edges, self.nb_choices))
        self.total_nodes = 2 + self.steps
        self.edge2index = cell0.edge2index


    def get_cell_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'cell_arch_parameters' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'cell_arch_parameters' not in name:
                yield param

    def arch_parameters(self):
        for name, param in self.named_parameters():
            if 'cell_arch_parameters' in name:
                yield param

    def cell_genotype_decode(self):

        genotypes = []
        with torch.no_grad():
            total_nodes = self.total_nodes
            edge2index = self.edge2index
            weight = self.cell_arch_parameters
            for i in range(2, total_nodes):
                xlist = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    branch_index = edge2index[node_str]
                    mixed_op_weight = weight[branch_index]
                    select_op_index = mixed_op_weight.argmax().item(
                        0)  # select operation with the highest prob for each edge.
                    xlist.append((node_str, select_op_index))  # the highest prob operation for all incoming edges.
                previous_two = sorted(xlist, key=lambda x: -weight[edge2index[node_str]][x[1]])[:2]  # select the highest two for each node.
                genotypes.append(previous_two)
            return genotypes

    def network_cell_arch_decode(self):
        # network-level structure is fixed
        # decode cell-level structure
        actual_path = [1,1,1,1,2,2,2,2,3,2,1,0]
        cell_genotype = self.cell_genotype_decode()

        return actual_path, cell_genotype

    def forward(self, x):

        size = x.size()[-2:]
        inter_features = []

        x = self.stem0(x)
        inter_features.append(x)
        x = self.stem1(x)
        inter_features.append(x)

        count = 0
        for layer in range(self.nb_layers):
            _result = self.cells[count](inter_features[-2], inter_features[-1])
            inter_features.append(_result)
            count += 1

        result = F.interpolate(inter_features[-1], size=size, mode='bilinear', align_corners=True)
        return result