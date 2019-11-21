'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.gumbel_cells import GumbelCell
from modules.my_modules import MyNetwork
from modules.operations import ASPP
from collections import OrderedDict



__all__ = ['GumbelAutoDeepLab']

class GumbelAutoDeepLab(MyNetwork):
    def __init__(self,
                 filter_multiplier, block_multiplier, steps, nb_classes,
                 nb_layers, conv_candidates):
        super(GumbelAutoDeepLab, self).__init__()

        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.nb_layers = nb_layers
        self.nb_classes = nb_classes
        self.conv_candidates = conv_candidates

        self.cells = nn.ModuleList()

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

        # cells
        # TODO: prev_prev_scale is considered the same with next_scale, which is inconsistent with the authors'.
        prev_prev_c = 32
        prev_c = 64
        for i in range(self.nb_layers):
            if i == 0:
                cell1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                   prev_prev_scale=None, prev_scale=prev_c, scale=0, conv_candidates=self.conv_candidates)
                cell2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                   prev_prev_scale=None, prev_scale=prev_c, scale=1, conv_candidates=self.conv_candidates)
                self.cells += [cell1]  # 0
                self.cells += [cell2]  # 1
            elif i == 1:
                cell1_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=prev_c, prev_scale=0, scale=0, conv_candidates=self.conv_candidates)
                cell1_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=prev_c, prev_scale=1, scale=0, conv_candidates=self.conv_candidates)
                cell2_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=0, scale=1, conv_candidates=self.conv_candidates)
                cell2_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=1, scale=1, conv_candidates=self.conv_candidates)
                cell3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                   prev_prev_scale=None, prev_scale=1, scale=2, conv_candidates=self.conv_candidates)
                self.cells += [cell1_1]  # 3
                self.cells += [cell2_1]  # 4
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3]
            elif i == 2:
                cell1_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=0, scale=0, conv_candidates=self.conv_candidates)
                cell1_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=1, scale=0, conv_candidates=self.conv_candidates)
                cell2_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=0, scale=1, conv_candidates=self.conv_candidates)
                cell2_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=1, scale=1, conv_candidates=self.conv_candidates)
                cell2_3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=2, scale=1, conv_candidates=self.conv_candidates)
                cell3_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=1, scale=2, conv_candidates=self.conv_candidates)
                cell3_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=2, scale=2, conv_candidates=self.conv_candidates)
                cell4 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                   prev_prev_scale=None, prev_scale=2, scale=3, conv_candidates=self.conv_candidates)
                self.cells += [cell1_1]
                self.cells += [cell2_1]
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3_1]
                self.cells += [cell2_3]
                self.cells += [cell3_2]
                self.cells += [cell4]
            elif i == 3:
                cell1_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=0, scale=0, conv_candidates=self.conv_candidates)
                cell1_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=1, scale=0, conv_candidates=self.conv_candidates)
                cell2_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=0, scale=1, conv_candidates=self.conv_candidates)
                cell2_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=1, scale=1, conv_candidates=self.conv_candidates)
                cell2_3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=2, scale=1, conv_candidates=self.conv_candidates)
                cell3_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=1, scale=2, conv_candidates=self.conv_candidates)
                cell3_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=2, scale=2, conv_candidates=self.conv_candidates)
                cell3_3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=3, scale=2, conv_candidates=self.conv_candidates)
                cell4_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=2, scale=3, conv_candidates=self.conv_candidates)
                cell4_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=None, prev_scale=3, scale=3, conv_candidates=self.conv_candidates)
                self.cells += [cell1_1]
                self.cells += [cell2_1]
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3_1]
                self.cells += [cell2_3]
                self.cells += [cell3_2]
                self.cells += [cell4_1]
                self.cells += [cell3_3]
                self.cells += [cell4_2]
            else:
                cell1_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=0, scale=0, conv_candidates=self.conv_candidates)
                cell1_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=0, prev_scale=1, scale=0, conv_candidates=self.conv_candidates)
                cell2_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=0, scale=1, conv_candidates=self.conv_candidates)
                cell2_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=1, scale=1, conv_candidates=self.conv_candidates)
                cell2_3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=1, prev_scale=2, scale=1, conv_candidates=self.conv_candidates)
                cell3_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=1, scale=2, conv_candidates=self.conv_candidates)
                cell3_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=2, scale=2, conv_candidates=self.conv_candidates)
                cell3_3 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=2, prev_scale=3, scale=2, conv_candidates=self.conv_candidates)
                cell4_1 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=3, prev_scale=2, scale=3, conv_candidates=self.conv_candidates)
                cell4_2 = GumbelCell(i, self.filter_multiplier, self.block_multiplier, self.steps,
                                     prev_prev_scale=3, prev_scale=3, scale=3, conv_candidates=self.conv_candidates)
                self.cells += [cell1_1]
                self.cells += [cell2_1]
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3_1]
                self.cells += [cell2_3]
                self.cells += [cell3_2]
                self.cells += [cell4_1]
                self.cells += [cell3_3]
                self.cells += [cell4_2]

        print('\t nb_cells: ',len(self.cells))
        scale4_outc = int(self.filter_multiplier * self.block_multiplier * 4 / 4)
        scale8_outc = int(self.filter_multiplier * self.block_multiplier * 8 / 4)
        scale16_outc = int(self.filter_multiplier * self.block_multiplier * 16 / 4)
        scale32_outc = int(self.filter_multiplier * self.block_multiplier * 32 / 4)

        # dilation as 96/scale
        # TODO: get rid of redundant argument nb_classes and output c
        self.aspp4 = ASPP(scale4_outc, self.nb_classes, 24, self.nb_classes)
        self.aspp8 = ASPP(scale8_outc, self.nb_classes, 12, self.nb_classes)
        self.aspp16 = ASPP(scale16_outc, self.nb_classes, 6, self.nb_classes)
        self.aspp32 = ASPP(scale32_outc, self.nb_classes, 3, self.nb_classes)

        self.nb_cells = len(self.cells)
        self.network_arch_parameters = nn.Parameter(1e-3*torch.randn(self.nb_layers, 4, 3))
        self.tau = 10

        #
        self.nb_edges =
        self.edge2index =


    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_network_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'network_arch_parameters' in name:
                yield param

    def get_cell_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'cell_arch_parameters' in name:
                yield param


    def viterbi_decode(self):


    def genotype_decode(self):


    def cell_arch_decode(self):


    def forward(self, input):
        # forward for network_level and cell_level

        # 1. generate hardwts for super network
        # 2. _forward for each node
        # 3. generate hardwts for each cell
        # 4. forward for each cell


        def _forward():
            # TODO: generate hardwts for each cell in sampled network

        # TODO: generate hardwts for choices in super network





        while True:
            # network_arch_parameters shape [12, 4, 3]
            gumbels = -torch.empty_like(self.network_arch_parameters).exponential_().log()
            logits = (self.network_arch_parameters.log_softmax(dim=-1) + gumbels) / self.tau
            probs = F.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[1] # return tuple, (value, index) shape as [12, 4, 1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0) # shape as [12, 4, 3]
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else: break

        # obtained hardwts [12, 4, 3] one_h matrix


        # TODO: how to perform network_level forward pass, according to hardwts

        # refer auto_deeplab forward, based

        # think proxy method again


        current_scale = 0 # init_scale after stem2
        for layer in range(self.nb_layers):







