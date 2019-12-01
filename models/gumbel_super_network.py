'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.gumbel_cells import GumbelCell
from modules.my_modules import MyNetwork
from modules.operations import ASPP
from collections import OrderedDict

from utils.common import get_pfeatures, detect_inputs_shape, append_scale_list, get_prev_c, get_cell_index, \
    detect_invalid_index, count_normal_conv_flop

__all__ = ['GumbelAutoDeepLab']

class GumbelAutoDeepLab(MyNetwork):
    def __init__(self,
                 filter_multiplier, block_multiplier, steps, nb_classes,
                 nb_layers, bn_momentum, bn_eps, conv_candidates, logger):
        super(GumbelAutoDeepLab, self).__init__()

        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.nb_layers = nb_layers
        self.nb_classes = nb_classes
        self.conv_candidates = conv_candidates
        self.logger = logger

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
                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
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
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
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
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
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
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]

        #print('\t nb_cells: ',len(self.cells))
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
        self.network_arch_parameters = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))
        self.aspp_arch_parameters = nn.Parameter(torch.Tensor(4))
        self.tau = 10

        # todo
        #self.nb_edges =
        #self.edge2index =

        self.set_bn_param(bn_momentum, bn_eps)
    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.arch_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise ValueError('invalid arch_parameters init_type {:}'.format(init_type))

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

    def get_aspp_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'aspp_arch_parameters' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'network_arch_parameters' not in name and 'cell_arch_parameters' not in name:
                yield param

    def arch_parameters(self):
        # include cell_arch_parameters and network_arch_parameters
        for name, param in self.named_parameters():
            if 'network_arch_parameters' in name or 'cell_arch_parameters' in name or 'aspp_arch_parameters' in name:
                yield param

    def get_flops(self, x):

        # get_flops should performed on super network with actual_path
        actual_path = self.viterbi_decode()
        inter_features = []
        print('actual_path', actual_path)
        flops = 0.
        flop_stem0 = count_normal_conv_flop(self.stem0.conv, x)
        x = self.stem0(x)
        flop_stem1 = count_normal_conv_flop(self.stem1.conv, x)
        x = self.stem1(x)
        inter_features.append((-1, x))
        flop_stem2 = count_normal_conv_flop(self.stem2.conv, x)
        x = self.stem2(x)
        inter_features.append((0, x))

        current_scale = 0
        for layer in range(self.nb_layers):
            next_scale = int(actual_path[layer])
            prev_prev_feature, prev_feature = get_prev_c(inter_features, next_scale)
            cell_index = get_cell_index(layer, current_scale, next_scale)

            # TODO, uncompleted
            frag_flop, out = self.cells[cell_index].get_flops(prev_prev_feature, prev_feature)

            flops = flops + frag_flop
            current_scale = next_scale
            inter_features.pop(0)
            inter_features.append([next_scale, out])

        last_scale = int(actual_path[-1])
        if last_scale == 0:
            flop_aspp, output = self.aspp4.get_flops(inter_features[-1][1])
        elif last_scale == 1:
            flop_aspp, output = self.aspp8.get_flops(inter_features[-1][1])
        elif last_scale == 2:
            flop_aspp, output = self.aspp16.get_flops(inter_features[-1][1])
        elif last_scale == 3:
            flop_aspp, output = self.aspp32.get_flops(inter_features[-1][1])
        else:
            raise ValueError('invalid last_scale value {}'.format(last_scale))

        return flops + flop_stem0 + flop_stem1 + flop_stem2 + flop_aspp, output

    def viterbi_decode(self):
        # network_space           [12, 4, 3], 0-layer is output of stem2 0: ↗, 1: →, 2: ↘
        # network_arch_parameters [12, 4, 3], 0-layer is true 0-layer,   0: ↘, 1: →, 2: ↗, w.r.t. each node in fabric
        with torch.no_grad():
            network_space = torch.zeros_like(self.network_arch_parameters)
            #network_space = np.zeros((self.nb_layers, 4, 3))
            #aspp_space = self.aspp_arch_parameters
            for layer in range(self.nb_layers):
                if layer == 0:
                    network_space[layer][0][1] = F.softmax(self.network_arch_parameters[layer][0][1], -1)
                    network_space[layer][1][0] = F.softmax(self.network_arch_parameters[layer][1][0], -1)
                elif layer == 1:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1)
                    network_space[layer][1][:2] = F.softmax(self.network_arch_parameters[layer][1][:2], -1)
                    network_space[layer][2][0] = F.softmax(self.network_arch_parameters[layer][2][0], -1)
                elif layer == 2:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1)
                    network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                    network_space[layer][2][:2] = F.softmax(self.network_arch_parameters[layer][2][:2], -1)
                    network_space[layer][3][0] = F.softmax(self.network_arch_parameters[layer][3][0], -1)
                else:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1)
                    network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                    network_space[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
                    network_space[layer][3][:2] = F.softmax(self.network_arch_parameters[layer][3][:2], -1)

        print('viterbi_phase:\n', network_space)
        prob_space = np.zeros(network_space.shape[:2]) # [12, 4]
        path_space = np.zeros(network_space.shape[:2]).astype('int8') # [12, 4]

        # prob_space [layer, sample] from layer sample to next layer
        #   0 1 2 3 4 5 6 7 8 9 network_space from here
        # 0 1 2 3 4 5 6 7 8 9
        # . . . . . . . . . . . . . .
        #   . . . . . . . . . . . . .
        #     . . . . . . . . . . . .
        #       . . . . . . . . . . .
        for layer in range(self.nb_layers):
            if layer == 0: #
                prob_space[layer][0] = network_space[layer][0][1]
                prob_space[layer][1] = network_space[layer][1][0]

                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(4):
                    if sample > layer + 1: continue
                    local_prob = []
                    for rate in range(3):
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            # rate means how it comes from
                            continue
                        else:
                            # 1-rate: path
                            rate2index = {0:2, 1:1, 2:0}
                            local_prob.append(
                                prob_space[layer-1][sample+1-rate] * \
                                network_space[layer][sample][rate2index[rate]]
                            )
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate # rate 0 ↗, 1 →, 2 ↘
                    path_space[layer][sample] = path          # path 1 ↗，0 →， -1 ↘ sample == 3, -1, -2

        output_sample = np.argmax(prob_space[-1, :], axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self.nb_layers):
            actual_path[-i-1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

        return actual_path

    def viterbi_decode_based_constraint(self):
        # network_space           [12, 4, 3], 0-layer is output of stem2 0: ↗, 1: →, 2: ↘
        # network_arch_parameters [12, 4, 3], 0-layer is true 0-layer,   0: ↘, 1: →, 2: ↗, w.r.t. each node in fabric
        with torch.no_grad():
            network_space = torch.zeros_like(self.network_arch_parameters)
            # if layer <= 8 can only have 0, 1
            # if layer > 8 can only have 1, 2
            for layer in range(self.nb_layers):
                if layer == 0:
                    network_space[layer][0][1] = F.softmax(self.network_arch_parameters[layer][0][1], -1) * (1 / 3)
                    network_space[layer][1][0] = F.softmax(self.network_arch_parameters[layer][1][0], -1) * (1 / 3)
                elif layer == 1:
                    network_space[layer][0][1] = F.softmax(self.network_arch_parameters[layer][0][1], -1) * (1 / 3)
                    network_space[layer][1][:2] = F.softmax(self.network_arch_parameters[layer][1][:2], -1) * (2 / 3)
                    network_space[layer][2][0] = F.softmax(self.network_arch_parameters[layer][2][0], -1) * (1 / 3)
                elif layer == 2:
                    network_space[layer][0][1] = F.softmax(self.network_arch_parameters[layer][0][1], -1) * (1 / 3)
                    network_space[layer][1][:2] = F.softmax(self.network_arch_parameters[layer][1][:2], -1) * (2 / 3)
                    network_space[layer][2][:2] = F.softmax(self.network_arch_parameters[layer][2][:2], -1) * (2 / 3)
                    network_space[layer][3][0] = F.softmax(self.network_arch_parameters[layer][3][0], -1) * (1 / 3)
                else:
                    if layer <= 8:
                        network_space[layer][0][1] = F.softmax(self.network_arch_parameters[layer][0][1], -1) * (1 / 3)
                        network_space[layer][1][:2] = F.softmax(self.network_arch_parameters[layer][1][:2], -1) * (2 / 3)
                        network_space[layer][2][:2] = F.softmax(self.network_arch_parameters[layer][2][:2], -1) * (2 / 3)
                        network_space[layer][3][:2] = F.softmax(self.network_arch_parameters[layer][3][:2], -1) * (2 / 3)
                    else:
                        network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                        network_space[layer][1][1:] = F.softmax(self.network_arch_parameters[layer][1][1:], -1) * (2 / 3)
                        network_space[layer][2][1:] = F.softmax(self.network_arch_parameters[layer][2][1:], -1) * (2 / 3)
                        network_space[layer][3][1] = F.softmax(self.network_arch_parameters[layer][3][1], -1) * (1 / 3)

        #print('viterbi_phase:\n', network_space)
        self.logger.log('network_arch_params:\n'+str(network_space), mode='network_space', display=False)
        prob_space = np.zeros(network_space.shape[:2]) # [12, 4]
        path_space = np.zeros(network_space.shape[:2]).astype('int8') # [12, 4]

        # prob_space [layer, sample] from layer sample to next layer
        #   0 1 2 3 4 5 6 7 8 9 network_space from here
        # 0 1 2 3 4 5 6 7 8 9
        # . . . . . . . . . . . . . .
        #   . . . . . . . . . . . . .
        #     . . . . . . . . . . . .
        #       . . . . . . . . . . .
        for layer in range(self.nb_layers):
            if layer == 0:  #
                prob_space[layer][0] = network_space[layer][0][1]
                prob_space[layer][1] = network_space[layer][1][0]

                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(4):
                    if sample > layer + 1: continue
                    local_prob = []
                    for rate in range(3):
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            # rate means how it comes from
                            continue
                        else:
                            # 1-rate: path
                            rate2index = {0: 2, 1: 1, 2: 0}
                            local_prob.append(
                                prob_space[layer - 1][sample + 1 - rate] * \
                                network_space[layer][sample][rate2index[rate]]
                            )
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate  # rate 0 ↗, 1 →, 2 ↘
                    path_space[layer][sample] = path  # path 1 ↗，0 →， -1 ↘ sample == 3, -1, -2

        output_sample = np.argmax(prob_space[-1, :], axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self.nb_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

        return actual_path

    def cell_genotype_decode(self, cell):
        genotypes = []
        with torch.no_grad():
            total_nodes = cell.total_nodes
            edge2index = cell.edge2index
            weight = cell.cell_arch_parameters
            ops = cell.ops
            for i in range(2, total_nodes): # for each node in a cell, excluding the first two nodes.
                xlist = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    branch_index = edge2index[node_str] # edge_index
                    # pay attention, include None operation.
                    if ops[node_str] is None:
                        assert j == 0, 'None operation, wrong edge.'
                        #xlist.append((node_str, None)) # excluding None edge.
                        #continue
                    #else:
                    mixed_op_weight = weight[branch_index]
                    select_op_index = mixed_op_weight.argmax().item() # for each edge, then get the previous two
                    # max weight for each edge tuple(node_str, weight)
                    xlist.append((node_str, select_op_index))
                # get the previous two
                previous_two = sorted(xlist, key=lambda x: -weight[edge2index[node_str]][x[1]])[:2] # (node_str, select_op_index)
                genotypes.append(previous_two) # (node_str, select_op_index)
            return genotypes

    def network_cell_arch_decode(self):
        #print('\t=> Super Network decoding ... ... ')
        actual_path = self.viterbi_decode_based_constraint()
        #print('acutal_path', actual_path)
        cell_genotypes = []
        current_scale = 0
        for layer in range(self.nb_layers):
            next_scale = int(actual_path[layer])
            cell_index = get_cell_index(layer, current_scale, next_scale)
            genotypes = self.cell_genotype_decode(self.cells[cell_index])
            cell_genotypes.append((cell_index, genotypes))
            current_scale = next_scale
        assert len(cell_genotypes) == 12, 'invalid length of cell_genotype'
        '''
        #print('cell_genotypes')
        for layer in range(self.nb_layers):
            cell_index, cell_genotype = cell_genotypes[layer]
            print('layer {} cell_index {} architecture'.format(layer, cell_index))
            for node_str, select_op_index in cell_genotype:
                print(node_str, self.conv_candidates[select_op_index])
        print('\t=> Super Network decoding done')
        '''
        return actual_path, cell_genotypes

    '''
    def forward_validate_test(self, x):

        actual_path = self.viterbi_decode() # pay attention int(actual_path)

        size = x.size()[-2:]
        inter_features = []

        x = self.stem0(x)
        x = self.stem1(x)
        inter_features.append((-1, x))
        x = self.stem2(x)
        inter_features.append((0, x))

        current_scale = 0
        for layer in range(self.nb_layers):
            next_scale = int(actual_path[layer])
            cell_index = get_cell_index(layer, current_scale, next_scale)
            prev_prev_feature, prev_feature = get_prev_c(inter_features, next_scale)
            state = self.cells[cell_index].forward_validate_test(prev_prev_feature, prev_feature)
            current_scale = next_scale
            inter_features.pop(0)
            inter_features.append((next_scale, state))

        last_scale = int(actual_path[-1])
        assert last_scale == inter_features[-1][0], 'actual_path[-1] and inter_features[-1][0] is inconsistent'
        if last_scale == 0:
            return F.interpolate(self.aspp4(inter_features[-1][1]), size, mode='bilinear', align_corners=True)
        elif last_scale == 1:
            return F.interpolate(self.aspp8(inter_features[-1][1]), size, mode='bilinear', align_corners=True)
        elif last_scale == 2:
            return F.interpolate(self.aspp16(inter_features[-1][1]), size, mode='bilinear', align_corners=True)
        elif last_scale == 3:
            return F.interpolate(self.aspp32(inter_features[-1][1]), size, mode='bilinear', align_corners=True)
    '''
    def single_path_forward(self, x):
        # forward for network_level and cell_level
        # 1. generate hardwts for super network √
        # 2. _forward for each node √
        # 3. generate hardwts for each cell √
        # 4. forward for each cell √
        # 5. re-order cells, have been re-ordered √

        def _single_path_gdas_weightsum(layer, current_scale, next_scale, cell_index, index, hardwts, prev_prev_feature, prev_feature):

            # active path output, ignore the outputs of the other two (at most).
            state = self.cell_gdas_forward(cell_index, prev_prev_feature, prev_feature)
            # get inter_result list [], according to current_scale, next_scale
            if current_scale == next_scale: # same 1
                inter_result = [None, state, None]
            elif current_scale - 1 == next_scale: # up 2
                inter_result = [None, None, state]
            elif current_scale + 1 == next_scale: # down 0
                inter_result = [state, None, None]
            else: raise ValueError('invalid scale relation between current_scale {:} and next_scale {:}'.format(current_scale, next_scale))
            #print(_index)
            assert inter_result[index[layer][next_scale]] is not None, 'Error in _single_path_gdas_weightsum'
            return sum(state * hardwts[layer][next_scale][_i] if _i == index[layer][next_scale] else hardwts[layer][next_scale][_i] for _i, state in enumerate(inter_result))

        #hardwts, index = self.get_network_arch_hardwts()
        # todo obtain hardwts and index with constraint.
        hardwts, index = self.get_network_arch_hardwts_with_constraint()
        log, flag = detect_invalid_index(index, self.nb_layers)
        assert flag, log
        # to get aspp hardwts
        while True:
            # shape as []
            aspp_gumbels = -torch.empty_like(self.aspp_arch_parameters).exponential_().log()
            aspp_logits = (self.aspp_arch_parameters.log_softmax(dim=-1) + aspp_gumbels) / self.tau
            aspp_probs = F.softmax(aspp_logits, dim=-1)
            aspp_index = aspp_probs.max(-1, keepdim=True)[1]
            aspp_one_h = torch.zeros_like(aspp_logits).scatter_(-1, aspp_index, 1.0)
            aspp_hardwts = aspp_one_h - aspp_probs.detach() + aspp_probs
            if (torch.isinf(aspp_gumbels).any()) or (torch.isinf(aspp_probs).any()) or (torch.isnan(aspp_probs).any()):
                continue
            else: break

        # TODO:
        # 1. according to aspp_hardwts, obtain single path of the super network. from backward √
        # 2. forward for the single path,
        #       need consider: how to process prev_prev_cell output. todo work with auto_deeplab Github issue.
        #                      add super network constraint to shrink network-level search space.
        #

        # After obtaining network_arch_parameters and aspp_arch_parameters.
        # network_arch_parameters: hardwts
        # aspp_arch_parameters: aspp_hardwts
        # todo, in the forward phrase, should perform sample_single_path, not viterbi_decode
        # it should be consistent with 'index'
        single_path = self.sample_single_path(self.nb_layers, aspp_index, index) # record next_scale from output of stem2

        # used to debug, record the sampled single_path
        self.logger.log(str(single_path), mode='single_path', display=False)

        # forward according to single_path
        size = x.size()[-2:]
        inter_features = []

        x = self.stem0(x)
        x = self.stem1(x)
        inter_features.append((-1, x))
        x = self.stem2(x)
        inter_features.append((0, x))

        # TODO: prev_prev_scale is considered the same as the next_scale, which is inconsistent with the origin implementation of AutoDeeplab.
        current_scale = 0
        for layer in range(self.nb_layers): # single_path loop layer-index w.r.t. output of stem2
            next_scale = int(single_path[layer])
            prev_prev_feature, prev_feature = get_prev_c(inter_features, next_scale)
            cell_index = get_cell_index(layer, current_scale, next_scale)
            # pay attention:: index from layer-1 (w.r.t. output of stem2 is layer-0)
            #_index = index[layer][next_scale]
            #_weight = hardwts[layer][next_scale]
            # need prev_prev_feature, prev_features, weight, index, active
            #print('_single_path_foward layer{} scale{} cell_index{}'.format(layer+1, next_scale, cell_index))
            state = _single_path_gdas_weightsum(layer, current_scale, next_scale, cell_index, index, hardwts, prev_prev_feature, prev_feature)
            current_scale = next_scale
            inter_features.pop(0)
            inter_features.append((next_scale, state))

        last_scale = inter_features[-1][0]
        if last_scale == 0:
            aspp_result = self.aspp4(inter_features[-1][1])
        elif last_scale == 1:
            aspp_result = self.aspp8(inter_features[-1][1])
        elif last_scale == 2:
            aspp_result = self.aspp16(inter_features[-1][1])
        elif last_scale == 3:
            aspp_result = self.aspp32(inter_features[-1][1])
        else:
            raise ValueError('invalid last_scale value {}'.format(last_scale))

        aspp_result = F.interpolate(aspp_result, size, mode='bilinear', align_corners=True)
        return aspp_result

    def _gdas_weighted_sum(self, layer, scale, cell_index, weight, index, prev_prev_c, prev_c):
        # prev_c is a feature list, including prev_c_up, prev_c_same, prev_c_down
        inter_result = []
        if layer == 0:
            if scale == 0:
                new_state = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                inter_result = [None, new_state, None]
            elif scale == 1:
                # print(detect_inputs_shape(prev_prev_c, prev_c[0]))
                # print('cell_index', cell_index)
                new_state = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                inter_result = [new_state, None, None]
        elif layer == 1:
            if scale == 0:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                # print(detect_inputs_shape(prev_prev_c, prev_c[1]))
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                inter_result = [None, new_state_1, new_state_2]
            elif scale == 1:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                # new_state_3 = _forward(cell_index+2, prev_prev_c, prev_c[2])
                inter_result = [new_state_1, new_state_2, None]
            elif scale == 2:
                new_state = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                inter_result = [new_state, None, None]
        elif layer == 2:
            if scale == 0:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                inter_result = [None, new_state_1, new_state_2]
            elif scale == 1:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                new_state_3 = self.cell_gdas_forward(cell_index + 2, prev_prev_c, prev_c[2])
                inter_result = [new_state_1, new_state_2, new_state_3]
            elif scale == 2:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                inter_result = [new_state_1, new_state_2, None]
            elif scale == 3:
                new_state = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                inter_result = [new_state, None, None]
        else:
            if scale == 0:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                inter_result = [None, new_state_1, new_state_2]
            elif scale == 1:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                new_state_3 = self.cell_gdas_forward(cell_index + 2, prev_prev_c, prev_c[2])
                inter_result = [new_state_1, new_state_2, new_state_3]
            elif scale == 2:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                new_state_3 = self.cell_gdas_forward(cell_index + 2, prev_prev_c, prev_c[2])
                inter_result = [new_state_1, new_state_2, new_state_3]
            elif scale == 3:
                new_state_1 = self.cell_gdas_forward(cell_index, prev_prev_c, prev_c[0])
                new_state_2 = self.cell_gdas_forward(cell_index + 1, prev_prev_c, prev_c[1])
                inter_result = [new_state_1, new_state_2, None]
        '''       
        #rt = 0.
        assert len(inter_result) == 3, 'error in inter_result of network level gumbel'
        # weight, one_hot vector of each super network node
        for _index, state in enumerate(inter_result):
            if state is None: continue
            #print(state.shape, wt)
            if _index == index:
                print(state.shape, weight[_index])
                assert state is not None, 'error in network level gumbel'
                rt += state * weight[_index]
            else:
                rt += weight[_index]
        #return rt
        '''
        return sum(
            state * weight[_index] if _index == index else weight[_index] for _index, state in enumerate(inter_result))

    def forward(self, x):

        hardwts, index = self.get_network_arch_hardwts()

        size = x.size()[-2:]

        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stem2(x)

        scale0_features = [x]
        #print(x.shape)
        scale1_features = []
        scale2_features = []
        scale3_features = []

        count = 0

        for layer in range(self.nb_layers):
            scale_list = {}
            for scale in range(4): # scale = 0, 1, 2, 3
                if layer == 0 and (scale == 2 or scale == 3):
                    scale_list[scale] = None
                    continue
                if layer == 1 and (scale == 3):
                    continue
                print('_forward for layer{} scale{}'.format(layer, scale))

                # weight and index of super network node
                _weight = hardwts[layer][scale]
                _index = index[layer][scale]
                # prev_prev_ and prev_ for cell _forward
                prev_prev_feature, prev_feature = get_pfeatures(layer, scale, scale0_features, scale1_features,
                                                                scale2_features, scale3_features)
                #print(prev_feature[0].shape)
                # get gdas_output
                new_state = self._gdas_weighted_sum(
                    layer, scale, count, _weight, _index,
                    prev_prev_feature, prev_feature
                )
                scale_list[scale] = new_state

                # count_base
                if layer == 0 and (scale == 0 or scale == 1):
                    count += 1
                elif layer == 1:
                    if scale == 0: count += 2
                    elif scale == 1: count += 2
                    elif scale == 2: count += 1
                elif layer == 2:
                    if scale == 0: count += 2
                    elif scale == 1: count += 3
                    elif scale == 2: count += 2
                    elif scale == 3: count += 1
                else:
                    if scale == 0: count += 2
                    elif scale == 1 or scale == 2: count += 3
                    elif scale == 3: count += 2

            # append
            append_scale_list(scale_list, scale0_features, scale1_features, scale2_features, scale3_features)

        aspp_result_0= F.interpolate(self.aspp4(scale0_features[-1]), size, mode='bilinear', align_corners=True)
        aspp_result_1= F.interpolate(self.aspp8(scale1_features[-1]), size, mode='bilinear', align_corners=True)
        aspp_result_2= F.interpolate(self.aspp16(scale2_features[-1]), size, mode='bilinear', align_corners=True)
        aspp_result_3= F.interpolate(self.aspp32(scale3_features[-1]), size, mode='bilinear', align_corners=True)

        return aspp_result_0 + aspp_result_1 + aspp_result_2 + aspp_result_3

    def cell_gdas_forward(self, cell_index, prev_prev_c, prev_c):

        cell = self.cells[cell_index]
        # assert cell.cell_arch_parameters.shape == torch.Size(self.nb_edges, cell.n_choices)
        while True:
            # TODO: cell_arch_parameters shape as [nb_edges, n_choices]
            gumbels = -torch.empty_like(cell.cell_arch_parameters).exponential_().log()
            logits = (cell.cell_arch_parameters.log_softmax(dim=-1) + gumbels) / self.tau
            probs = F.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else:
                break
        # print(detect_inputs_shape(prev_prev_c, prev_c))
        return cell.forward_gdas(prev_prev_c, prev_c, hardwts, index)
    # get single path, according to network-level hardwts, starts from aspp_hardwts.
    def sample_single_path(self, nb_layers, aspp_index, network_index):
        # according hardwts and aspp_hardwts
        # refer viterbi_decode
        scale2index = {0: -1, 1: 0, 2: 1}
        single_path = np.zeros(12).astype('uint8')
        with torch.no_grad():
            last_scale = aspp_index
            single_path[-1] = last_scale
            for i in range(1, nb_layers):  # [1, 11]
                single_path[-1 - i] = single_path[-i] + scale2index[network_index[-i][single_path[-i]].item()]

        # network_index[-i][single_path[-i]]:: the choice from single_path[-1-i] to single_path[-i]
        # single_path[-1-i]:: scale of layer -i
        # single_path[-i]:: scale of layer -i+1
        # network_index 0, single_path[-i] - 1, network_index 1, single_path[-i], network_index 2, single_path[-i] + 1
        # single_path records next_scale, starts from layer-0 (output of stem2)

        # len()=12, record the next scale from 0-layer(output of stem2)
        return single_path

    def get_network_arch_hardwts(self):
        while True:
            # network_arch_parameters shape [12, 4, 3]
            # both requires_grad
            gumbels = -torch.empty_like(self.network_arch_parameters).exponential_().log()
            logits = torch.zeros_like(self.network_arch_parameters)
            probs = torch.zeros_like(self.network_arch_parameters)
            #index = torch.zeros(self.nb_layers, 4, 1)# initialized -1
            #with torch.no_grad(): # to avoid the indice operation over probs cause in-place modification error.
            #print(self.network_arch_parameters.requires_grad)
            for layer in range(self.nb_layers):
                if layer == 0: # scale 0 and scale 1
                    scale = 0
                    logits[layer][scale][1] = \
                    (self.network_arch_parameters[layer][scale][1].log_softmax(dim=-1) * (1/3) + gumbels[layer][scale][1]) / self.tau
                    probs[layer][scale][1] = F.softmax(logits.clone()[layer][scale][1], dim=-1) * (1/3)
                    #probs[layer][scale][1] = logits[layer][scale][1].softmax(dim=-1) * (1 / 3)
                    # probs over only one value is one, the other two are zeros
                    # index[layer][scale][0] = probs[layer][scale][1].max(-1, keepdim=True)[1]
                    scale = 1
                    logits[layer][scale][0] = \
                    (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3) + gumbels[layer][scale][0]) / self.tau
                    probs[layer][scale][0] = F.softmax(logits.clone()[layer][scale][0], dim=-1) * (1/3)
                    #probs[layer][scale][0] = logits[layer][scale][0].softmax(dim=-1) * (1/3)
                    # probs over only one value is one, the other two are zeros
                    # index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                elif layer == 1: # scale 0, scale 1, and scale 2
                    scale = 0
                    logits[layer][scale][1:] = \
                    (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][1:]) / self.tau
                    probs[layer][scale][1:] = F.softmax(logits.clone()[layer][scale][1:], dim=-1) * (2/3)
                    #probs[layer][scale][1:] = logits[layer][scale][1:].softmax(dim=-1) * (2/3)
                    # index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                    scale = 1
                    logits[layer][scale][:2] = \
                    (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][:2]) / self.tau
                    probs[layer][scale][:2] = F.softmax(logits.clone()[layer][scale][:2], dim=-1) * (2/3)
                    #probs[layer][scale][:2] = logits[layer][scale][:2].softmax(dim=-1) * (2/3)
                    # index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]
                    scale = 2
                    logits[layer][scale][0] = \
                    (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3) + gumbels[layer][scale][0]) / self.tau
                    probs[layer][scale][0] = F.softmax(logits.clone()[layer][scale][0], dim=-1) * (1/3)
                    #probs[layer][scale][0] = logits[layer][scale][0].softmax(dim=-1) * (1/3)
                    # index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                elif layer == 2: # scale 0, scale 1, scale 2, and scale 3
                    scale = 0
                    logits[layer][scale][1:] = \
                    (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][1:]) / self.tau
                    probs[layer][scale][1:] = F.softmax(logits.clone()[layer][scale][1:], dim=-1) * (2/3)
                    #probs[layer][scale][1:] = logits.data[layer][scale][1:].softmax(dim=-1) * (2/3)
                    # index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                    scale = 1
                    logits[layer][scale] = \
                    (self.network_arch_parameters[layer][scale].log_softmax(dim=-1) + gumbels[layer][scale]) / self.tau
                    probs[layer][scale] = F.softmax(logits.clone()[layer][scale], dim=-1)
                    #probs[layer][scale] = logits[layer][scale].softmax(dim=-1)
                    # index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                    scale = 2
                    logits[layer][scale][:2] = \
                    (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][:2]) / self.tau
                    probs[layer][scale][:2] = F.softmax(logits.clone()[layer][scale][:2], dim=-1) * (2/3)
                    #probs[layer][scale][:2] = logits[layer][scale][:2].softmax(dim=-1) * (2/3)
                    # index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]
                    scale = 3
                    logits[layer][scale][0] = \
                    (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3)+gumbels[layer][scale][0]) / self.tau
                    probs[layer][scale][0] = F.softmax(logits.clone()[layer][scale][0], dim=-1) * (1/3)
                    #probs[layer][scale][0] = logits[layer][scale][0].softmax(dim=-1) * (1/3)
                    # index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                else: # 0, 1, 2, 3
                    scale = 0
                    logits[layer][scale][1:] = \
                    (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][1:]) / self.tau
                    probs[layer][scale][1:] = F.softmax(logits.clone()[layer][scale][1:], dim=-1) * (2/3)
                    #probs[layer][scale][1:] = logits[layer][scale][1:].softmax(dim=-1) * (2/3)
                    # index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                    scale = 1
                    logits[layer][scale] = \
                    (self.network_arch_parameters[layer][scale].log_softmax(dim=-1)+gumbels[layer][scale]) / self.tau
                    probs[layer][scale] = F.softmax(logits.clone()[layer][scale], dim=-1,)
                    #probs[layer][scale] = logits[layer][scale].softmax(dim=-1)
                    # index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                    scale = 2
                    logits[layer][scale] = \
                    (self.network_arch_parameters[layer][scale].log_softmax(dim=-1)+gumbels[layer][scale]) / self.tau
                    probs[layer][scale] = F.softmax(logits.clone()[layer][scale], dim=-1)
                    #probs[layer][scale] = logits[layer][scale].softmax(dim=-1)
                    # index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                    scale = 3
                    logits[layer][scale][:2] = \
                    (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][:2]) / self.tau
                    probs[layer][scale][:2] = F.softmax(logits.clone()[layer][scale][:2], dim=-1) * (2/3)
                    #probs[layer][scale][:2] = logits[layer][scale][:2].softmax(dim=-1) * (2/3)
                    # index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]

            #print('probs', probs.requires_grad)
            #print('logits', logits.requires_grad)
            # prob of invalid choice is zero, index can always not select invalid choices.
            index = probs.max(-1, keepdim=True)[1] # [12, 4, 1]
            # according to index, one_hot
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0) # shape as [12, 4, 3]
            hardwts = one_h - probs.detach() + probs
            #print('hardwts', hardwts.requires_grad)
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else: break

        return hardwts, index
    def get_network_arch_hardwts_with_constraint(self):
        # the first nine layers only perform same or down operation. each node only active 0 or 1
        # the last three layers only perform same or up operation.   each node only active 1 or 2
        # means: scale 0 only have 'same operation' in the first nine layers
        #        scale 3 only have 'same operation' in the last three layers
        #
        while True:
            # 0-layer from true 0-layer
            gumbels = -torch.empty_like(self.network_arch_parameters).exponential_().log()
            logits = torch.zeros_like(self.network_arch_parameters)
            probs = torch.zeros_like(self.network_arch_parameters)
            for layer in range(self.nb_layers):
                if layer == 0:
                    logits[layer][0][1] = \
                        (self.network_arch_parameters[layer][0][1].log_softmax(dim=-1) * (1 / 3)  + gumbels[layer][0][1]) / self.tau
                    probs[layer][0][1] = F.softmax(logits.clone()[layer][0][1], dim=-1) * (1 / 3)
                    logits[layer][1][0] = \
                        (self.network_arch_parameters[layer][1][0].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][1][0]) / self.tau
                    probs[layer][1][0] = F.softmax(logits.clone()[layer][1][0], dim=-1) * (1 / 3)
                elif layer == 1:
                    logits[layer][0][1] = \
                        (self.network_arch_parameters[layer][0][1].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][0][1]) / self.tau
                    probs[layer][0][1] = F.softmax(logits.clone()[layer][0][1], dim=-1) * (1 / 3)
                    logits[layer][1][:2] = \
                        (self.network_arch_parameters[layer][1][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][1][:2]) / self.tau
                    probs[layer][1][:2] = F.softmax(logits.clone()[layer][1][:2], dim=-1) * (2 / 3)
                    logits[layer][2][0] = \
                        (self.network_arch_parameters[layer][2][0].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][2][0]) / self.tau
                    probs[layer][2][0] = F.softmax(logits.clone()[layer][2][0], dim=-1) * (1 / 3)
                elif layer == 2:
                    logits[layer][0][1] = \
                        (self.network_arch_parameters[layer][0][1].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][0][1]) / self.tau
                    probs[layer][0][1] = F.softmax(logits.clone()[layer][0][1], dim=-1) * (1 / 3)
                    logits[layer][1][:2] = \
                        (self.network_arch_parameters[layer][1][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][1][:2]) / self.tau
                    probs[layer][1][:2] = F.softmax(logits.clone()[layer][1][:2], dim=-1) * (2 / 3)
                    logits[layer][2][:2] = \
                        (self.network_arch_parameters[layer][2][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][2][:2]) / self.tau
                    probs[layer][2][:2] = F.softmax(logits.clone()[layer][2][:2], dim=-1) * (2 / 3)
                    logits[layer][3][0] = \
                        (self.network_arch_parameters[layer][3][0].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][3][0]) / self.tau
                    probs[layer][3][0] = F.softmax(logits.clone()[layer][3][0], dim=-1) * (1 /3)
                else:
                    if layer <= 8:
                        logits[layer][0][1] = \
                            (self.network_arch_parameters[layer][0][1].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][0][1]) / self.tau
                        probs[layer][0][1] = F.softmax(logits.clone()[layer][0][1], dim=-1) * (1 / 3)
                        logits[layer][1][:2] = \
                            (self.network_arch_parameters[layer][1][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][1][:2]) / self.tau
                        probs[layer][1][:2] = F.softmax(logits.clone()[layer][1][:2], dim=-1) * (2 / 3)
                        logits[layer][2][:2] = \
                            (self.network_arch_parameters[layer][2][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][2][:2]) / self.tau
                        probs[layer][2][:2] = F.softmax(logits.clone()[layer][2][:2], dim=-1) * (2 / 3)
                        logits[layer][3][:2] = \
                            (self.network_arch_parameters[layer][3][:2].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][3][:2]) / self.tau
                        probs[layer][3][:2] = F.softmax(logits.clone()[layer][3][:2], dim=-1) * (2 / 3)
                    else:
                        logits[layer][0][1:] = \
                            (self.network_arch_parameters[layer][0][1:].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][0][1:]) / self.tau
                        probs[layer][0][1:] = F.softmax(logits.clone()[layer][0][1:], dim=-1) * (2 / 3)
                        logits[layer][1][1:] = \
                            (self.network_arch_parameters[layer][1][1:].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][1][1:]) / self.tau
                        probs[layer][1][1:] = F.softmax(logits.clone()[layer][1][1:], dim=-1) * (2 / 3)
                        logits[layer][2][1:] = \
                            (self.network_arch_parameters[layer][2][1:].log_softmax(dim=-1) * (2 / 3) + gumbels[layer][2][1:]) / self.tau
                        probs[layer][2][1:] = F.softmax(logits.clone()[layer][2][1:], dim=-1) * (2 / 3)
                        logits[layer][3][1] = \
                            (self.network_arch_parameters[layer][3][1].log_softmax(dim=-1) * (1 / 3) + gumbels[layer][3][1]) / self.tau
                        probs[layer][3][1] = F.softmax(logits.clone()[layer][3][1], dim=-1) * (1 /3)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else: break
        return hardwts, index