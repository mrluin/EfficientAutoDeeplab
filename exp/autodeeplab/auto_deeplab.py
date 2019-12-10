# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.my_modules import MyNetwork
from collections import OrderedDict
from models.gumbel_cells import proxyless, autodeeplab, my_search_space
from modules.operations import ASPP
from exp.autodeeplab.cells import NormalCell

class AutoDeeplab(MyNetwork):
    def __init__(self,
                 filter_multiplier, block_multiplier, steps, nb_classes, nb_layers,
                 search_space, logger, affine=False):
        super(AutoDeeplab, self).__init__()
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
        else:
            raise ValueError('search space {:}, is not supported!'.format(search_space))
        self.logger = logger
        self.affine = affine
        self.cells = nn.ModuleList()

        # two-layer stem, output channels with 16 and 32
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

        prev_prev_c = 16
        prev_c = 32
        for i in range(self.nb_layers):
            if i == 0:
                cell1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                   pps=prev_prev_c, ps=prev_c, affine=self.affine)
                cell2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                   pps=prev_prev_c, ps=prev_c, affine=self.affine)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=prev_c, ps=0, affine=self.affine)
                cell1_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=prev_c, ps=1, affine=self.affine)
                cell2_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=prev_c, ps=0, affine=self.affine)
                cell2_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=prev_c, ps=1, affine=self.affine)
                cell3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                   pps=prev_c, ps=1, affine=self.affine)
                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell3]
            elif i == 2:
                cell1_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=0, affine=self.affine)
                cell1_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=1, affine=self.affine)
                cell2_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=0, affine=self.affine)
                cell2_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=1, affine=self.affine)
                cell2_3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=2, affine=self.affine)
                cell3_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=1, ps=1, affine=self.affine)
                cell3_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=1, ps=2, affine=self.affine)
                cell4 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 3, self.search_space,
                                   pps=1, ps=2, affine=self.affine)
                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell4]
            elif i == 3:
                cell1_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=0, affine=self.affine)
                cell1_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=1, affine=self.affine)
                cell2_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=0, affine=self.affine)
                cell2_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=1, affine=self.affine)
                cell2_3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=2, affine=self.affine)
                cell3_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=1, affine=self.affine)
                cell3_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=2, affine=self.affine)
                cell3_3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=3, affine=self.affine)
                cell4_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 3, self.search_space,
                                     pps=2, ps=2, affine=self.affine)
                cell4_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 3, self.search_space,
                                     pps=2, ps=3, affine=self.affine)
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
                cell1_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=0, affine=self.affine)
                cell1_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 0, self.search_space,
                                     pps=0, ps=1, affine=self.affine)
                cell2_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=0, affine=self.affine)
                cell2_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=1, affine=self.affine)
                cell2_3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 1, self.search_space,
                                     pps=1, ps=2, affine=self.affine)
                cell3_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=1, affine=self.affine)
                cell3_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=2, affine=self.affine)
                cell3_3 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 2, self.search_space,
                                     pps=2, ps=3, affine=self.affine)
                cell4_1 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 3, self.search_space,
                                     pps=3, ps=2, affine=self.affine)
                cell4_2 = NormalCell(i, self.filter_multiplier, self.block_multiplier, self.steps, 3, self.search_space,
                                     pps=3, ps=3, affine=self.affine)
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

        scale4_outc = int(self.filter_multiplier * self.block_multiplier * 4 / 4)
        scale8_outc = int(self.filter_multiplier * self.block_multiplier * 8 / 4)
        scale16_outc = int(self.filter_multiplier * self.block_multiplier * 16 / 4)
        scale32_outc = int(self.filter_multiplier * self.block_multiplier * 32 / 4)

        # dilation as 96/scale, change into 16 8 4 2
        self.aspp4 = ASPP(scale4_outc, self.nb_classes, 16, affine=affine)
        self.aspp8 = ASPP(scale8_outc, self.nb_classes, 8, affine=affine)
        self.aspp16 = ASPP(scale16_outc, self.nb_classes, 4, affine=affine)
        self.aspp32 = ASPP(scale32_outc, self.nb_classes, 2, affine=affine)

        self.network_arch_parameters = nn.Parameter(1e-3 * torch.randn(self.nb_layers, 4, 3))

        # for cells
        nb_edges = len([j for i in range(self.steps) for j in range(i+2)])
        nb_choices = len(self.conv_candidates)
        self.cell_arch_parameters = nn.Parameter(1e-3 * torch.randn(nb_edges, nb_choices))
        self.total_nodes = 2 + self.steps
        self.edge2index = self.cells[0].edge2index
        self.ops = self.cells[0].ops


    def get_network_arch_parameters(self):
        for name, param, in self.named_parameters():
            if 'network_arch_parameters' in name:
                yield param
    def get_cell_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'cell_arch_parameters' in name:
                yield param
    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'network_arch_parameters' not in name and 'cell_arch_parameters' not in name:
                yield param
    def arch_parameters(self):
        for name, param in self.named_parameters():
            if 'network_arch_parameters' in name or 'cell_arch_parameters' in name:
                yield param

    def viterbi_decode(self):
        with torch.no_grad():
            network_space = torch.zeros_like(self.network_arch_parameters)
            for layer in range(self.nb_layers):
                if layer == 0:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                elif layer == 1:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                    network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                elif layer == 2:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                    network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                    network_space[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
                else:
                    network_space[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                    network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                    network_space[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
                    network_space[layer][3][:2] = F.softmax(self.network_arch_parameters[layer][3][:2], -1) * (2 / 3)

        print('viterbi_phase:\n', network_space)
        prob_space = torch.zeros(network_space.shape[:2])
        path_space = torch.zeros(network_space.shape[:2]).astype('int8')
        for layer in range(self.nb_layers):
            if layer == 0:
                prob_space[layer][0] = network_space[layer][0][1]
                prob_space[layer][1] = network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(4):
                    if sample > layer + 1: continue
                    local_prob = []
                    for rate in range(3):
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0): continue
                        else:
                            local_prob.append(prob_space[layer-1][sample+1-rate] *
                                              network_space[layer][sample+1-rate][rate])

                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path
        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(self.nb_layers).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self.nb_layers):
            actual_path[-i-1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

    def cell_genotype_decode(self):
        # for shared cell structure
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
                    select_op_index = mixed_op_weight.argmax().item(0) # select operation with the highest prob for each edge.
                    xlist.append((node_str, select_op_index)) # the highest prob operation for all incoming edges.
                previous_two = sorted(xlist, key=lambda x: -weight[edge2index[node_str]][x[1]])[:2] # select the highest two for each node.
                genotypes.append(previous_two)
            return genotypes
    def network_cell_arch_decode(self):
        actual_path = self.viterbi_decode()
        #cell_genotypes = []
        #current_scale = 0
        cell_genotype = self.cell_genotype_decode()
        '''
        # for un-shared cell structure
        for layer in range(self.nb_layers):
            next_scale = int(actual_path[layer])
            cell_index = get_cell_index(layer, current_scale, next_scale)
            genotypes = self.cell_genotype_decode()
            cell_genotypes.append((cell_index, genotypes))
            current_scale = next_scale
            '''
        return actual_path, cell_genotype

    def forward(self, x):
        size = x.size()[2:]
        scale4 = []
        scale8 = []
        scale16 = []
        scale32 = []

        x = self.stem0(x)
        out_stem0 = x
        x = self.stem1(x)
        out_stem1 = x

        n_network_arch_param = torch.randn(self.nb_layers, 4, 3).to(x.device)
        for layer in range(self.nb_layers):
            if layer == 0:
                n_network_arch_param[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
            elif layer == 1:
                n_network_arch_param[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                n_network_arch_param[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
            elif layer == 2:
                n_network_arch_param[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                n_network_arch_param[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                n_network_arch_param[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
            else:
                n_network_arch_param[layer][0][1:] = F.softmax(self.network_arch_parameters[layer][0][1:], -1) * (2 / 3)
                n_network_arch_param[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                n_network_arch_param[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
                n_network_arch_param[layer][3][:2] = F.softmax(self.network_arch_parameters[layer][3][:2], -1) * (2 / 3)

        count = 0
        for layer in range(self.nb_layers):
            if layer == 0:
                scale4_new = self.cells[count](out_stem0, out_stem1, self.cell_arch_parameters)
                scale4_new = scale4_new * n_network_arch_param[layer][0][1]
                count += 1
                scale8_new = self.cells[count](out_stem0, out_stem1, self.cell_arch_parameters)
                scale8_new = scale8_new * n_network_arch_param[layer][0][2]
                count += 1
                scale4.append(scale4_new)
                scale8.append(scale8_new)
            elif layer == 1:
                scale4_new_1 = self.cells[count](out_stem1, scale4[-1], self.cell_arch_parameters)
                count += 1
                scale4_new_2 = self.cells[count](out_stem1, scale8[-1], self.cell_arch_parameters)
                count += 1
                scale4_new = scale4_new_1 * n_network_arch_param[layer][0][1] + scale4_new_2 * n_network_arch_param[layer][1][0]
                scale8_new_1 = self.cells[count](out_stem1, scale4[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_2 = self.cells[count](out_stem1, scale8[-1], self.cell_arch_parameters)
                count += 1
                scale8_new = scale8_new_1 * n_network_arch_param[layer][0][2] + scale8_new_2 * n_network_arch_param[layer][1][1]
                scale16_new = self.cells[count](out_stem1, scale8[-1], self.cell_arch_parameters)
                count += 1
                scale16_new = scale16_new * n_network_arch_param[layer][1][2]
                scale4.append(scale4_new)
                scale8.append(scale8_new)
                scale16.append(scale16_new)
            elif layer == 2:
                scale4_new_1 = self.cells[count](scale4[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale4_new_2 = self.cells[count](scale4[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale4_new = scale4_new_1 * n_network_arch_param[layer][0][1] + scale4_new_2 * n_network_arch_param[layer][1][0]
                scale8_new_1 = self.cells[count](scale8[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_2 = self.cells[count](scale8[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_3 = self.cells[count](scale8[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale8_new = scale8_new_1 * n_network_arch_param[layer][0][2] + scale8_new_2 * n_network_arch_param[layer][1][1] + scale8_new_3 * n_network_arch_param[layer][2][0]
                scale16_new_1 = self.cells[count](scale8[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale16_new_2 = self.cells[count](scale8[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale16_new = scale16_new_1 * n_network_arch_param[layer][1][2] + scale16_new_2 * n_network_arch_param[layer][2][1]
                scale32_new = self.cells[count](scale8[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale32_new = scale32_new * n_network_arch_param[layer][2][2]
                scale4.append(scale4_new)
                scale8.append(scale8_new)
                scale16.append(scale16_new)
                scale32.append(scale32_new)
            elif layer == 3:
                scale4_new_1 = self.cells[count](scale4[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale4_new_2 = self.cells[count](scale4[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale4_new = scale4_new_1 * n_network_arch_param[layer][0][1] + scale4_new_2 * n_network_arch_param[layer][1][0]
                scale8_new_1 = self.cells[count](scale8[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_2 = self.cells[count](scale8[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_3 = self.cells[count](scale8[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale8_new = scale8_new_1 * n_network_arch_param[layer][0][2] + scale8_new_2 * n_network_arch_param[layer][1][1] + scale8_new_3 * n_network_arch_param[layer][2][0]
                scale16_new_1 = self.cells[count](scale16[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale16_new_2 = self.cells[count](scale16[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale16_new_3 = self.cells[count](scale16[-2], scale32[-1], self.cell_arch_parameters)
                count += 1
                scale16_new = scale16_new_1 * n_network_arch_param[layer][1][2] + scale16_new_2 * n_network_arch_param[layer][2][1] + scale16_new_3 * n_network_arch_param[layer][3][0]
                scale32_new_1 = self.cells[count](scale16[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale32_new_2 = self.cells[count](scale16[-2], scale32[-1], self.cell_arch_parameters)
                count += 1
                scale32_new = scale32_new_1 * n_network_arch_param[layer][2][2] + scale32_new_2 * n_network_arch_param[layer][3][1]
                scale4.append(scale4_new)
                scale8.append(scale8_new)
                scale16.append(scale16_new)
                scale32.append(scale32_new)
            else:
                scale4_new_1 = self.cells[count](scale4[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale4_new_2 = self.cells[count](scale4[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale4_new = scale4_new_1 * n_network_arch_param[layer][0][1] + scale4_new_2 * n_network_arch_param[layer][1][0]
                scale8_new_1 = self.cells[count](scale8[-2], scale4[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_2 = self.cells[count](scale8[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale8_new_3 = self.cells[count](scale8[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale8_new = scale8_new_1 * n_network_arch_param[layer][0][2] + scale8_new_2 * n_network_arch_param[layer][1][1] + scale8_new_3 * n_network_arch_param[layer][2][0]
                scale16_new_1 = self.cells[count](scale16[-2], scale8[-1], self.cell_arch_parameters)
                count += 1
                scale16_new_2 = self.cells[count](scale16[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale16_new_3 = self.cells[count](scale16[-2], scale32[-1], self.cell_arch_parameters)
                count += 1
                scale16_new = scale16_new_1 * n_network_arch_param[layer][1][2] + scale16_new_2 * n_network_arch_param[layer][2][1] + scale16_new_3 * n_network_arch_param[layer][3][0]
                scale32_new_1 = self.cells[count](scale32[-2], scale16[-1], self.cell_arch_parameters)
                count += 1
                scale32_new_2 = self.cells[count](scale32[-2], scale32[-1], self.cell_arch_parameters)
                count += 1
                scale32_new = scale32_new_1 * n_network_arch_param[layer][2][2] + scale32_new_2 * n_network_arch_param[layer][3][1]
                scale4.append(scale4_new)
                scale8.append(scale8_new)
                scale16.append(scale16_new)
                scale32.append(scale32_new)

        aspp4 = F.interpolate(self.aspp4(scale4[-1]), size=size, mode='bilinear', align_corners=True)
        aspp8 = F.interpolate(self.aspp8(scale8[-1]), size=size, mode='bilinear', align_corners=True)
        aspp16 = F.interpolate(self.aspp16(scale16[-1]), size=size, mode='bilinear', align_corners=True)
        aspp32 = F.interpolate(self.aspp32(scale32[-1]), size=size, mode='bilinear', align_corners=True)

        return aspp4 + aspp8 + aspp16 + aspp32

