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

from utils.common import get_pfeatures, detect_inputs_shape, append_scale_list

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
        self.network_arch_parameters = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))
        self.aspp_arch_parameters = nn.Parameter(torch.Tensor(4))
        self.tau = 10

        # todo
        self.nb_edges =
        self.edge2index =


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

    def viterbi_decode(self):

        network_space = torch.zeros_like(self.network_arch_parameters)
        for layer in range(self.nb_layers):
            if layer == 0:
                network_space[layer][0][1] = F.softmax(self.network_arch_parameters.data[layer][0][1], -1) * (1/3)
                network_space[layer][1][0] = F.softmax(self.network_arch_parameters.data[layer][1][0], -1) * (1/3)
            elif layer == 1:
                network_space[layer][0][1:] = F.softmax(self.network_arch_parameters.data[layer][0][1:], -1) * (2/3)
                network_space[layer][1][:2] = F.softmax(self.network_arch_parameters.data[layer][1][:2], -1) * (2/3)
                network_space[layer][2][0] = F.softmax(self.network_arch_parameters.data[layer][2][0], -1) * (1/3)
            elif layer == 2:
                network_space[layer][0][1:] = F.softmax(self.network_arch_parameters.data[layer][0][1:], -1) * (2/3)
                network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                network_space[layer][2][:2] = F.softmax(self.network_arch_parameters[layer][2][:2], -1) * (2/3)
                network_space[layer][3][0] = F.softmax(self.network_arch_parameters[layer][3][0], -1) * (1/3)
            else:
                network_space[layer][0][1:] = F.softmax(self.network_arch_parameters.data[layer][0][1:], -1) * (2 / 3)
                network_space[layer][1] = F.softmax(self.network_arch_parameters[layer][1], -1)
                network_space[layer][2] = F.softmax(self.network_arch_parameters[layer][2], -1)
                network_space[layer][3][:2] = F.softmax(self.network_arch_parameters[layer][3][:2], -1) * (2 / 3)


        prob_space = np.zeros(network_space.shape[:2]) # [12, 4]
        path_space = np.zeros(network_space.shape[:2]).astype('int8') # [12, 4]

        # prob_space [layer, sample] from layer sample to next layer
        #   0 1 2 3 4 5 6 7 8 9network_space from here
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
                            rate2index = {0:2, 1:1, 2: 0}
                            local_prob.append(
                                prob_space[layer-1][sample+1-rate] * \
                                network_space[layer][sample][rate2index[rate]]
                            )
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path
        output_sample = np.argmax(prob_space[-1, :], axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self.nb_layers):
            actual_path[-i-1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

        return actual_path
    '''
    def genotype_decode(self):
        genotypes = []
        for i in range(2, total_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    branch_index = edge2index[node_str]
                    weight = cell_arch_parameters[branch_index]
                    op_name = op_name[weight.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist)) # operation for each node
        return Structure(genotypes)
    '''
    def cell_arch_decode(self):
        raise NotImplementedError

    def forward(self, x):
        # forward for network_level and cell_level

        # 1. generate hardwts for super network
        # 2. _forward for each node
        # 3. generate hardwts for each cell
        # 4. forward for each cell
        # 5. re-order cells, have been re-ordered

        def _gdas_weighted_sum(layer, scale, cell_index, weight, index, prev_prev_c, prev_c):
            # prev_c is a feature list, including prev_c_up, prev_c_same, prev_c_down
            inter_result = []
            if layer == 0:
                if scale == 0:
                    new_state = _forward(cell_index, prev_prev_c, prev_c[0])
                    inter_result = [None, new_state, None]
                elif scale == 1:
                    #print(detect_inputs_shape(prev_prev_c, prev_c[0]))
                    #print('cell_index', cell_index)
                    new_state = _forward(cell_index, prev_prev_c, prev_c[0])
                    inter_result = [new_state, None, None]
            elif layer == 1:
                if scale == 0:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    #print(detect_inputs_shape(prev_prev_c, prev_c[1]))
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    inter_result = [None, new_state_1, new_state_2]
                elif scale == 1:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    #new_state_3 = _forward(cell_index+2, prev_prev_c, prev_c[2])
                    inter_result = [new_state_1, new_state_2, None]
                elif scale == 2:
                    new_state = _forward(cell_index, prev_prev_c, prev_c[0])
                    inter_result = [new_state, None, None]
            elif layer == 2:
                if scale == 0:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    inter_result = [None, new_state_1, new_state_2]
                elif scale == 1:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    new_state_3 = _forward(cell_index+2, prev_prev_c, prev_c[2])
                    inter_result = [new_state_1, new_state_2, new_state_3]
                elif scale == 2:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    inter_result = [new_state_1, new_state_2, None]
                elif scale == 3:
                    new_state = _forward(cell_index, prev_prev_c, prev_c[0])
                    inter_result = [new_state, None, None]
            else:
                if scale == 0:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    inter_result = [None, new_state_1, new_state_2]
                elif scale == 1:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    new_state_3 = _forward(cell_index+2, prev_prev_c, prev_c[2])
                    inter_result = [new_state_1, new_state_2, new_state_3]
                elif scale == 2:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
                    new_state_3 = _forward(cell_index+2, prev_prev_c, prev_c[2])
                    inter_result = [new_state_1, new_state_2, new_state_3]
                elif scale == 3:
                    new_state_1 = _forward(cell_index, prev_prev_c, prev_c[0])
                    new_state_2 = _forward(cell_index+1, prev_prev_c, prev_c[1])
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
            return sum(state * weight[_index] if _index==index else weight[_index] for _index, state in enumerate(inter_result))

        def _forward(cell_index, prev_prev_c, prev_c):

            cell = self.cells[cell_index]
            #assert cell.cell_arch_parameters.shape == torch.Size(self.nb_edges, cell.n_choices)
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
                else: break

            #print(detect_inputs_shape(prev_prev_c, prev_c))
            return cell.forward_gdas(prev_prev_c, prev_c, hardwts, index)

        # to get network-level hardwts
        while True:
            # network_arch_parameters shape [12, 4, 3]
            # both requires_grad
            gumbels = -torch.empty_like(self.network_arch_parameters).exponential_().log()
            logits = torch.zeros_like(self.network_arch_parameters)
            probs = torch.zeros_like(self.network_arch_parameters)

            #index = torch.zeros(self.nb_layers, 4, 1)# initialized -1
            with torch.no_grad(): # to avoid the indice operation over probs cause in-place modification error.
                for layer in range(self.nb_layers):
                    if layer == 0: # scale 0 and scale 1
                        scale = 0
                        logits[layer][scale][1] = \
                            (self.network_arch_parameters[layer][scale][1].log_softmax(dim=-1) * (1/3) + gumbels[layer][scale][1]) / self.tau
                        probs[layer][scale][1] = F.softmax(logits[layer][scale][1], dim=-1) * (1/3)
                        # probs over only one value is one, the other two are zeros
                        #index[layer][scale][0] = probs[layer][scale][1].max(-1, keepdim=True)[1]
                        scale = 1
                        logits[layer][scale][0] = \
                            (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3) + gumbels[layer][scale][0]) / self.tau
                        probs[layer][scale][0] = F.softmax(logits[layer][scale][0], dim=-1) * (1/3)
                        # probs over only one value is one, the other two are zeros
                        #index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                    elif layer == 1: # scale 0, scale 1, and scale 2
                        scale = 0
                        logits[layer][scale][1:] = \
                            (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][1:]) / self.tau
                        probs[layer][scale][1:] = F.softmax(logits[layer][scale][1:], dim=-1) * (2/3)
                        #index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                        scale = 1
                        logits[layer][scale][:2] = \
                            (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][:2]) / self.tau
                        probs[layer][scale][:2] = F.softmax(logits[layer][scale][:2], dim=-1) * (2/3)
                        #index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]
                        scale = 2
                        logits[layer][scale][0] = \
                            (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3) + gumbels[layer][scale][0]) / self.tau
                        probs[layer][scale][0] = F.softmax(logits[layer][scale][0], dim=-1) * (1/3)
                        #index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                    elif layer == 2: # scale 0, scale 1, scale 2, and scale 3
                        scale = 0
                        logits[layer][scale][1:] = \
                            (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3) + gumbels[layer][scale][1:]) / self.tau
                        probs[layer][scale][1:] = F.softmax(logits[layer][scale][1:], dim=-1) * (2/3)
                        #index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                        scale = 1
                        logits[layer][scale] = \
                            (self.network_arch_parameters[layer][scale].log_softmax(dim=-1) + gumbels[layer][scale]) / self.tau
                        probs[layer][scale] = F.softmax(logits[layer][scale], dim=-1)
                        #index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                        scale = 2
                        logits[layer][scale][:2] = \
                            (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][:2]) / self.tau
                        probs[layer][scale][:2] = F.softmax(logits[layer][scale][:2], dim=-1) * (2/3)
                        #index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]
                        scale = 3
                        logits[layer][scale][0] = \
                            (self.network_arch_parameters[layer][scale][0].log_softmax(dim=-1)*(1/3)+gumbels[layer][scale][0]) / self.tau
                        probs[layer][scale][0] = F.softmax(logits[layer][scale][0], dim=-1) * (1/3)
                        #index[layer][scale][0] = probs[layer][scale][0].max(-1, keepdim=True)[1]
                    else: # 0, 1, 2, 3
                        scale = 0
                        logits[layer][scale][1:] = \
                            (self.network_arch_parameters[layer][scale][1:].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][1:]) / self.tau
                        probs[layer][scale][1:] = F.softmax(logits[layer][scale][1:], dim=-1) * (2/3)
                        #index[layer][scale][1:] = probs[layer][scale][1:].max(-1, keepdim=True)[1]
                        scale = 1
                        logits[layer][scale] = \
                            (self.network_arch_parameters[layer][scale].log_softmax(dim=-1)+gumbels[layer][scale]) / self.tau
                        probs[layer][scale] = F.softmax(logits[layer][scale], dim=-1)
                        #index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                        scale = 2
                        logits[layer][scale] = \
                            (self.network_arch_parameters[layer][scale].log_softmax(dim=-1)+gumbels[layer][scale]) / self.tau
                        probs[layer][scale] = F.softmax(logits[layer][scale], dim=-1)
                        #index[layer][scale] = probs[layer][scale].max(-1, keepdim=True)[1]
                        scale = 3
                        logits[layer][scale][:2] = \
                            (self.network_arch_parameters[layer][scale][:2].log_softmax(dim=-1)*(2/3)+gumbels[layer][scale][:2]) / self.tau
                        probs[layer][scale][:2] = F.softmax(logits[layer][scale][:2], dim=-1) * (2/3)
                        #index[layer][scale][:2] = probs[layer][scale][:2].max(-1, keepdim=True)[1]

            # prob of invalid choice is zero, index can always not select invalid choices.
            index = probs.max(-1, keepdim=True)[1] # [12, 4, 1]
            # according to index, one_hot
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0) # shape as [12, 4, 3]
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue
            else: break

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

        # get single path, according to network-level hardwts, starts from aspp_hardwts.

        def sample_single_path(nb_layers):
            # according hardwts and aspp_hardwts
            # refer viterbi_decode
            single_path = np.zeros(12).astype('uint8')
            with torch.no_grad():
                last_scale = aspp_index
                single_path[-1] = last_scale
                for i in range(1, nb_layers): # [1, 11]
                    single_path[-1-i] = index[-i][single_path[-i]]
            # len()=12, record the next scale from 0-layer(output of stem2)
            return single_path


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
                new_state = _gdas_weighted_sum(
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

