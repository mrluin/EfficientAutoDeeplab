import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

from queue import Queue
from models.cell import Cell
from genotype import PRIMITIVES, Genotype
from modules.operations import *
from run_manager import *
from nas_manager import *
from utils.common import save_inter_tensor

class SplitFabricAutoDeepLab(MyNetwork):
    MODE = None
    def __init__(self, run_config: RunConfig, arch_search_config: ArchSearchConfig, conv_candidates):
        super(SplitFabricAutoDeepLab, self).__init__()

        self._redundant_modules = None
        self._unused_modules = None

        self.run_config = run_config
        self.arch_search_config = arch_search_config
        self.conv_candidates = conv_candidates
        self.nb_layers = self.run_config.nb_layers
        self.nb_classes = self.run_config.nb_classes

        self.cells = nn.ModuleList()
        # network level parameter and binary_gates
        self.fabric_path_alpha = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))
        self.fabric_path_wb = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))

        self.active_index = np.zeros((self.nb_layers, 4, 1)) # active_index always only has one path
        self.inactive_index = None

        # something like mixed operation
        self.log_prob = None
        self.current_prob_over_paths = None

        # TODO: criterion has calculated in run_manager

        # init arch_network_parameters
        # self.arch_network_parameters = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))

        # TODO: architecture params init in nas_manager

        # three init stems
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # all the cell added

        prev_prev_c = 64
        prev_c = 128
        for i in range(self.nb_layers):
            if i == 0:
                cell1 = Cell(self.run_config, self.conv_candidates, 4, prev_c=prev_c, prev_prev_c=None, types=['same'], )
                cell2 = Cell(self.run_config, self.conv_candidates, 8, prev_c=prev_c, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=128, types=['up','same'])
                cell2 = Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                cell3 = Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
            elif i == 2:
                cell1 = Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                cell4 = Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
            elif i == 3:
                cell1 = Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types=['reduction','same','up'])
                cell4 = Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
            else:
                cell1 = Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell4 = Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        # whether passing through ASPP module
        scale4_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 4 / 4)
        scale8_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 8 / 4)
        scale16_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 16 / 4)
        scale32_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 32 / 4)

        # dilation as 96/scale
        self.aspp4 = ASPP(scale4_outc, self.nb_classes, 24, self.run_config.nb_classes)
        self.aspp8 = ASPP(scale8_outc, self.nb_classes, 12, self.run_config.nb_classes)
        self.aspp16 = ASPP(scale16_outc, self.nb_classes, 6, self.run_config.nb_classes)
        self.aspp32 = ASPP(scale32_outc, self.nb_classes, 3, self.run_config.nb_classes)

    @property
    def n_choices(self):
        return 3

    @property
    def probs_over_paths(self):
        probs = F.softmax(self.fabric_path_alpha, dim=-1)
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        # TODO: index type int array
        index = np.argmax(probs, dim=-1)
        return index, probs[index]

    @property
    def random_index(self):
        index = np.zeros((self.nb_layers, 4, 1))
        for l in range(self.nb_layers):
            for s in range(4):
                random_choice = np.random.choice([0,1,2], 1)[0]
                index[l][s][1] = random_choice

        return index

    @property
    def active_fabric_path(self):
        return self.active_index

    def set_chosen_path_active(self):
        index, _ = self.chosen_index
        self.active_index = index
        for l in self.nb_layers:
            for s in range(4):
                self.inactive_index[l][s] = [_i for _i in range(0, index[l][s][0])] + \
                                            [_i for _i in range(index[l][s][0]+1, self.n_choices)]

    def binarize(self):
        self.log_prob = None
        self.fabric_path_wb.data.zero_()

        probs = self.probs_over_paths
        if SplitFabricAutoDeepLab.MODE == 'two':
            sample_path = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.fabric_path_alpha[index] for index in sample_path
            ]), dim=0)
            self.current_prob_over_paths = torch.zeros_like(probs)
            for i, index in enumerate(sample_path):
                self.current_prob_over_paths[index] = probs_slice[i] # set probs to two related paths
            # get one path from total two
            c = torch.multinomial(probs_slice, 1)[0]
            



    @property
    def config(self):
        raise ValueError('not needed')
    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')



    def forward(self, x):
        size = x.size()[2:]
        # keep two value for each scale
        scale4 = []
        scale8 = []
        scale16 = []
        scale32 = []

        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stem2(x)
        save_inter_tensor(scale4, x)

        # TODO: cell weight is useless
        norm_arch_network_alpha = torch.randn(self.nb_layers, 4, 3)#.to('cuda:{}'.format(self.run_config.gpu_ids))
        # need arch_path_alpha rather than alpha_cell
        # norm_alpha_cell = F.softmax(self.alpha_cell, dim=1)

        for layer in range(self.nb_layers):
            if layer == 0:
                norm_arch_network_alpha[layer][0][1:] = F.softmax(self.arch_network_parameters[layer][0][1:], dim=-1)
            elif layer == 1:
                norm_arch_network_alpha[layer][0][1:] = F.softmax(self.arch_network_parameters[layer][0][1:], dim=-1)
                norm_arch_network_alpha[layer][1] = F.softmax(self.arch_network_parameters[layer][1], dim=-1)
            elif layer == 2:
                norm_arch_network_alpha[layer][0][1:] = F.softmax(self.arch_network_parameters[layer][0][1:], dim=-1)
                norm_arch_network_alpha[layer][1] = F.softmax(self.arch_network_parameters[layer][1], dim=-1)
                norm_arch_network_alpha[layer][2] = F.softmax(self.arch_network_parameters[layer][2], dim=-1)
            else:
                norm_arch_network_alpha[layer][0][1:] = F.softmax(self.arch_network_parameters[layer][0][1:], dim=-1)
                norm_arch_network_alpha[layer][1] = F.softmax(self.arch_network_parameters[layer][1], dim=-1)
                norm_arch_network_alpha[layer][2] = F.softmax(self.arch_network_parameters[layer][2], dim=-1)
                norm_arch_network_alpha[layer][3][:2] = F.softmax(self.arch_network_parameters[layer][3][:2], dim=-1)
        count = 0
        for layer in range(self.nb_layers):
            if layer == 0:
                scale4_new, = self.cells[count](None, None, scale4[-1], None,)# norm_alpha_cell)
                scale4_new = scale4_new * norm_arch_network_alpha[layer][0][1]
                count += 1
                scale8_new, = self.cells[count](None, scale4[-1], None, None,)# norm_alpha_cell)
                scale8_new = scale8_new * norm_arch_network_alpha[layer][0][2]
                count += 1
                save_inter_tensor(scale4, scale4_new)
                save_inter_tensor(scale8, scale8_new)
            elif layer == 1:
                scale4_new_1, scale4_new_2, = self.cells[count](scale4[-2], None, scale4[-1], scale8[-1],)# norm_alpha_cell)
                scale4_new = scale4_new_1 * norm_arch_network_alpha[layer][0][1] + scale4_new_2 * norm_arch_network_alpha[layer][1][0]
                count += 1
                scale8_new_1, scale8_new_2, = self.cells[count](None, scale4[-1], scale8[-1], None,)# norm_alpha_cell)
                scale8_new = scale8_new_1 * norm_arch_network_alpha[layer][0][2] + scale8_new_2 * norm_arch_network_alpha[layer][1][1]
                count += 1
                scale16_new, = self.cells[count](None, scale8[-1], None, None,)# norm_alpha_cell)
                scale16_new = scale16_new * norm_arch_network_alpha[layer][1][2]
                count += 1
                save_inter_tensor(scale4, scale4_new)
                save_inter_tensor(scale8, scale8_new)
                save_inter_tensor(scale16, scale16_new)
            elif layer == 2:
                scale4_new_1, scale4_new_2, = self.cells[count](scale4[-2], None, scale4[-1], scale8[-1],)# norm_alpha_cell)
                #print(scale4_new_1.shape, scale4_new_2.shape)
                scale4_new = scale4_new_1 * norm_arch_network_alpha[layer][0][1] + scale4_new_2 * norm_arch_network_alpha[layer][1][0]
                count += 1
                scale8_new_1, scale8_new_2, scale8_new_3 = self.cells[count](scale8[-2], scale4[-1], scale8[-1], scale16[-1],)# norm_alpha_cell)
                scale8_new = scale8_new_1 * norm_arch_network_alpha[layer][0][2] + scale8_new_2 * norm_arch_network_alpha[layer][1][1] + scale8_new_3 * norm_arch_network_alpha[layer][2][0]
                count += 1
                scale16_new_1, scale16_new_2, = self.cells[count](None, scale8[-1], scale16[-1], None,)# norm_alpha_cell)
                scale16_new = scale16_new_1 * norm_arch_network_alpha[layer][1][2] + scale16_new_2 * norm_arch_network_alpha[layer][2][1]
                count += 1
                scale32_new, = self.cells[count](None, scale16[-1], None, None,)# norm_alpha_cell)
                scale32_new = scale32_new * norm_arch_network_alpha[layer][2][2]
                count += 1
                save_inter_tensor(scale4, scale4_new)
                save_inter_tensor(scale8, scale8_new)
                save_inter_tensor(scale16, scale16_new)
                save_inter_tensor(scale32, scale32_new)
            elif layer == 3:
                scale4_new_1, scale4_new_2, = self.cells[count](scale4[-2], None, scale4[-1], scale8[-1],)# norm_alpha_cell)
                scale4_new = scale4_new_1 * norm_arch_network_alpha[layer][0][1] + scale4_new_2 * norm_arch_network_alpha[layer][1][0]
                count += 1
                scale8_new_1, scale8_new_2, scale8_new_3 = self.cells[count](scale8[-2], scale4[-1], scale8[-1], scale16[-1],)# norm_alpha_cell)
                scale8_new = scale8_new_1 * norm_arch_network_alpha[layer][0][2] + scale8_new_2 * norm_arch_network_alpha[layer][1][1] + scale8_new_3 * norm_arch_network_alpha[layer][2][0]
                count += 1
                scale16_new_1, scale16_new_2, scale16_new_3 = self.cells[count](scale16[-2], scale8[-1], scale16[-1], scale32[-1],)# norm_alpha_cell)
                scale16_new = scale16_new_1 * norm_arch_network_alpha[layer][1][2] + scale16_new_2 * norm_arch_network_alpha[layer][2][1] + scale16_new_3 * norm_arch_network_alpha[layer][3][0]
                count += 1
                scale32_new_1, scale32_new_2, = self.cells[count](None, scale16[-1], scale32[-1], None,)# norm_alpha_cell)
                scale32_new = scale32_new_1 * norm_arch_network_alpha[layer][2][2] + scale32_new_2 * norm_arch_network_alpha[layer][3][1]
                count += 1
                save_inter_tensor(scale4, scale4_new)
                save_inter_tensor(scale8, scale8_new)
                save_inter_tensor(scale16, scale16_new)
                save_inter_tensor(scale32, scale32_new)
            else:
                scale4_new_1, scale4_new_2, = self.cells[count](scale4[-2], None, scale4[-1], scale8[-1],)# norm_alpha_cell)
                scale4_new = scale4_new_1 * norm_arch_network_alpha[layer][0][1] + scale4_new_2 * norm_arch_network_alpha[layer][1][0]
                count += 1
                scale8_new_1, scale8_new_2, scale8_new_3 = self.cells[count](scale8[-2], scale4[-1], scale8[-1], scale16[-1],)# norm_alpha_cell)
                scale8_new = scale8_new_1 * norm_arch_network_alpha[layer][0][2] + scale8_new_2 * norm_arch_network_alpha[layer][1][1] + scale8_new_3 * norm_arch_network_alpha[layer][2][0]
                count += 1
                scale16_new_1, scale16_new_2, scale16_new_3 = self.cells[count](scale16[-2], scale8[-1], scale16[-1], scale32[-1],)# norm_alpha_cell)
                scale16_new = scale16_new_1 * norm_arch_network_alpha[layer][1][2] + scale16_new_2 * norm_arch_network_alpha[layer][2][1] + scale16_new_3 * norm_arch_network_alpha[layer][3][0]
                count += 1
                scale32_new_1, scale32_new_2, = self.cells[count](scale32[-2], scale16[-1], scale32[-1], None,)# norm_alpha_cell)
                scale32_new = scale32_new_1 * norm_arch_network_alpha[layer][2][2] + scale32_new_2 * norm_arch_network_alpha[layer][3][1]
                count += 1
                save_inter_tensor(scale4, scale4_new)
                save_inter_tensor(scale8, scale8_new)
                save_inter_tensor(scale16, scale16_new)
                save_inter_tensor(scale32, scale32_new)

        aspp4 = self.aspp4(scale4[-1])
        aspp8 = self.aspp8(scale8[-1])
        aspp16 = self.aspp16(scale16[-1])
        aspp32 = self.aspp32(scale32[-1])

        aspp4 = F.interpolate(aspp4, size=size, mode='bilinear', align_corners=True)
        aspp8 = F.interpolate(aspp8, size=size, mode='bilinear', align_corners=True)
        aspp16 = F.interpolate(aspp16, size=size, mode='bilinear', align_corners=True)
        aspp32 = F.interpolate(aspp32, size=size, mode='bilinear', align_corners=True)

        return aspp4 + aspp8 + aspp16 + aspp32

    def decode_network(self):
        # TODO: dfs and viterbi
        best_result = []
        max_prop = 0.
        def _parse(network_weight, layer, curr_value, curr_result, last):
            nonlocal best_result
            nonlocal max_prop
            if layer == self.nb_layers:
                if max_prop < curr_value:
                    best_result = curr_result[:]
                    max_prop = curr_value
                return
            if layer == 0:
                print('begin layer 0')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end0-1')
            elif layer == 1:
                print('begin layer 1')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end1-2')
            elif layer == 2:
                print('begin layer 2')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end1-2')
                scale = 2
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end2-1')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end2-2')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end2-3')

            else:
                print('begin layer {}'.format(layer))
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end1-2')

                scale = 2
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end2-1')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end2-2')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    print('end2-3')
                scale = 3
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    print('end3-2')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    print('end3-3')
        network_weight = F.softmax(self.alpha_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print(max_prop)
        return best_result

    def genotype(self):
        # TODO cell decode
        raise NotImplementedError
    '''
    def initialize_alphas(self):
        # TODO: get rid of this method
        # TODO: add init ratio
        alphas_network = torch.tensor(1e-3 * torch.randn(self.nb_layer, 4, 3), device='cuda:{}'.format(self.run_config.gpu_id), requires_grad=True)
        self.register_parameter(self.arch_param_names[1], nn.Parameter(alphas_network))
        self.alpha_network_mask = torch.ones(self.nb_layers, 4, 3)
        '''
    def architecture_path_parameters(self):
        # only architecture_path_parameters, within cells
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def architecture_network_parameters(self):
        # only architecture_network_parameters, network level
        for name, param in self.named_parameters():
            if 'arch_network_parameters' in name:
                yield param

    def binary_gates(self):
        # only binary gates
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield  param

    def weight_parameters(self):
        # network weight parameters
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'arch_network_parameters' not in name and 'AP_path_wb' not in name:
                yield  param

    def architecture_parameters(self):
        # architecture_path_parameters and architecture_network_parameters
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name or 'arch_network_parameters' in name:
                yield param

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_path_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

        for param in self.architecture_network_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    @property
    def redundant_modules(self):
        # proxy related to modules
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules =  module_list
        # get all the mixededge module
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            #try:
                #print(dir(m))
                #print(hasattr(m, 'binarize')) # True
            m.binarize()
            # except AttributeError:
            #    print(type(m), 'do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            #try:
            m.set_arch_param_grad()
            #except AttributeError:
            #    print(type(m), 'do not support set_arch_param_grad')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            #try:
            m.rescale_updated_arch_param()
            #except AttributeError:
            #    print(type(m), 'do not support rescale_updated_arch_param')

    def unused_modules_off(self):
        self._unused_modules = [] # [unused_modules for redundant1, 2, ...] for each MixedEdges
        for m in self.redundant_modules:
            unused = {} # unused modules in each redundant modules
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i] # unused, index, operations
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        # m: MixedEdge, related unused_modules
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused: # i, index
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            #try:
            m.set_chosen_op_active()
            #except AttributeError:
            #    print(type(m), 'do not support set_chosen_op_active')

    def set_active_via_net(self, net):
        # setting from existing net
        assert isinstance(net, AutoDeepLab)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)
    def expected_latency(self):
        raise NotImplementedError

    def expected_flops(self):
        raise NotImplementedError

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            # module._modules is OrderedDict
            # m is module name
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op # get the operation with max probability and apply to the path
                else:
                    queue.put(child)
        return AutoDeepLab(self.run_config, self.arch_search_config)