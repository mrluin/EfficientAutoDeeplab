import torch.nn.functional as F
import numpy as np
import copy

from queue import Queue
from models.cell import Cell
from models.proxy_cell import Proxy_cell
from genotype import PRIMITIVES, Genotype
from modules.operations import *
from run_manager import *
from nas_manager import *
from utils.common import save_inter_tensor
from utils.common import count_conv_flop
from utils.common import get_prev_c
from utils.common import get_cell_decode_type
from utils.common import get_list_index
from utils.common import count_normal_conv_flop
from utils.common import network_layer_to_space
from collections import OrderedDict

class ProxyAutoDeepLab(MyNetwork):
    def __init__(self, run_config: RunConfig, arch_search_config: ArchSearchConfig, conv_candidates):
        super(ProxyAutoDeepLab, self).__init__()

        self._redundant_modules = None
        self._unused_modules = None
        self.cells = nn.ModuleList()

        self.run_config = run_config
        self.arch_search_config = arch_search_config
        self.conv_candidates = conv_candidates

        self.nb_layers = self.run_config.nb_layers
        self.nb_classes = self.run_config.nb_classes
        # TODO: criterion has calculated in run_manager

        # init arch_network_parameters
        self.arch_network_parameters = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))
        # TODO: architecture params init in nas_manager

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

        prev_prev_c = 32
        prev_c = 64
        for i in range(self.nb_layers):
            if i == 0:
                cell1 = Proxy_cell(self.run_config, self.conv_candidates, 4, prev_c=prev_c, prev_prev_c=None, types=['same'], )
                cell2 = Proxy_cell(self.run_config, self.conv_candidates, 8, prev_c=prev_c, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = Proxy_cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=64, types=['up','same'])
                cell2 = Proxy_cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                cell3 = Proxy_cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
            elif i == 2:
                cell1 = Proxy_cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Proxy_cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Proxy_cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                cell4 = Proxy_cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types=['reduction'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
            elif i == 3:
                cell1 = Proxy_cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Proxy_cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Proxy_cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types=['reduction','same','up'])
                cell4 = Proxy_cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types=['reduction','same'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
            else:
                cell1 = Proxy_cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types=['same', 'up'])
                cell2 = Proxy_cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell3 = Proxy_cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same', 'up'])
                cell4 = Proxy_cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=-1, types=['reduction', 'same'])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        scale4_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 4 / 4)
        scale8_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 8 / 4)
        scale16_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 16 / 4)
        scale32_outc = int(self.run_config.filter_multiplier * self.run_config.block_multiplier * 32 / 4)

        # dilation as 96/scale
        self.aspp4 = ASPP(scale4_outc, self.nb_classes, 24, self.run_config.nb_classes)
        self.aspp8 = ASPP(scale8_outc, self.nb_classes, 12, self.run_config.nb_classes)
        self.aspp16 = ASPP(scale16_outc, self.nb_classes, 6, self.run_config.nb_classes)
        self.aspp32 = ASPP(scale32_outc, self.nb_classes, 3, self.run_config.nb_classes)

        self.set_bn_param(momentum=self.run_config.bn_momentum, eps=self.run_config.bn_eps)

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
        norm_arch_network_alpha = torch.randn(self.nb_layers, 4, 3).to(x.device)
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
        max_prob = 0.
        def _parse(network_weight, layer, curr_value, curr_result, last):
            nonlocal best_result
            nonlocal max_prob
            if layer == self.nb_layers:
                if max_prob < curr_value:
                    best_result = curr_result[:]
                    max_prob = curr_value
                return
            if layer == 0:
                #print('begin layer 0')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end0-1')
            elif layer == 1:
                #print('begin layer 1')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end1-2')
            elif layer == 2:
                #print('begin layer 2')
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end1-2')
                scale = 2
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end2-1')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end2-2')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end2-3')

            else:
                #print('begin layer {}'.format(layer))
                scale = 0
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end0-0')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end0-1')
                scale = 1
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 0])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=0)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end1-0')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end1-1')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end1-2')

                scale = 2
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 1])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=1)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end2-1')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end2-2')
                    curr_value = curr_value * network_weight[layer][scale][2]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][2]
                    curr_result.pop()
                    #print('end2-3')
                scale = 3
                if last == scale:
                    curr_value = curr_value * network_weight[layer][scale][0]
                    curr_result.append([scale, 2])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=2)
                    curr_value = curr_value / network_weight[layer][scale][0]
                    curr_result.pop()
                    #print('end3-2')
                    curr_value = curr_value * network_weight[layer][scale][1]
                    curr_result.append([scale, 3])
                    _parse(network_weight, layer+1, curr_value, curr_result, last=3)
                    curr_value = curr_value / network_weight[layer][scale][1]
                    curr_result.pop()
                    #print('end3-3')
        network_weight = F.softmax(self.arch_network_parameters, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print('\tDecode Network max_prob:', max_prob)
        return best_result

    def viterbi_decode(self):

        #network_space = torch.zeros(self.nb_layers, 4, 3) # [12, 4, 3]
        network_space = np.zeros((self.nb_layers, 4, 3))
        for layer in range(self.nb_layers):
            if layer == 0:
                network_space[layer][0][1:] = F.softmax(self.arch_network_parameters.data[layer][0][1:], dim=-1) * (
                            2 / 3)
            elif layer == 1:
                network_space[layer][0][1:] = F.softmax(self.arch_network_parameters.data[layer][0][1:], dim=-1) * (
                            2 / 3)
                network_space[layer][1] = F.softmax(self.arch_network_parameters.data[layer][1], dim=-1)
            elif layer == 2:
                network_space[layer][0][1:] = F.softmax(self.arch_network_parameters.data[layer][0][1:], dim=-1) * (
                            2 / 3)
                network_space[layer][1] = F.softmax(self.arch_network_parameters.data[layer][1], dim=-1)
                network_space[layer][2] = F.softmax(self.arch_network_parameters.data[layer][2], dim=-1)
            else:
                network_space[layer][0][1:] = F.softmax(self.arch_network_parameters.data[layer][0][1:], dim=-1) * (
                            2 / 3)
                network_space[layer][1] = F.softmax(self.arch_network_parameters.data[layer][1], dim=-1)
                network_space[layer][2] = F.softmax(self.arch_network_parameters.data[layer][2], dim=-1)
                network_space[layer][3][:2] = F.softmax(self.arch_network_parameters.data[layer][3][:2], dim=-1) * (
                            2 / 3)

        prob_space = np.zeros((network_space.shape[:2]))
        path_space = np.zeros((network_space.shape[:2])).astype('int8')

        # prob_space [layer, sample] means the layer-the choice go to sample-th scale
        # network space 0 ↗, 1 →, 2 ↘  , rate means choice
        # path_space    1    0   -1      1-rate means path
        for layer in range(network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = network_space[layer][0][1] # 0-layer go to next 0-scale prob
                prob_space[layer][1] = network_space[layer][0][2] # 0-layer go to next 1-scale prob

                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(network_space.shape[1]):
                    if sample > layer + 1: # control valid sample in each layer
                        continue
                    local_prob = []
                    for rate in range(network_space.shape[2]):
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            # if the next scale is 0, does not come from rate 2: reduction
                            # if the next scale is 3, does not come from rate 0: up
                            continue
                        else:
                            # sample is target scale, sample+(1-rate) is current scale
                            # prob_space[layer-1][sample+(1-rate)], the prob of last layer to current scale
                            # rate = 0, current to target up, then current is target + 1 (i.e.) 1-rate = 1
                            # rate = 1, current to target same, then current is the same as target 1-rate = 0
                            # rate = 2, current to target reduce, then current is target - 1 (i.e.) 1-rate = -1
                            local_prob.append(prob_space[layer-1][sample+1-rate]*
                                              network_space[layer][sample+1-rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1-rate if sample != 3 else -rate
                    path_space[layer][sample] = path
        output_sample = np.argmax(prob_space[-1, :], axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample # have known tha last scale
        for i in range(1, self.nb_layers): # get scale path according to path_space
            actual_path[-i-1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

        return actual_path,  network_layer_to_space(actual_path, 12)


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
        #print(self._unused_modules)

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
            m.set_chosen_op_active()

    def set_active_via_net(self, net):
        # setting from existing net
        assert isinstance(net, ProxyAutoDeepLab)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)
    def expected_latency(self): # TODO: related to gradient_step()
        raise NotImplementedError

    def expected_flops(self):  # TODO: related to gradient_step()
        raise NotImplementedError


    def module_str(self):
        best_result = self.decode_network()
        log_str = 'Network-Level-Best-Result:\n {}\n'.format(best_result)

        stem0_module_str = 'Stem_s{}_{}x{}ConvBnReLU'.format(self.stem0.conv.stride[0],
                                                             self.stem0.conv.kernel_size[0],
                                                             self.stem0.conv.kernel_size[1])
        stem1_module_str = 'Stem_s{}_{}x{}ConvBnReLU'.format(self.stem1.conv.stride[0],
                                                             self.stem1.conv.kernel_size[0],
                                                             self.stem1.conv.kernel_size[1])
        stem2_module_str = 'Stem_s{}_{}x{}ConvBnReLU'.format(self.stem2.conv.stride[0],
                                                             self.stem2.conv.kernel_size[0],
                                                             self.stem2.conv.kernel_size[1])
        log_str += '{}. {}\n'.format(0, stem0_module_str)
        log_str += '{}. {}\n'.format(1, stem1_module_str)
        log_str += '{}. {}\n'.format(2, stem2_module_str)

        for layer in range(self.nb_layers):
            prev_prev_c = False
            if layer >= 1 and best_result[layer][1] == best_result[layer-1][0]:
                prev_prev_c = True
            current_scale = best_result[layer][0]
            next_scale = best_result[layer][1]
            index = get_list_index(layer, next_scale)
            type = get_cell_decode_type(current_scale, next_scale)
            frag_cell_log = '(Layer {} Scale {} Index {})\n'\
                                .format(layer + 1, next_scale, index) + self.cells[index].module_str(type)  # each proxy cell and its mixed operations
            log_str += frag_cell_log

        last_scale = best_result[-1][1]
        if last_scale == 0:
            log_str += 'Final:\t{}\n'.format(self.aspp4.module_str())
        elif last_scale == 1:
            log_str += 'Final:\t{}\n'.format(self.aspp8.module_str())
        elif last_scale == 2:
            log_str += 'Final:\t{}\n'.format(self.aspp16.module_str())
        elif last_scale == 3:
            log_str += 'Final:\t{}\n'.format(self.aspp32.module_str())
        return log_str

    def get_flops(self, x):
        # TODO: change into using viterbi algorithm
        # x is the tensor with the same shape of input
        # get each cell flops and aspp flops
        # TODO: there are some issues in self.convert_to_normal_net(), will have effect on the following module_str()
        #cell_decode_network = self.convert_to_normal_net() # cell path level
        #best_result = self.decode_network() # [(scale, next_scale)] * 12 network path level
        actual_path, _ = self.viterbi_decode()
        #assert len(best_result) == self.run_config.nb_layers, 'Error in self.net.decode_network'
        assert len(actual_path) == self.nb_layers, 'Error in actual_path of net.get_flops'
        flops = 0.
        flop_stem0 = count_normal_conv_flop(self.stem0.conv, x)
        x = self.stem0(x)
        flop_stem1 = count_normal_conv_flop(self.stem1.conv, x)
        x = self.stem1(x)
        flop_stem2 = count_normal_conv_flop(self.stem2.conv, x)
        x = self.stem2(x)

        prev_scale = 0
        inter_features = [[0, None], [0, x]] # save prev_prev_output and prev_output
        for layer in range(self.nb_layers):
            current_scale = prev_scale
            #current_scale = best_result[layer][0]
            next_scale = actual_path[layer] # scale of layer+1
            prev_prev_c, prev_c = get_prev_c(inter_features, next_scale)
            type = get_cell_decode_type(current_scale, next_scale)
            index = get_list_index(layer, next_scale)
            frag_flop, out = self.cells[index].get_flops(prev_prev_c, prev_c, type)
            flops = flops + frag_flop
            prev_scale = next_scale
            inter_features.pop(0)
            inter_features.append([next_scale, out])

        # aspp flops
        last_scale = inter_features[-1][0]

        #print(last_scale)
        if last_scale == 0:
            flop_aspp, output = self.aspp4.get_flops(inter_features[-1][1])
        elif last_scale == 1:
            flop_aspp, output = self.aspp8.get_flops(inter_features[-1][1])
        elif last_scale == 2:
            flop_aspp, output = self.aspp16.get_flops(inter_features[-1][1])
        elif last_scale == 3:
            flop_aspp, output = self.aspp32.get_flops(inter_features[-1][1])
        else:
            raise ValueError('invalid scale choice of {}'.format(last_scale))
        return flop_stem1 + flop_stem0 + flop_stem2 + flops + flop_aspp, output


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
        return ProxyAutoDeepLab(self.run_config, self.arch_search_config, self.conv_candidates)

    def cell_arch_decode(self):
        genes = [] # [nb_cells, nb_edges, edge_index, best_op]
        # TODO: confirm nb_choices
        nb_choices = 7
        def _parse(alphas, steps, has_none):
            # TODO: just include None edge, probs of all operation are all zero, it will never be selected
            gene = []
            start = 0
            n = 2  # offset
            for i in range(steps):
                end = start + n
                # all the edge ignore Zero operation TODO: reconfirm Zero operation index
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))
                top1edge = edges[0]  # edge index
                best_op_index = np.argmax(alphas[top1edge])  #
                gene.append([top1edge, best_op_index])
                start = end
                n += 1  # move offset

                # len(gene) related to steps, each step chose one path
                # shape as [nb_steps, operation_index]

            return np.array(gene)

        # todo alphas is AP_path_alpha for all the paths in each cell not single node

        # TODO: nb_edges in cells
        nb_edges = 2
        for cell in self.cells:
            alpha = np.zeros((nb_edges, nb_choices))
            has_none = False
            for index, op in enumerate(cell.ops):
                #print(index)
                # each op is MobileInvertedResidual
                # MixedEdge is op.mobile_inverted_conv
                # Each MixedEdge has 'None' case, when prev_prev_c is None and edge_index==0
                # so the cell_arch of each cell in fabric will raise size mismatch error
                # TODO: each cell_arch list or array, cannot use concatenate
                # if mobile_inverted_conv is None and shortcut is None, then ops.appends(None)
                if op is None:
                    #print('find None operation')
                    assert index == 0, 'invalid edge_index, {} is None'.format(index)
                    has_none = True
                elif op is not None:
                    mixededge = op.mobile_inverted_conv
                    assert mixededge.__str__().startswith('MixedEdge'), 'Error in cell_arch_decode'
                    alpha[index] = mixededge.AP_path_alpha.data.cpu().numpy()
            #print(alpha)
            #print(alpha.shape)
            # alpha is a list, including [path_index, path_alpha] in a cell
            gene = _parse(alpha, self.run_config.steps, has_none)
            #print('---')
            #print(gene)
            #print(gene)
            genes.append(gene)
            # return genes, select which edge, which operation in each cell
            # [path_index, operation_index]
        return np.array(genes)
        # shape as [nb_cells, nb_steps, operation_index]

    def network_arch_cell_arch_decode(self):
        actual_path, network_space = self.viterbi_decode()
        assert len(actual_path) == 12, 'invalid actual_path length {}'.format(len(actual_path))
        nb_edges = len([_j for _i in range(self.run_config.steps) for _j in range(_i+2)])
        # TODO: property object cannot be interpreted as an integer
        nb_choices = 7

        gene = []

        def _parse(cell_alphas, steps):
            _gene = []
            start = 0
            n = 2  # offset
            for i in range(steps):
                end = start + n
                # TODO: reconfirm Zero operation index
                edges = sorted(range(start, end), key=lambda x: -np.max(cell_alphas[x, :-1]))
                top1edge = edges[0]  # edge index
                best_op_index = np.argmax(cell_alphas[top1edge])  #
                _gene.append([top1edge, best_op_index])
                start = end
                n += 1  # move offset
            return _gene

        for i in range(self.nb_layers):
            next_scale = actual_path[i]
            cell_index = get_list_index(i, next_scale)
            # just include None edge, probs of all operation are all zero, it will never be selected
            alpha = np.zeros((nb_edges, nb_choices)) # shape as [n, 7]
            for op_index, op in enumerate(self.cells[cell_index].ops):
                if op is None:
                    # None operations are not all at index 0, in the case of multi-nodes
                    continue
                else:
                    mixededge = op.mobile_inverted_conv
                    assert mixededge.__str__().startswith('MixedEdge'), 'Error in cell_arch_decode'
                    alpha[op_index] = mixededge.AP_path_alpha.data.cpu().numpy()
            _gene = _parse(alpha, self.run_config.steps)

            gene.append(_gene)

        #print(len(gene))
        assert len(gene) == 12, 'Error in network_arch_cell_arch_decode'
        return actual_path, network_space, np.array(gene)





