import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

from models.cell import Split_Cell

from queue import Queue

from genotype import PRIMITIVES, Genotype
from modules.operations import *
from run_manager import RunConfig
from utils.common import save_inter_tensor, get_cell_decode_type, get_list_index_split
from utils.common import get_next_scale, get_list_index, get_prev_c, delta_ij, detach_variable
from collections import OrderedDict
from models.new_model import network_layer_to_space
from modules.mixed_op import MixedEdge


class SplitFabricAutoDeepLab(MyNetwork):
    MODE = None
    def __init__(self, run_config: RunConfig, conv_candidates):
        super(SplitFabricAutoDeepLab, self).__init__()

        self._redundant_modules = None
        self._unused_modules = None
        self.cells = nn.ModuleList()

        self.run_config = run_config
        #self.arch_search_config = arch_search_config
        self.conv_candidates = conv_candidates

        self.nb_layers = self.run_config.nb_layers
        self.nb_classes = self.run_config.nb_classes

        # network level parameter and binary_gates
        self.fabric_path_alpha = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))
        self.fabric_path_wb = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))

        # TODO: active_index and inactive_index should not be list, but tensors
        # and list_index is calculate by layer and scales, active_index is initialized as -1
        self.active_index = torch.zeros((self.nb_layers, 4, 1), dtype=torch.int64)

        self.inactive_index = None # inactive_index.shape is not fixed, 'two' mode is one and other mode are two
        self.tmp_active_alphas = None
        self.tmp_inactive_alphas = None


        # something like mixed operation
        self.log_prob = None
        self.current_prob_over_paths = None

        # TODO: criterion has calculated in run_manager

        # init arch_network_parameters
        # self.arch_network_parameters = nn.Parameter(torch.Tensor(self.nb_layers, 4, 3))

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

        # all the cell added

        prev_prev_c = 32
        prev_c = 64
        for i in range(self.nb_layers):
            if i == 0:
                cell1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=prev_c, prev_prev_c=None, types='same', )
                cell2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=prev_c, prev_prev_c=None, types='reduction')
                self.cells += [cell1] # 0
                self.cells += [cell2] # 1
            elif i == 1:
                cell1_1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=64, types='same')
                cell1_2 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=64, types='up')
                cell2_1 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=None, types='reduction')
                cell2_2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=None, types='same')
                cell3 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types='reduction')
                self.cells += [cell1_1] # 3
                self.cells += [cell2_1] # 4
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3]
            elif i == 2:
                cell1_1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='same')
                cell1_2 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='up')
                cell2_1 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell2_2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='same')
                cell2_3 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='up')
                cell3_1 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types='reduction')
                cell3_2 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=None, types='same')
                cell4 = Split_Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types='reduction')
                self.cells += [cell1_1]
                self.cells += [cell2_1]
                self.cells += [cell1_2]
                self.cells += [cell2_2]
                self.cells += [cell3_1]
                self.cells += [cell2_3]
                self.cells += [cell3_2]
                self.cells += [cell4]
            elif i == 3:
                cell1_1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='same')
                cell1_2 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='up')
                cell2_1 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell2_2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='same')
                cell2_3 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='up')
                cell3_1 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell3_2 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='same')
                cell3_3 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='up')
                cell4_1 = Split_Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types='reduction')
                cell4_2 = Split_Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=None, types='same')
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
                cell1_1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='same')
                cell1_2 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=-1, types='up')
                cell2_1 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell2_2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='same')
                cell2_3 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=-1, prev_prev_c=-1, types='up')
                cell3_1 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell3_2 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='same')
                cell3_3 = Split_Cell(self.run_config, self.conv_candidates, 16, prev_c=-1, prev_prev_c=-1, types='up')
                cell4_1 = Split_Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=-1, types='reduction')
                cell4_2 = Split_Cell(self.run_config, self.conv_candidates, 32, prev_c=-1, prev_prev_c=-1, types='same')
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

        self.set_bn_param(momentum=self.run_config.bn_momentum, eps=self.run_config.bn_eps)

    @property
    def config(self):
        raise ValueError('not needed')
    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    @property
    def n_choices(self):
        # path choices
        return 3

    @property
    def probs_over_paths(self):
        # todo fix the invalid value issue in scale zero and scale three
        probs = torch.zeros(self.nb_layers, 4, 3)
        for layer in range(self.nb_layers):
            for scale in range(4):
                if scale == 0:
                    probs[layer][scale][1:] = F.softmax(self.fabric_path_alpha.data[layer][scale][1:], dim=-1)
                elif scale == 1 or scale == 2:
                    probs[layer][scale] = F.softmax(self.fabric_path_alpha.data[layer][scale], dim=-1)
                elif scale == 3:
                    probs[layer][scale][:2] = F.softmax(self.fabric_path_alpha.data[layer][scale][:2], dim=-1)
        return probs # [12, 4, 3]

    @property
    def chosen_index(self):
        # confirmed
        probs = self.probs_over_paths
        index = torch.argmax(probs, dim=-1, keepdim=True)#np.argmax(probs, axis=-1) # shape as [self.nb_layers, scales]
        # shape as [12, 4, 1]
        # torch.argmax returns int64
        return index, torch.gather(probs, -1, index)

    @property
    def active_fabric_path(self):
        # confirmed
        return self.active_index # tensor with shape [12,4,3]

    def set_chosen_path_active(self):
        # confirmed
        index, _ = self.chosen_index
        # index with shape [12, 4, 1], self.active_index is initialized all the -1
        #for l in self.nb_layers:
        #    for s in range(4):
        #        self.active_index[l][s] = index[l][s]
        self.active_index = index
        for l in range(self.nb_layers):
            for s in range(4):
                # self.inactive_index is also tensor, with shape [12, 4, 2]
                self.inactive_index[l][s] = torch.from_numpy(np.array([_i for _i in range(0, index[l][s][0])] + \
                                                                      [_i for _i in range(index[l][s][0]+1, self.n_choices)]))

    @property
    def random_index(self):
        # cannot perform random index for each node in fabric, the nodes should have spatial wise relationship.
        active_index = np.zeros((self.nb_layers, 4, 1), dtype=torch.int64)
        random_result = []
        last = 0
        layer = 0
        def random_dfs(layer, random_result, last):
            #nonlocal random_result
            nonlocal active_index
            if layer == 0:
                scale = 0
                if last == scale:
                    sample_index = np.random.choice([1, 2], 1)[0] # item
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index]) # [scale, choice] x 12
                    random_dfs(layer+1, random_result, last=get_next_scale(sample_index, scale))
            elif layer == 1:
                scale = 0
                if last == scale:
                    sample_index = np.random.choice([1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer+1, random_result, last=get_next_scale(sample_index, scale))
                scale = 1
                if last == scale:
                    sample_index = np.random.choice([0, 1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer+1, random_result, last=get_next_scale(sample_index, scale))
            elif layer == 2:
                scale = 0
                if last == scale:
                    sample_index = np.random.choice([1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
                scale = 1
                if last == scale:
                    sample_index = np.random.choice([0, 1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
                scale = 2
                if last == scale:
                    sample_index = np.random.choice([0, 1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
            else:
                scale = 0
                if last == scale:
                    sample_index = np.random.choice([1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
                scale = 1
                if last == scale:
                    sample_index = np.random.choice([0, 1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
                scale = 2
                if last == scale:
                    sample_index = np.random.choice([0, 1, 2], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
                scale = 3
                if last == scale:
                    sample_index = np.random.choice([0, 1], 1)[0]
                    active_index[layer][scale][0] = sample_index
                    random_result.append([scale, sample_index])
                    random_dfs(layer + 1, random_result, last=get_next_scale(sample_index, scale))
        # from layer 0, random_result is [], last = 0
        random_dfs(layer, random_result, last)
        print('\t-> random_path_result:', random_result)
        return active_index, random_result

    ''' modifications '''
    # todo self.active_index should be [12, 4, 1]



    def binarize(self):
        #print('super_network binarize')
        # confirmed
        # binarize should apply on each node in fabric, should apply to the whole fabric
        # network, rather than each node separable
        #self.log_prob = torch.Tensor(self.nb_layers, 4, 1)
        probs = self.probs_over_paths.data  # shape as [12, 4, 3]
        self.fabric_path_wb.data.zero_()
        self.current_prob_over_paths = torch.zeros_like(probs)

        if SplitFabricAutoDeepLab.MODE == 'two':
            self.inactive_index = torch.zeros((self.nb_layers, 4, 1), dtype=torch.int64)
        else:
            self.inactive_index = torch.zeros((self.nb_layers, 4, 2), dtype=torch.int64)


        # todo this reset operation does not account for
        #self.active_index = torch.zeros(self.nb_layers, 4, 1) - 1
        #print(self.fabric_path_alpha)

        # argument layer, last=0
        def _binarize(layer, scale):
            #print('\t->Binarizing layer {} scale {}'.format(layer, scale)) # used for test method
            if SplitFabricAutoDeepLab.MODE == 'two':
                #self.inactive_index = torch.zeros(self.nb_layers, 4, 1)
                if scale == 0:
                    # sample_paths should be used as index
                    sample_paths = torch.multinomial(probs.data[layer][scale][1:], 2, replacement=False)
                    sample_paths = sample_paths + 1
                elif scale == 3:
                    sample_paths = torch.multinomial(probs.data[layer][scale][:2], 2, replacement=False)
                else:
                    sample_paths = torch.multinomial(probs.data[layer][scale], 2, replacement=False)
                probs_slice = F.softmax(torch.stack([
                    self.fabric_path_alpha[layer][scale][index] for index in sample_paths
                ]), dim=0)  # shape as [2]

                for i, index in enumerate(sample_paths):
                    self.current_prob_over_paths[layer][scale][index] = probs_slice[i]

                c = torch.multinomial(probs_slice.data, 1)[0]  # select 0 index or 1 index
                # index
                active_path = sample_paths[c].item()
                inactive_path = sample_paths[1 - c].item()

                self.active_index[layer][scale] = active_path
                self.inactive_index[layer][scale] = inactive_path
                self.fabric_path_wb.data[layer][scale][active_path] = 1.0
            else:
                #self.inactive_index = torch.zeros(self.nb_layers, 4, 2)
                if scale == 0:
                    sample_path = torch.multinomial(probs.data[layer][scale][1:], 1)[0].item()
                    sample_path = sample_path + 1
                    #print('sample_path', sample_path)
                elif scale == 3:
                    sample_path = torch.multinomial(probs.data[layer][scale][:2], 1)[0].item()
                else:
                    sample_path = torch.multinomial(probs.data[layer][scale], 1)[0].item()
                self.active_index[layer][scale] = sample_path
                self.inactive_index[layer][scale] = torch.from_numpy(np.array(
                    [[_i for _i in range(0, sample_path)] + [_i for _i in range(sample_path + 1, self.n_choices)]]
                ))
                self.current_prob_over_paths[layer][scale] = probs[layer][scale]
                #self.log_prob[layer][scale] = torch.log(probs[layer][scale][sample_path])
                self.fabric_path_wb.data[layer][scale][sample_path] = 1.0

            #print(self.active_index)
            last_scale = scale
            if self.active_index[layer][scale] == 0:
                if scale == 0:
                    raise ValueError('scale 0 do not support IncreasedOperation')
                last_scale = scale - 1
            elif self.active_index[layer][scale] == 1:
                last_scale = scale
            elif self.active_index[layer][scale] == 2:
                if scale == 3:
                    raise ValueError('scale 3 do not support ReducedOperation')
                last_scale = scale + 1

            assert last_scale >= 0 and last_scale <=3, 'invalid value of last_scale'
            return last_scale

        # binarize of each node in fabric
        last_scale = 0
        for layer in range(self.nb_layers):
            if layer == 0:
                scale = 0
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
            elif layer == 1:
                scale = 0
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 1
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
            elif layer == 2:
                scale = 0
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 1
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 2
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
            else:
                scale = 0
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 1
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 2
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue
                scale = 3
                if last_scale == scale:
                    last_scale = _binarize(layer, scale)
                    continue

        # TODO: avoid over-regularization

    def set_fabric_arch_param_grad(self):
        # confirmed
        # calculate gradient of network_arch_alpha based on gradients of binary gates
        fabric_binary_grads = self.fabric_path_wb.grad.data # shape as [12, 4, 3]
        if self.fabric_path_alpha.grad is None:
            self.fabric_path_alpha.grad = torch.zeros_like(self.fabric_path_alpha.data)

        _fabric_path_alpha = self.fabric_path_alpha.data.cpu()
        if SplitFabricAutoDeepLab.MODE == 'two':
            # [12, 4, 1] concat [12, 4, 1] = [12, 4, 2]
            #print(self.active_index, self.inactive_index)
            involved_idx = torch.cat([self.active_index, self.inactive_index], dim=-1) # CPU VERSION
            # shape [12, 4, 2]
            #print(self.fabric_path_alpha.data.device, involved_idx.device)
            probs_slice = F.softmax(torch.gather(_fabric_path_alpha, -1, involved_idx), -1).to(self.fabric_path_alpha.device)
            for l in range(self.nb_layers):
                for s in range(4):
                    # compute for each node in fabric
                    for i in range(2):
                        for j in range(2):
                            origin_i = involved_idx[l][s][i]
                            origin_j = involved_idx[l][s][j]
                            self.fabric_path_alpha.grad.data[origin_i] += \
                                fabric_binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            # TODO: set tuple, index, alpha for rescale
            # set_grad -> optimizer -> if two rescale
            # saving current grad
            # shape as [12, 4, 1]
            self.tmp_active_alphas = torch.gather(_fabric_path_alpha, -1, self.active_index)
            self.tmp_inactive_alphas = torch.gather(_fabric_path_alpha, -1, self.inactive_index) # shape as [12, 4, 1]
        else:
            probs = self.probs_over_paths.data.to(self.fabric_path_alpha.device)
            for l in range(self.nb_layers):
                for s in range(4):
                    for i in range(self.n_choices):
                        for j in range(self.n_choices):
                            self.fabric_path_alpha.grad.data[i] += fabric_binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])

        return

    def rescale_updated_fabric_arch_param(self):
        # only under the case of 'two'
        _fabric_path_alpha = self.fabric_path_alpha.data.cpu()
        involved_index = torch.concat([self.active_index, self.inactive_index], dim=-1) # [12, 4, 1] concat [12, 4, 1]
        old_alphas = torch.concat([self.tmp_active_alphas, self.tmp_inactive_alphas], dim=-1) # shape as [12, 4, 2]
        new_alphas = torch.gather(_fabric_path_alpha, -1, involved_index, dim=-1) # shape as [12, 4, 2]

        offset = None # shape as [self.nb_layers, 4, 1]
        for l in range(self.nb_layers):
            for s in range(4):
                offset[l][s][0] = math.log(sum([math.exp(alpha) for alpha in new_alphas[l][s]]) / sum([math.exp(alpha) for alpha in old_alphas[l][s]]))

        for l in range(self.nb_layers):
            for s in range(4):
                for index in involved_index[l][s]:
                    self.fabric_path_alpha.data[l][s][index] -= offset[l][s][0]

    def _forward(self, layer, scale, cell_index, prev_prev_c, prev_c):

        if SplitFabricAutoDeepLab.MODE == 'full' or SplitFabricAutoDeepLab.MODE == 'two':
            output = 0
            # cells have been appended in ordered
            for _i in self.active_index[layer][scale]:
                # next_scale = get_next_scale(_i, scale)
                # cell_index = get_list_index_split(layer, scale, next_scale)
                output_i = self.cells[cell_index](prev_prev_c, prev_c)
                output = output + self.fabric_path_wb[layer][scale][_i] * output_i
            for _i in self.inactive_index[layer][scale]:
                # next_scale = get_next_scale(_i, scale)
                # cell_index = get_list_index_split(layer, scale, next_scale)
                output_i = self.cells[cell_index](prev_prev_c, prev_c)
                output = output + self.fabric_path_wb[layer][scale][_i] * output_i.detach()

        elif SplitFabricAutoDeepLab.MODE == 'full_v2':
            def run_function(candidate_paths, active_id, cell_index):  # change active_id to cell_index
                def fforward(_prev_prev_c, _prev_c):
                    #print('_forward of layer {} scale {} cell_index {}'.format(layer, scale, cell_index))
                    return candidate_paths[cell_index](_prev_prev_c, _prev_c)
                return fforward

            def backward_function(candidate_paths, active_id, cell_index, binary_gates):
                def backward(_prev_prev_c, _prev_c, _output, grad_output):
                    print('_backward of layer {} scale {} cell_index {}'.format(layer, scale, cell_index))
                    # print('_backward for layer {} scale {} cell_index {}'.format(layer, scale, cell_index))
                    # print('before super_network backwards _forward: ',
                    #      candidate_paths[cell_index].ops[1].mobile_inverted_conv.AP_path_wb.grad)
                    binary_grads = torch.zeros_like(binary_gates.data)

                    with torch.no_grad():
                        for k in range(3):  # change len(candidate_paths) to 3
                            if scale == 0 and k == 0:
                                out_k = 0
                                grad_k = torch.sum(out_k * grad_output)
                                binary_grads[layer][scale][k] = grad_k
                                continue
                            if scale == 3 and k == 2:
                                out_k = 0
                                grad_k = torch.sum(out_k * grad_output)
                                binary_grads[layer][scale][k] = grad_k
                                continue
                            if k != active_id:  # cell_index is related to active_id
                                #if _prev_prev_c is not None:
                                out_k = candidate_paths[cell_index-active_id+k](None, _prev_c.data)
                                #else:
                                #    out_k = candidate_paths[cell_index](_prev_prev_c, _prev_c.data)
                            else:
                                out_k = _output.data


                            # todo issue in dim mismatch of grad_k
                            print(active_id, k)
                            print(out_k.shape, grad_output.shape)
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[layer][scale][k] = grad_k

                    #        print('after super_network backwards _forward: ', candidate_paths[cell_index].ops[1].mobile_inverted_conv.AP_path_wb.grad)
                    return binary_grads
                return backward
            output = SuperArchGradientFunction.apply(
                prev_prev_c, prev_c, self.fabric_path_wb,
                run_function(self.cells, self.active_index[layer][scale][0], cell_index),
                backward_function(self.cells, self.active_index[layer][scale][0], cell_index, self.fabric_path_wb)
            )
        else:
            # for None
            #print('go this')
            output = self.cells[cell_index](prev_prev_c, prev_c)
        return output
    def forward(self, x):

        size = x.size()[2:]
        intermediate_result = []
        x = self.stem0(x) # 1
        x = self.stem1(x) # 2
        x = self.stem2(x) # 4

        current_scale = 0
        prev_prev_c = None
        prev_c = x
        intermediate_result.append([-1, prev_prev_c]) # append prev_prev_c
        intermediate_result.append([0, prev_c]) # append prev_c
        #print(intermediate_result[-1][0])
        #print(self.active_index)
        for layer in range(self.nb_layers):

            next_scale = get_next_scale(self.active_index[layer][current_scale], current_scale)
            prev_prev_c, prev_c = get_prev_c(intermediate_result, next_scale)
            #if next_scale is None:
            #    print('layer {} current_scale {}'.format(layer, current_scale))
            #    print('current_scale: ', current_scale)
            #    print('current_choice: ', self.active_index[layer][current_scale])
            cell_index = get_list_index_split(layer, current_scale, next_scale)
            inter_feature = self._forward(layer, current_scale, cell_index, prev_prev_c, prev_c)
            current_scale = next_scale
            intermediate_result.pop(0)
            intermediate_result.append([next_scale, inter_feature])
        last_scale = intermediate_result[-1][0]

        if last_scale == 0:
            aspp = self.aspp4(intermediate_result[-1][1])
        elif last_scale == 1:
            aspp = self.aspp8(intermediate_result[-1][1])
        elif last_scale == 2:
            aspp = self.aspp16(intermediate_result[-1][1])
        elif last_scale == 3:
            aspp = self.aspp32(intermediate_result[-1][1])
        else:
            raise ValueError('do not support last_scale {}'.format(last_scale))

        aspp = F.interpolate(aspp, size=size, mode='bilinear', align_corners=True)

        #del intermediate_result

        return aspp

    def decode_network(self):
        # TODO: dfs and viterbi
        # dfs is re-confirmed
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

        network_weight = F.softmax(self.fabric_path_alpha, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print('\tDecode Network max_prob:', max_prob)
        return best_result

    def viterbi_search(self):

        network_space = torch.zeros((self.nb_layers, 4, 3))
        for layer in range(self.nb_layers):
            if layer == 0:
                network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (
                        2 / 3)
            elif layer == 1:
                network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (
                        2 / 3)
                network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
            elif layer == 2:
                network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (
                        2 / 3)
                network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
                network_space[layer][2] = F.softmax(self.fabric_path_alpha.data[layer][2], dim=-1)
            else:
                network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (
                        2 / 3)
                network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
                network_space[layer][2] = F.softmax(self.fabric_path_alpha.data[layer][2], dim=-1)
                network_space[layer][3][:2] = F.softmax(self.fabric_path_alpha.data[layer][3][:2], dim=-1) * (
                        2 / 3)

        prob_space = np.zeros((network_space.shape[:2]))
        path_space = np.zeros((network_space.shape[:2])).astype('int8')

        # prob_space [layer, sample] means the layer-the choice go to sample-th scale
        # network space 0 ↗, 1 →, 2 ↘  , rate means choice
        # path_space    1    0   -1      1-rate means path
        for layer in range(network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = network_space[layer][0][1]  # 0-layer go to next 0-scale prob
                prob_space[layer][1] = network_space[layer][0][2]  # 0-layer go to next 1-scale prob

                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(network_space.shape[1]):
                    if sample > layer + 1:  # control valid sample in each layer
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
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path
        output_sample = np.argmax(prob_space[-1, :], axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample  # have known tha last scale
        for i in range(1, self.nb_layers):  # get scale path according to path_space
            actual_path[-i - 1] = actual_path[-i] + path_space[self.nb_layers - i, actual_path[-i]]

        return actual_path, network_layer_to_space(actual_path)

    def genotype_decode(self):
        # use cell_arch_decode to replace genetype_decode
        raise NotImplementedError

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

    def architecture_path_parameters(self):
        # only architecture_path_parameters, within cells
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def architecture_network_parameters(self):
        # only architecture_network_parameters, network level
        for name, param in self.named_parameters():
            if 'fabric_path_alpha' in name:
                yield param

    def cell_binary_gates(self):
        # only binary gates
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield  param
    def network_binary_gates(self):
        for name, param in self.named_parameters():
            if 'fabric_path_wb' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'fabric_path_wb' in name or 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        # network weight parameters
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'fabric_path_alpha' not in name \
                    and 'AP_path_wb' not in name and 'fabric_path_wb' not in name:
                yield  param

    def architecture_parameters(self):
        # architecture_path_parameters and architecture_network_parameters
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name or 'fabric_path_alpha' in name:
                yield param

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_path_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError
        # init AP_path_alpha and fabric_path_alpha
        for param in self.architecture_network_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError
    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy


    '''
        # have some modification on redundant_modules
        
        redundant_modules should based on the selected path, rather than
        the whole super network.
    
    
    '''


    @property
    def redundant_modules(self):

        '''
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules =  module_list
        # get all the mixededge module
        return self._redundant_modules
        '''

        # proxy related to modules
        # after binarize, set redundant modules according to active_index and inactive_index

        if self._redundant_modules is None:
            print('\tset_redundant_modules')
            current_scale = 0
            module_list = []
            for layer in range(self.nb_layers):
                next_scale = get_next_scale(self.active_index[layer][current_scale][0], current_scale)
                cell_index = get_list_index_split(layer, current_scale, next_scale)

                for m in self.cells[cell_index].modules():
                    if m.__str__().startswith('MixedEdge'):
                        #print('layer {} add '.format(layer))
                        module_list.append(m)
                current_scale = next_scale
            self._redundant_modules = module_list

        return self._redundant_modules

    def reset_binary_gates(self):
        # reset each mixed operation
        self.binarize()
        for m in self.redundant_modules:
            m.binarize()

    def set_arch_param_grad(self):
        # set AP_path_alpha.grad in each mixed operation
        #self.set_fabric_arch_param_grad()
        for m in self.redundant_modules:
            m.set_arch_param_grad()

    def rescale_updated_arch_param(self):
        # rescale grad
        self.rescale_updated_fabric_arch_param()
        for m in self.redundant_modules:
            m.rescale_updated_arch_param()

    def set_chosen_op_active(self):
        # set chosen operation active for each mixed operation
        self.set_chosen_path_active()
        for m in self.redundant_modules:
            m.set_chosen_op_active()

    # TODO: do we need unused_modules_off and unused_modules_back in network-level?
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

        # TODO: control self._redundant_modules
        self._redundant_modules = None

    def set_active_via_net(self, net):
        # setting from existing net
        assert isinstance(net, SplitFabricAutoDeepLab)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self):
        # related to final objective, hardware target
        raise NotImplementedError

    def expected_flops(self):
        # related to final objective, hardware target
        raise NotImplementedError

    def module_str(self):
        # TODO: need reconfirm
        #best_result = self.decode_network()
        actual_path, _ = self.viterbi_search()
        # TODO: test viterbi algorithm
        log_str = 'Network-Level-Best-Result:\n {}\n'.format(actual_path)
        stem0_module_str = 'Stem_s{}_{}x{} ConvBnReLU'.format(self.stem0.conv.stride[0],
                                                             self.stem0.conv.kernel_size[0],
                                                             self.stem0.conv.kernel_size[1])
        stem1_module_str = 'Stem_s{}_{}x{} ConvBnReLU'.format(self.stem1.conv.stride[0],
                                                             self.stem1.conv.kernel_size[0],
                                                             self.stem1.conv.kernel_size[1])
        stem2_module_str = 'Stem_s{}_{}x{} ConvBnReLU'.format(self.stem2.conv.stride[0],
                                                             self.stem2.conv.kernel_size[0],
                                                             self.stem2.conv.kernel_size[1])
        log_str += '{}. {}\n'.format(0, stem0_module_str)
        log_str += '{}. {}\n'.format(1, stem1_module_str)
        log_str += '{}. {}\n'.format(2, stem2_module_str)

        prev_prev = -1
        prev = 0
        for layer in range(self.nb_layers):
            current_scale = prev
            next_scale = int(actual_path[layer])
            prev_prev_c = False
            if next_scale == prev_prev:
                prev_prev_c = True

            # TODO: test get_list_index_split
            index = get_list_index_split(layer, current_scale, next_scale)
            type = get_cell_decode_type(current_scale, next_scale)
            frag_cell_log = '(Layer {} Scale {} Index {})\n'\
                                .format(layer + 1, next_scale, index) + self.cells[index].module_str(prev_prev_c, type)  # each proxy cell and its mixed operations
            log_str += frag_cell_log
            prev_prev = prev
            prev = next_scale

        last_scale = int(actual_path[-1])
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

        #best_result = self.decode_network()
        actual_path, _ = self.viterbi_search()
        print(actual_path)
        #assert len(best_result) == self.run_config.nb_layers, 'Error in length of best_result'
        assert len(actual_path) == self.nb_layers, 'Error in actual_path of net.get_flops'
        flops = 0.
        flop_stem0 = count_normal_conv_flop(self.stem0.conv, x)
        x = self.stem0(x)
        flop_stem1 = count_normal_conv_flop(self.stem1.conv, x)
        x = self.stem1(x)
        flop_stem2 = count_normal_conv_flop(self.stem2.conv, x)
        x = self.stem2(x)

        prev_scale = 0
        inter_features = [[0, None], [0, x]]
        for layer in range(self.nb_layers):
            current_scale = prev_scale
            next_scale = int(actual_path[layer])
            prev_prev_c, prev_c = get_prev_c(inter_features, next_scale)
            #type = get_cell_decode_type(current_scale, next_scale)
            index = get_list_index_split(layer, current_scale, next_scale)
            #print('cell_index:  ',index)
            #print('current_scale: ',current_scale)
            #print('next_scale: ', next_scale)
            #print('index: ', index)
            #print(prev_prev_c, prev_c)
            frag_flop, out = self.cells[index].get_flops(prev_prev_c, prev_c)
            flops = flops + frag_flop
            prev_scale = next_scale
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
            raise ValueError('invalid scale choice of {}'.format(last_scale))

        return flop_stem0 + flop_stem1 + flop_stem2 + flops + flop_aspp, output

    def convert_to_normal_net(self):
        # not used
        # TODO: pay attention to call this method, each MixedEdge will be replaced by chosen layer, module_str() will raise errors
        # network level architecture is obtained by decoder, cell level architecture is obtained by this.
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
        return SplitFabricAutoDeepLab(self.run_config, self.conv_candidates)

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

        prev_scale = 0
        for i in range(self.nb_layers):
            current_scale = prev_scale
            next_scale = actual_path[i]
            cell_index = get_list_index_split(i, current_scale ,next_scale)
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

            prev_scale = next_scale
            gene.append(_gene)

        #print(len(gene))
        assert len(gene) == 12, 'Error in network_arch_cell_arch_decode'
        return actual_path, network_space, np.array(gene)

class SuperArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prev_prev_c, prev_c, binary_gates, run_func, backward_func):
        #print('super_network _forward function')
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        if prev_prev_c is not None:
            detached_prev_prev_c = detach_variable(prev_prev_c)
        else: detached_prev_prev_c = None
        detached_prev_c = detach_variable(prev_c)

        with torch.enable_grad():
            output = run_func(detached_prev_prev_c, detached_prev_c)
        ctx.save_for_backward(detached_prev_prev_c , detached_prev_c, output)

        return output.data

    @staticmethod
    def backward(ctx, grad_outputs):
        #print('super_network _backward function')
        detached_prev_prev_c, detached_prev_c, output = ctx.saved_tensors
        #detached_prev_c, output = ctx.saved_tensors
        #output = ctx.saved_tensors
        # TODO: needs confirm
        if detached_prev_prev_c is not None:
            grad_prev_prev_c = torch.autograd.grad(output, detached_prev_prev_c, grad_outputs, only_inputs=True, retain_graph=True)
        else: grad_prev_prev_c = None
        grad_prev_c = torch.autograd.grad(output, detached_prev_c, grad_outputs, only_inputs=True, retain_graph=False)
        if detached_prev_prev_c is not None:
            binary_grads = ctx.backward_func(detached_prev_prev_c.data, detached_prev_c.data, output.data, grad_outputs.data)
        else:
            binary_grads = ctx.backward_func(detached_prev_prev_c, detached_prev_c.data, output.data, grad_outputs.data)
        #print('in super_network backward:', binary_grads)
        return  grad_prev_prev_c[0] if grad_prev_prev_c is not None else None, grad_prev_c[0], binary_grads, None, None

