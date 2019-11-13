import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

from models.cell import Split_Cell

from queue import Queue

from genotype import PRIMITIVES, Genotype
from modules.operations import *
from run_manager import *
from nas_manager import *
from utils.common import save_inter_tensor
from utils.common import get_next_scale, get_list_index, get_prev_c


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


        # TODO: active_index and inactive_index should not be list, but tensors
        # and list_index is calculate by layer and scales
        self.active_index = torch.zeros(self.nb_layers, 4, 1) # active_index always only has one path
        self.inactive_index = None # inactive_index.shape is not fixed, 'two' mode is one and other mode are two

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
                cell1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=prev_c, prev_prev_c=None, types='same', )
                cell2 = Split_Cell(self.run_config, self.conv_candidates, 8, prev_c=prev_c, prev_prev_c=None, types='reduction')
                self.cells += [cell1] # 0
                self.cells += [cell2] # 1
            elif i == 1:
                cell1_1 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=128, types='same')
                cell1_2 = Split_Cell(self.run_config, self.conv_candidates, 4, prev_c=-1, prev_prev_c=128, types='up')
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

    @property
    def n_choices(self):
        return 3

    @property
    def probs_over_paths(self):
        # confirmed
        probs = F.softmax(self.fabric_path_alpha, dim=-1)
        return probs

    @property
    def chosen_index(self):
        # confirmed
        probs = self.probs_over_ops.data#.cpu().numpy()
        index = torch.argmax(probs, dim=-1, keepdim=True)#np.argmax(probs, axis=-1) # shape as [self.nb_layers, scales]
        # shape as [12, 4, 1]
        return index, torch.gather(probs, -1, index)

    @property
    def random_index(self):
        # TODO: here can not perform random index, related to spatial size
        active_index = np.zeros((self.nb_layers, 4, 1))
        random_result = []
        last = 0
        layer = 0
        def random_dfs(layer, random_result, last):
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
        random_dfs(layer, random_result, last)
        print('random_path_result:', random_result)
        return active_index, random_result

    @property
    def active_fabric_path(self):
        # confirmed
        return self.active_index # tensor with shape [12,4,1]

    def set_chosen_path_active(self):
        # confirmed
        index, _ = self.chosen_index
        # index with shape [12, 4, 1]
        self.active_index = index
        for l in self.nb_layers:
            for s in range(4):
                # self.inactive_index is also tensor, with shape [12, 4, 2]
                self.inactive_index[l][s] = torch.from_numpy(np.array([_i for _i in range(0, index[l][s][0])] + \
                                                                      [_i for _i in range(index[l][s][0]+1, self.n_choices)]))
    def binarize(self):
        # confirmed
        # binarize should apply on each node in fabric, should apply to the whole fabric
        # network, rather than each node separable
        self.log_prob = None
        probs = self.probs_over_paths  # shape as [12, 4, 3]
        self.fabric_path_wb.data.zero_()
        self.current_prob_over_paths = torch.zeros_like(probs)

        # argument layer, last=0
        def _binarize(layer, scale):
            print('\t->Binarizing layer {} scale {}'.format(layer, scale)) # used for test method
            if SplitFabricAutoDeepLab.MODE == 'two':
                if scale == 0:
                    sample_paths = torch.multinomial(probs.data[layer][scale][1:], 2, replacement=False)
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
                active_path = sample_paths[c].item()
                inactive_path = sample_paths[1 - c].item()

                self.active_index[layer][scale][0] = active_path
                self.inactive_index[layer][scale] = inactive_path
                self.fabric_path_wb.data[layer][scale][active_path] = 1.0
            else:
                if scale == 0:
                    sample_path = torch.multinomial(probs.data[layer][scale][1:], 1)[0].item()
                elif scale == 3:
                    sample_path = torch.multinomial(probs.data[layer][scale][:2], 1)[0].item()
                else:
                    sample_path = torch.multinomial(probs.data[layer][scale], 1)[0].item()
                self.active_index[layer][scale][0] = sample_path
                self.inactive_index[layer][scale] = torch.from_numpy(np.array(
                    [[_i for _i in range(0, sample_path)] + [_i for _i in range(sample_path + 1, self.n_choices)]]
                ))
                self.current_prob_over_paths[layer][scale] = probs[layer][scale]
                self.log_prob[layer][scale][0] = torch.log(probs[layer][scale][sample_path])
                self.fabric_path_wb.data[layer][scale][sample_path] = 1.0

            last_scale = scale
            if self.active_index[layer][scale][0] == 0:
                if scale == 0:
                    raise ValueError('scale 0 do not support IncreasedOperation')
                last_scale = scale - 1
            elif self.active_index[layer][scale][0] == 1:
                last_scale = scale
            elif self.active_index[layer][scale][0] == 2:
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
                    continue
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
        '''
        # TODO: consider the relationship between layers
        if SplitFabricAutoDeepLab.MODE == 'two':
            probs = probs.data.view(-1, 3)
            sample_path = torch.multinomial(probs, 2, replacement=False) # multinomial with 2d tensor weight
            sample_path = sample_path.view(self.nb_layers, 4, 2) # [12, 4, 2]
            probs_slice = F.softmax(torch.gather(self.fabric_path_alpha, -1, sample_path), -1) # softmax on chosen two paths
            self.current_prob_over_paths = torch.zeros_like(probs) # [12, 4, 3]

            for l in range(self.nb_layers):
                for s in range(4):
                    tmp_index = sample_path[l][s]
                    # probs_slice [12, 4, 2]
                    self.current_prob_over_paths[l][s][tmp_index] = probs_slice[l][s]

            # get one path from total two
            c = torch.multinomial(probs_slice.view(-1, 2), 1)[0] # c.shape [self.nb_layers * 4, 1]
            c = c.view(self.nb_layers, 4, 1) # [12, 4, 1]
            active_op = torch.gather(sample_path, -1, c) # select [12, 4, 1] from [12, 4, 2]
            inactive_op = torch.gather(sample_path, -1, 1-c) #
            self.active_index = active_op
            self.inactive_index = inactive_op
            for l in range(self.nb_layers):
                for s in range(4):
                    tmp_index = active_op[l][s]
                    self.fabric_path_wb.data[l][s][tmp_index] = 1.0
        else:
            self.inactive_index = torch.zeros(self.nb_layers, 4, 2)
            sample_path = torch.multinomial(probs.data.view(-1, 3), 1)
            sample_path = sample_path.view(self.nb_layers, 4, 1) # shape as [12, 4, 1]
            self.active_index = sample_path
            for l in range(self.nb_layers):
                for s in range(4):
                    self.inactive_index[l][s] = torch.from_numpy(np.array([_i for _i in range(0, sample_path[l][s][0])] + \
                                                                          [_i for _i in range(sample_path[l][s][0]+1, self.n_choices)]))

            self.log_prob = torch.log(torch.gather(probs.data, -1, sample_path)) # log prob of the sampled path
            self.current_prob_over_paths = probs
            self.fabric_path_wb.data[sample_path] = 1.0
            for l in range(self.nb_layers):
                for s in range(4):
                    tmp_index = sample_path[l][s][0]
                    self.fabric_path_wb.data[l][s][tmp_index] = 1.0
                    '''
        # TODO: avoid over-regularization

    def set_fabric_arch_param_grad(self):
        # confirmed
        fabric_binary_grads = self.fabric_path_wb.grad.data # shape as [12, 4, 3]
        if self.fabric_path_alpha.grad is None:
            self.fabric_path_alpha.grad = torch.zeros_like(self.fabric_path_alpha.data)

        if SplitFabricAutoDeepLab.MODE == 'two':
            involved_idx = torch.concat([self.active_index, self.inactive_index], dim=-1) # [12, 4, 1] concat [12, 4, 1]
            probs_slice = F.softmax(torch.gather(self.fabric_path_alpha, -1, involved_idx), -1) # shape [12, 4, 2]
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
            self.tmp_active_alphas = torch.gather(self.fabric_path_alpha, -1, self.active_index) # shape as [12, 4, 1]
            self.tmp_inactive_alphas = torch.gather(self.fabric_path_alpha, -1, self.inactive_index) # shape as [12, 4, 1]
        else:
            probs = self.probs_over_paths.data
            for l in range(self.nb_layers):
                for s in range(4):
                    for i in range(self.n_choices):
                        for j in range(self.n_choices):
                            self.fabric_path_alpha.grad.data[i] += fabric_binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])

        return

    def rescale_updated_fabric_arch_param(self):
        # only under the case of 'two'
        involved_index = torch.concat([self.active_index, self.inactive_index], dim=-1) # [12, 4, 1] concat [12, 4, 1]
        old_alphas = torch.concat([self.tmp_active_alphas, self.tmp_inactive_alphas], dim=-1) # shape as [12, 4, 2]
        new_alphas = torch.gather(self.fabric_path_alpha.data, -1, involved_index, dim=-1) # shape as [12, 4, 2]

        offset = None # shape as [self.nb_layers, 4, 1]
        for l in range(self.nb_layers):
            for s in range(4):
                offset[l][s] = math.log(sum([math.exp(alpha) for alpha in new_alphas[l][s]]) / sum([math.exp(alpha) for alpha in old_alphas[l][s]]))

        for l in range(self.nb_layers):
            for s in range(4):
                for index in involved_index[l][s]:
                    self.fabric_path_alpha.data[l][s][index] -= offset[l][s][0]

    @property
    def config(self):
        raise ValueError('not needed')
    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def forward(self, x):

        size = x.size()[2:]
        # keep two value for each scale
        # TODO: get rid of, when only have one path
        #scale4 = []
        #scale8 = []
        #scale16 = []
        #scale32 = []

        intermediate_result = []

        x = self.stem0(x) # 1
        x = self.stem1(x) # 2
        x = self.stem2(x) # 4
        #save_inter_tensor(scale4, x)

        def _forward(layer, scale, base, prev_c, prev_prev_c):
            if SplitFabricAutoDeepLab.MODE == 'full' or SplitFabricAutoDeepLab.MODE == 'two':
                output = 0
                for _i in self.active_index[layer][scale]:
                    output_i = self.cells[base+_i](prev_prev_c, prev_c)
                    output = output + self.fabric_path_wb[layer][scale][_i] * output_i
                for _i in self.inactive_index[layer][scale]:
                    output_i = self.cells[base+_i](prev_prev_c, prev_c)
                    output = output + self.fabric_path_wb[layer][scale][_i] * output_i.detach()
            elif SplitFabricAutoDeepLab.MODE == 'full_v2':
                def run_function(candidate_paths, active_id):
                    def fforward(_prev_prev_c, _prev_c):
                        if _prev_prev_c is not None:
                            return candidate_paths[base+active_id](_prev_prev_c.data, _prev_c.data)
                        else:
                            return candidate_paths[base+active_id](_prev_prev_c, _prev_c.data)
                    return fforward
                def backward_function(candidate_paths, active_id, binary_gates):
                    def backward(_prev_prev_c, _prev_c, _output, grad_output):
                        binary_grads = torch.zeros_like(binary_gates)
                        with torch.no_grad():
                            for k in range(3): # change len(candidate_paths) to 3
                                if k != active_id:
                                    out_k = candidate_paths[base+active_id](_prev_prev_c, _prev_c)
                                else:
                                    out_k = _output.data
                                grad_k = torch.sum(out_k * grad_output)
                                binary_grads[layer][scale][k] = grad_k
                        return binary_grads
                    return backward
                output = ArchGradientFunction.apply(
                    x, self.fabric_path_wb, run_function(self.cells, self.active_index[layer][scale][0]),
                    backward_function(self.cells, self.active_index[layer][scale][0], self.fabric_path_wb)
                )
            else:
                output = self.cells[base+self.active_index[layer][scale][0]](x)
            return output

        scale = 0
        prev_prev_c = None
        prev_c = x
        intermediate_result.append([0, prev_prev_c]) # append prev_prev_c
        intermediate_result.append([0, prev_c]) # append prev_c
        for layer in range(self.nb_layers):
            if layer == 0:
                count = 0
                if self.active_index[layer][scale][0] == 1:
                    prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                    inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                    scale = scale
                    intermediate_result.pop()
                    intermediate_result.append([scale, inter_feature])
                elif self.active_index[layer][scale][0] == 2:
                    prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                    inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                    scale = scale + 1
                    intermediate_result.pop()
                    intermediate_result.append([scale, inter_feature])
            elif layer == 1:
                if scale == 0:
                    count = 2
                    if self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 1:
                    count = 4
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale-1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
            elif layer == 2:
                if scale == 0:
                    count = 7
                    if self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 1:
                    count = 9
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale-1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 2:
                    count = 12
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale -1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
            else:
                if scale == 0:
                    count = (layer-3)*10 + 15
                    if self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale+1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 1:
                    count = (layer-3)*10 + 17
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale-1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale + 1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 2:
                    count = (layer-3)*10 + 20
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale - 1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 2:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale + 1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale + 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                elif scale == 3:
                    count = (layer-3)*10 + 23
                    if self.active_index[layer][scale][0] == 0:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale - 1)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale - 1
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])
                    elif self.active_index[layer][scale][0] == 1:
                        prev_prev_c, prev_c = get_prev_c(intermediate_result, scale)
                        inter_feature = _forward(layer, scale, count, prev_prev_c, prev_c)
                        scale = scale
                        intermediate_result.pop()
                        intermediate_result.append([scale, inter_feature])

        scale = intermediate_result[-1][0]
        aspp = None
        if scale == 0:
            aspp = self.aspp4(intermediate_result[-1][1])
        elif scale == 1:
            aspp = self.aspp8(intermediate_result[-1][1])
        elif scale == 2:
            aspp = self.aspp16(intermediate_result[-1][1])
        elif scale == 3:
            aspp = self.aspp32(intermediate_result[-1][1])

        aspp = F.interpolate(aspp, size=size, mode='bilinear', align_corners=True)
        return aspp

    def decode_network(self):
        # TODO: dfs and viterbi
        # dfs is re-confirmed
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

        network_weight = F.softmax(self.fabric_path_alpha, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print(max_prop)
        return best_result

    def viterbi_search(self):

        self.network_space = torch.zeros(self.nb_layers, 4, 3)
        for layer in range(self.nb_layers):
            if layer == 0:
                self.network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (2/3)
            elif layer == 1:
                self.network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
            elif layer == 2:
                self.network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self.fabric_path_alpha.data[layer][2], dim=-1)
            else:
                self.network_space[layer][0][1:] = F.softmax(self.fabric_path_alpha.data[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self.fabric_path_alpha.data[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self.fabric_path_alpha.data[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self.fabric_path_alpha.data[layer][3][:2], dim=-1) * (2/3)

    def genotype_decode(self):
        # proxy-cell does not need genotype parser any more
        # raise NotImplementedError
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
            if 'fabric_path_alpha' in name:
                yield param

    def binary_gates(self):
        # only binary gates
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield  param

    def weight_parameters(self):
        # network weight parameters
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'fabric_path_alpha' not in name and 'AP_path_wb' not in name:
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

    def reset_binary_gates(self):
        # reset each mixed operation
        for m in self.redundant_modules:
            m.binarize()
    def set_arch_param_grad(self):
        # set AP_path_alpha.grad in each mixed operation
        for m in self.redundant_modules:
            m.set_arch_param_grad()

    def rescale_updated_arch_param(self):
        # rescale grad
        for m in self.redundant_modules:
            m.rescale_updated_arch_param()

    def set_chosen_op_active(self):
        # set chosen operation active for each mixed operation
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

    def set_active_via_net(self, net):
        # setting from existing net
        assert isinstance(net, SplitFabricAutoDeepLab)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self):
        raise NotImplementedError

    def expected_flops(self):
        raise NotImplementedError

    def convert_to_normal_net(self):
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
        return SplitFabricAutoDeepLab(self.run_config, self.arch_search_config, self.conv_candidates)


class ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prev_prev_c, prev_c, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_prev_prev_c = detach_variable(prev_prev_c)
        detached_prev_c = detach_variable(prev_c)

        with torch.enable_grad():
            output = run_func(detached_prev_prev_c, detached_prev_c)
        ctx.save_for_backward(detached_prev_prev_c, detached_prev_c, output)
        return output.data
    @staticmethod
    def backward(ctx, grad_outputs):
        detached_prev_prev_c, detached_prev_c, output = ctx.saved_tensors
        grad_prev_prev_c = torch.autograd.grad(output, detached_prev_prev_c, grad_outputs, only_inputs=True)
        grad_prev_c = torch.autograd.grad(output, detached_prev_c, grad_outputs, only_inputs=True)
        binary_grads = ctx.backward_func(detached_prev_prev_c, detached_prev_c, output.data, grad_outputs.data)

        return  grad_prev_prev_c[0], grad_prev_c[0], binary_grads, None, None


