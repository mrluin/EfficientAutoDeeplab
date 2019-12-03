# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gumbel_cells import counter, build_candidate_ops, MixedOp
from modules.my_modules import MyModule
from modules.operations import FactorizedIncrease, ConvLayer, FactorizedReduce, Identity, MobileInvertedResidualBlock, MBInvertedConvLayer

'''
# CounterCell is used to test model capacity, only has prev_input, rather than prev-prev-input and prev-input
'''

class CounterCell(MyModule):
    def __init__(self, layer, filter_multiplier, block_multiplier, steps,
                 prev_scale, scale, search_space, affine=True):
        super(CounterCell, self).__init__()
        self.layer = layer
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.prev_scale = prev_scale
        self.scale = scale
        self.search_space = search_space
        self.conv_candidates = counter
        self.affine = affine

        self.total_nodes = 1 + self.steps
        index2channel = {
            0: 32,
            1: 64,
            2: 128,
            3: 256,
        }
        self.outc = index2channel[self.scale]
        if index2channel.get(self.prev_scale) is not None:
            self.prev_c = index2channel[self.prev_scale]
            if self.prev_scale == self.scale + 1: # up
                self.preprocess1 = FactorizedIncrease(self.prev_c, self.outc, affine=affine)
            elif self.prev_scale == self.scale: # same
                self.preprocess1 = ConvLayer(self.prev_c, self.outc, 1, 1, False, affine=affine)
            elif self.prev_scale == self.scale - 1: # down
                self.preprocess1 = FactorizedReduce(self.prev_c, self.outc, affine=affine)
            else: raise ValueError('relation error in prev_scale and current scale layer-scale {:}-{:}'.format(self.layer, self.scale))
        else:
            # level4-node1 and level8-node1
            self.prev_c = prev_scale
            if self.layer == 0 and self.scale == 0: # level4-node1 using stem1 output as prev_input, with stride1
                self.preprocess1 = ConvLayer(self.prev_c, self.outc, 1, 1, False, affine=affine)
            elif self.layer == 0 and self.scale == 1: # level8-node1 using stem1 output as prev_input, with stride2
                self.preprocess1 = FactorizedReduce(self.prev_c, self.outc, affine=affine)
            else: raise ValueError('relation error between prev_scale and current scale layer-scale {:}-{:}'.format(self.layer, self.scale))

        self.ops = nn.ModuleDict()
        for i in range(1, self.total_nodes):
            for j in range(i):
                edge_str = '{:}<-{:}'.format(i, j)
                mobile_inverted_conv = MixedOp(
                        build_candidate_ops(self.conv_candidates,
                                            in_channels=self.outc, out_channels=self.outc, stride=1,
                                            ops_order='act_weight_bn',
                                            affine=self.affine))  # normal MixedOp, ModuleList with weight
                #shortcut = Identity(self.outc, self.outc)
                #inverted_residual_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                self.ops[edge_str] = mobile_inverted_conv

        self.finalconv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)
        self.edge_keys = sorted(list(self.ops.keys()))  # 'sorted by {:}<-{:}'
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}  # {:}<-{:} : index
        self.nb_edges = len(self.ops)
        self.cell_arch_parameters = nn.Parameter(1e-3 * torch.randn(self.nb_edges, self.n_choice))
    @property
    def n_choice(self):
        return len(self.conv_candidates)

    def forward(self, x):
        x = self.preprocess1(x)
        states = [x]
        for i in range(2, self.total_nodes): # current node index
            new_states = []
            for j in range(i): # previous nodes
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]
                # self.ops has None operation when self.prev_prev_scale is none and j == 0
                if self.ops[edge_str] is None:
                    assert related_hidden is None, \
                        'None operation branch and None prev_prev_input are inconsistent'
                    continue
                new_state = self.ops[edge_str].forward_single(related_hidden)
                new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        # concatenation of last self.steps intermediate tensors
        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.finalconv1x1(concat_feature)

        return concat_feature